import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool, global_max_pool
import torch.nn.functional as F
import numpy as np
class HydraAttention(nn.Module):
    # implemented from https://arxiv.org/pdf/2209.07484.pdf
    def __init__(self, embedding_dim, output_layer='linear', dropout=0.0):
        super(HydraAttention, self).__init__()
        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3)
        self.out = nn.Linear(embedding_dim, embedding_dim) if output_layer == 'linear' else nn.Identity()
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x, mask=None): # x is (batchsize, N, M)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        if mask is not None:
            k = k.masked_fill(mask.unsqueeze(-1), 0)
        kvw = k * v
        if self.dropout.p > 0:
            kvw = self.dropout(kvw.transpose(-1, -2)).transpose(-1, -2) # dropout in seq dimension 
        out = kvw.sum(dim=-2, keepdim=True) * q
        return self.out(out)

class TransformerBlock(nn.Module):
    def __init__(self, input_dim, embedding_dim, output_layer='linear', dropout=0.0):
        super(TransformerBlock, self).__init__()
        self.lin1 = nn.Linear(input_dim, embedding_dim)
        self.lin2 = nn.Linear(embedding_dim, embedding_dim)
        self.lin3 = nn.Linear(embedding_dim, embedding_dim)
        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attention = HydraAttention(embedding_dim, output_layer, dropout)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
    def forward(self, x, mask=None):
        x = self.norm1(self.lin1(x))
        x_ = self.norm2(self.attention(x_, mask))
        x_ = self.activation(self.lin2(x_))
        x_ = self.activation(self.lin3(x_)) + x
        return x_
    
class ConditionalFlowModel(nn.Module):
# Modified GFLowNet model for DAG generation conditioned on a given observation
# The model takes as input an adjacency matrix of the current icomplete DAG, 
# the current observation and a mask for the valid actions.
# Two heads are used: one to compute the probability to stop the sampling process,
# and another to compute the logits of transitioning to a new graph, given that we didn't stop.
    def __init__(self, net_config):
        super(ConditionalFlowModel).__init__()
        self.net_config = net_config

        # Edge embedding, edges are pairs of indices (source, target)
        self.embedding = nn.Embedding(2 * net_config['num_variables'], net_config['embedding_dim']) 
        self.embedding.weight.data.uniform_(-1, 1)

        # Observation encoder, depends on observation type (image, grid tensor or tuple)
        if net_config['obs_type'] == 'image':
            # Convolutional encoder
            self.obs_encoder = nn.ModuleList(
                [nn.Conv2d(net_config['obs_channels'], net_config['obs_channels'], 3, padding=1),
                nn.BatchNorm2d(net_config['obs_channels']),
                nn.ReLU()]*net_config['num_conv_layers']
            )
            self.obs_encoder.append(nn.Flatten())
            self.obs_encoder.append(
                nn.Linear(net_config['obs_channels'] * net_config['obs_size']**2, 
                          net_config['embedding_dim'])
                          )
        elif net_config['obs_type'] == 'grid':
            # Linear encoder
            self.obs_encoder = nn.Sequential(
                nn.Linear(net_config['obs_size']**2, net_config['embedding_dim']), 
                nn.ReLU(),
                nn.Linear(net_config['embedding_dim'], net_config['embedding_dim']),
                nn.ReLU(),
                nn.Linear(net_config['embedding_dim'], net_config['embedding_dim']),
                nn.ReLU()
            )
        elif net_config['obs_type'] == 'tuple':
            # Linear encoder
            self.obs_encoder = nn.Sequential(
                nn.Linear(net_config['obs_size'], net_config['embedding_dim']), 
                nn.ReLU(),
                nn.Linear(net_config['embedding_dim'], net_config['embedding_dim']),
                nn.ReLU(),
                nn.Linear(net_config['embedding_dim'], net_config['embedding_dim']),
                nn.ReLU()
            )
        else:
            raise ValueError('Invalid observation type')



        # Common transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(2*net_config['embedding_dim'], 
                                net_config['hidden_dim'], 
                                net_config['dropout']) 
                                for _ in range(net_config['num_transformer_blocks'])]
                            )

        # Heads
        self.stop_head = nn.ModuleList(
            [TransformerBlock(net_config['hidden_dim'],
                                net_config['hidden_dim'],
                                net_config['dropout'])
                                for _ in range(net_config['num_tb_stop_head'])]
        )
        self.stop_head.append(nn.Linear(net_config['hidden_dim'], 1))
        self.stop_head.append(nn.Sigmoid())

        self.transition_head = nn.ModuleList(
            [TransformerBlock(net_config['hidden_dim'],
                                net_config['hidden_dim'],
                                net_config['dropout'])
                                for _ in range(net_config['num_tb_transition_head'])]
        )
        self.transition_head.append(nn.Linear(net_config['hidden_dim'], net_config['num_variables'] * (net_config['num_variables'] - 1)))
        self.transition_head.append(nn.Softmax(dim=-1))

    def forward(self, adj, obs, mask):
        # create edges as (source, target) pairs
        edges = torch.stack(torch.meshgrid(torch.arange(adj.shape[1]), torch.arange(adj.shape[2])), dim=-1).reshape(-1, 2)
        # embed edges
        edges = self.embedding(edges)
        # embed observation
        obs = self.obs_encoder(obs)
        # concatenate edges and observation
        x = torch.cat([edges, obs], dim=-1)
        # apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        # get stop probability
        stop_x = x
        for block in self.stop_head:
            stop_x = block(stop_x)
        # get transition logits
        logits_x = x
        for block in self.transition_head:
            logits_x = block(logits_x)
        # reshape logits
        logits_x = logits_x.reshape(-1, adj.shape[1], adj.shape[2])
        # apply mask
        logits_x = logits_x * mask
        return stop_x, logits_x
    
class GAT(torch.nn.Module):
    # Graph attention block
    def __init__(self, in_dim, out_dim, hid, in_head, out_head, dropout):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_dim, self.hid, heads=self.in_head, dropout=dropout)
        self.conv2 = GATConv(self.hid*self.in_head, out_dim, concat=False,
                             heads=self.out_head, dropout=dropout)
def forward(self,x, edge_index):
        # x: node feature matrix of shape [num_nodes, in_channels]
        # edge_index: graph connectivity in COO format with shape [2, num_edges]
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x,F.log_softmax(x, dim=1)
class GraphScorer(nn.Module):
    # Learns to score partial DAGs based on either the reconstruction accuracy or the reward associated with using the DAG by the actor
    def __init__(self, net_config):
        super(GraphScorer, self).__init__()
        self.net_config = net_config
        # local scorer to predict the score associated with a node in the DAG given its parents
        self.local_scorer = nn.ModuleList(
            [TransformerBlock(2*net_config['embedding_dim'],
                                net_config['hidden_dim'],
                                net_config['dropout'])
                                for _ in range(net_config['num_tb_local_scorer'])]
        )
        
        # Embedding layer for the nodes
        self.embedding = nn.Embedding(net_config['num_variables'], net_config['embedding_dim'])
        
    def loss_rec(self, adj, true_loss_fun, device='cpu'):
        # Adj: adjacency matrix of shape [batch_size, num_variables, num_variables]
        # true_loss_fun: function that computes the value loss (critic/reconstruction) given an adjacency matrix
        # device: 'cpu' or cuda device
        # Loop thorugh the nodes and compute the score
        # reconstruction loss with full graph
        rec_loss = true_loss_fun(adj, device)
        score_loss = 0
        global_score = 0
        for i in range(adj.shape[1]):
            # get parents
            parents = adj[:, i, :].nonzero()
            # get node embedding
            node_embedding = self.embedding(torch.tensor([i]).to(device))
            # get parent embeddings
            parent_embeddings = self.embedding(parents[:, 1]).to(device)
            # concatenate node embedding and parent embeddings
            x = torch.cat([node_embedding, parent_embeddings], dim=-1)
            # apply transformer blocks
            for block in self.local_scorer:
                x = block(x)
            # get score
            global_score += x
            # reconstruction loss without node i
            adj_i = adj.clone()
            adj_i[:, i, :] = 0
            adj_i[:, :, i] = 0
            rec_loss_i = true_loss_fun(adj_i, device)
            delta_i = F.mse_loss(rec_loss.detach(), rec_loss_i)
            score_loss += delta_i
        
        score_loss += F.mse_loss(global_score, rec_loss.detach())
        
        return score_loss

class GraphGenerator(nn.module):
    # Learns to generate DAGs conditioned on observation
    def __init__(self, net_config, mode='rec'):
        self.net_config = net_config
        self.mode = mode
        
        self.gflow = ConditionalFlowModel(net_config)
        self.scorer = GraphScorer(net_config)
        
        # Graph attention for embedding, finish at net_config['graph_embedding_dim_sq']
        self.graph_encoding = nn.ModuleList(
            [GAT(net_config['embedding_dim'], net_config['graph_embedding_dim_sq'], net_config['hidden_dim'], 1, net_config['num_heads'], net_config['dropout'])] +
            [GAT(net_config['graph_embedding_dim_sq'], net_config['graph_embedding_dim_sq'], net_config['hidden_dim'], net_config['num_heads'], net_config['num_heads'], net_config['dropout']) for _ in range(net_config['num_gat']-1)]
            [nn.Flatten(), nn.Linear(net_config['graph_embedding_dim_sq']*net_config['num_heads'], net_config['graph_embedding_dim_sq'])]
        )
        if mode == 'rec':
            # latent graph rec to reconstructed observation by succesivly bigger convolutions, until obs_size reached
            # enough deconvolutions to get back to obs_size from graph_embedding_dim_sq
            self.value_model = nn.ModuleList(
                    [nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1, output_padding=1)]*int(np.log2(net_config['obs_size'])-np.log2(net_config['graph_embedding_dim_sq'])) +
                    [nn.Conv2d(1, 1, 3, padding=1)]
            )
        elif mode == 'reward':
            # latent graph rec to reward
            self.value_model = nn.ModuleList(
                [nn.Linear(net_config['graph_embedding_dim_sq'], net_config['graph_embedding_dim_sq']) for _ in range(net_config['num_critic'])] +
                [nn.Linear(net_config['graph_embedding_dim_sq'], 1)]
            )

    def train_step_value_model(self, episodes, mode, optim, device):
        graph = self.gflow(...)
        graph_embedding = self.graph_encoding(graph)
        if mode == 'rec':
            obs_rec = self.value_model(graph_embedding)
            rec_loss = F.mse_loss(episodes["obs"], obs_rec)
            rec_loss.backward()
            optim.step()
            optim.zero_grad()

        elif mode == 'reward':
            pred_reward = self.value_model(graph_embedding)
            critic_loss = F.mse_loss(episodes["reward"], pred_reward)
            critic_loss.backward()
            optim.step()
            optim.zero_grad()
         
