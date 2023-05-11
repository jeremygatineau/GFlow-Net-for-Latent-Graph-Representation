import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from torch_geometric.nn import GATConv, GPSConv, GINEConv, global_mean_pool
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data, Batch
from util import get_mask
import networkx as nx
class HydraAttention(nn.Module):
    # implemented from https://arxiv.org/pdf/2209.07484.pdf
    def __init__(self, embedding_dim, out_dim, output_layer='linear', dropout=0.0):
        super(HydraAttention, self).__init__()
        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3)
        self.out = nn.Linear(embedding_dim, out_dim) if output_layer == 'linear' else nn.Identity()
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

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, out_dim, output_layer='linear', dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.attentions = nn.ModuleList([HydraAttention(embedding_dim, out_dim, output_layer, dropout) for _ in range(num_heads)])
        self.out = nn.Linear(out_dim * num_heads, out_dim) if output_layer == 'linear' else nn.Identity()
        
    def forward(self, x, mask=None):
        return self.out(torch.cat([attn(x, mask) for attn in self.attentions], dim=-1))

class TransformerBlock(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_heads, out_dim, mix_dim=None, output_layer='linear', dropout=0.0, residual=True):
        super(TransformerBlock, self).__init__()
        self.residual = residual
        if residual and input_dim != out_dim:
            self.residual_proj = nn.Linear(input_dim, out_dim)
        else:
            self.residual_proj = nn.Identity()
        self.lin1 = nn.Linear(input_dim, embedding_dim)
        self.lin2 = nn.Linear(embedding_dim, embedding_dim)
        self.lin3 = nn.Linear(out_dim, out_dim)
        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadAttention(embedding_dim, num_heads, out_dim, output_layer, dropout)
        self.norm2 = nn.LayerNorm(out_dim)
        self.mix_layer = nn.Linear(input_dim+mix_dim, input_dim) if mix_dim is not None else nn.Identity()
    def forward(self, x, mix=None, mask=None):
        # x is (B, seq_len, input_dim), each seq element is an edge as (src_emb, dst_emb) with positional encoding added
        # mix is (B, mix_dim)
        # mask is (B, seq_len) with 1s for valid edges and 0s for invalid edges

        # apply the layers to each edge in the sequence
        
        # mix the input with the mix vector (repeated to match shape)
        
        if self.residual:
            res = self.residual_proj(x)
        if mix is not None:
            x_ = self.mix_layer(torch.cat([x, mix.unsqueeze(1).repeat(1, x.shape[1], 1)], dim=-1))
        else:
            x_ = x
        x_1 = self.norm1(self.activation(self.lin1(x_)))
        x_2= self.activation(self.lin2(x_1))
        x_3 = self.activation(self.lin3(self.norm2(self.attention(x_2, mask))))
        if self.residual:
            x_4 = x_3 + res
        else:
            x_4 = x_3
        return x_4

class ConditionalFlowModel(nn.Module):
    # Modified GFLowNet model for DAG generation conditioned on a given observation
    # The model takes as input an adjacency matrix of the current icomplete DAG, 
    # the current observation and a mask for the valid actions.
    # Two heads are used: one to compute the probability to stop the sampling process,
    # and another to compute the logits of transitioning to a new graph, given that we didn't stop.
    def __init__(self, net_config):
        super(ConditionalFlowModel, self).__init__()
        self.net_config = net_config

        # Edge embedding, edges are pairs of indices (source, target)
        self.embedding = nn.Embedding(2 * net_config['num_variables'], net_config['embedding_dim']) 
        self.embedding.weight.data.uniform_(-1, 1)

        # Observation encoder, depends on observation type (image, grid tensor or tuple)
        if net_config['obs_type'] == 'image':
            # Convolutional encoder
            layers = []
            for _ in range(net_config['num_conv_layers']):
                layers.append(nn.Conv2d(net_config['obs_channels'], net_config['obs_channels'], 3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Conv2d(net_config['obs_channels'], net_config['obs_channels'], 3, padding=1, stride=2))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.BatchNorm2d(net_config['obs_channels']))



            self.obs_encoder = nn.ModuleList(layers)
            final_dim = net_config['obs_size']
            for _ in range(net_config['num_conv_layers']):
                final_dim = int(np.ceil(final_dim / 2))
            final_dim = final_dim**2 * net_config['obs_channels']

            self.obs_encoder.append(nn.Flatten())
            self.obs_encoder.append(
                nn.Linear(final_dim, 
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
        
        layers = []
        for i in range(net_config['num_transformer_blocks']):
            if i == 0:
                input_dim = 2*net_config['node_embedding_dim']
            else:
                input_dim = net_config['embedding_dim']
            layers.append(TransformerBlock(input_dim=input_dim,
                                embedding_dim=net_config['embedding_dim'],
                                num_heads=net_config['num_heads'],
                                out_dim=net_config['embedding_dim'],
                                mix_dim=net_config['embedding_dim'],
                                output_layer='linear',
                                dropout=net_config['dropout'],
                                residual=True))
        self.transformer_blocks = nn.ModuleList(layers)

        # Heads
        self.stop_head = nn.ModuleList(
            [TransformerBlock(input_dim=net_config['embedding_dim'],
                                embedding_dim=net_config['embedding_dim'],
                                num_heads=net_config['num_heads'],
                                out_dim=net_config['embedding_dim'],
                                mix_dim=None,
                                output_layer='linear',
                                dropout=net_config['dropout'],
                                residual=True)
                                for _ in range(net_config['num_tb_stop_heads'])]
        )
        self.stop_head.append(nn.Linear(net_config['embedding_dim'], 1))

        self.fwd_transition_head = nn.ModuleList(
            [TransformerBlock(input_dim=net_config['embedding_dim'],
                                embedding_dim=net_config['embedding_dim'],
                                num_heads=net_config['num_heads'],
                                out_dim=net_config['embedding_dim'],
                                mix_dim=None,
                                output_layer='linear',
                                dropout=net_config['dropout'],
                                residual=True)
                                for _ in range(net_config['num_tb_transition_heads'])]
        )
        self.fwd_transition_head.append(nn.Linear(net_config['embedding_dim'], 1))
        self.bck_transition_head = nn.ModuleList(
            [TransformerBlock(input_dim=net_config['embedding_dim'],
                                embedding_dim=net_config['embedding_dim'],
                                num_heads=net_config['num_heads'],
                                out_dim=net_config['embedding_dim'],
                                mix_dim=None,
                                output_layer='linear',
                                dropout=net_config['dropout'],
                                residual=True)
                                for _ in range(net_config['num_tb_transition_heads'])]
        )
        self.bck_transition_head.append(nn.Linear(net_config['embedding_dim'], 1))
        self.flow_head = nn.ModuleList(
            [TransformerBlock(input_dim=net_config['embedding_dim'],
                                embedding_dim=net_config['embedding_dim'],
                                num_heads=net_config['num_heads'],
                                out_dim=net_config['embedding_dim'],
                                mix_dim=None,
                                output_layer='linear',
                                dropout=net_config['dropout'],
                                residual=False)
                                for _ in range(net_config['num_flow_layers'])])
        self.flow_head.append(nn.Linear(net_config['embedding_dim'], 1))

        self.node_embedding_head = nn.ModuleList(
            [TransformerBlock(input_dim=net_config['embedding_dim'],
                                embedding_dim=net_config['embedding_dim'],
                                num_heads=net_config['num_heads'],
                                out_dim=net_config['node_embedding_dim'],
                                mix_dim=None,
                                output_layer='linear',
                                dropout=net_config['dropout'],
                                residual=True)
                                for _ in range(net_config['num_tb_transition_heads'])]
        )
    def forward(self, adj, node_embeddings, obs, mask):
        """
        adj: (B, max_nodes, max_nodes)
        node_embeddings: (B, max_nodes, node_embedding_dim), 0 for non-nodes
        obs: (B, obs_channels, obs_size, obs_size)
        mask: (B, max_nodes, max_nodes)
        retunrs: stop_logits (B, 1), transition_logits (B, max_nodes, max_nodes) of toggling an edge
        """
        
        # create sequence of edges as pairs of node embeddings (source, target), whith 0 for non-edges
        edge_embeddings = torch.cat([node_embeddings.unsqueeze(2).repeat(1, 1, self.net_config['max_nodes'], 1),
                                    node_embeddings.unsqueeze(1).repeat(1, self.net_config['max_nodes'], 1, 1)], 
                                    dim=-1)
        edge_embeddings = edge_embeddings.view(adj.shape[0], -1, 2*node_embeddings.shape[2] )  # (B, max_nodes*max_nodes, 2*node_embedding_dim)
        # set non-edges to 0
        edge_embeddings_ = edge_embeddings.mul(adj.view(adj.shape[0], -1, 1).clone())
        # embed observation
        obs_embedding = obs
        for layer in self.obs_encoder:
            obs_embedding = layer(obs_embedding)

        
        # embed the edge sequence with transformer blocks
        for tb in self.transformer_blocks:
            edge_embeddings_ = tb(edge_embeddings_, obs_embedding, mask=None)
        # forward transition probabilities, one value per element in the edge sequence
        for ix, layer in enumerate(self.fwd_transition_head):
            fwd_transition_ = layer(edge_embeddings_)
        fwd_transition_= fwd_transition_.squeeze(-1)  # (B, max_nodes*max_nodes)
        # set bad transition probabilities to 0 
        fwd_transition_ = fwd_transition_.masked_fill(mask.view(adj.shape[0], -1) == 0, -1e9)
        # softmax over the edge sequence
        fwd_transition_probs = F.softmax(fwd_transition_, dim=-1).view(adj.shape[0], adj.shape[1], adj.shape[2])  # (B, max_nodes, max_nodes)

        # same for backward transition probabilities
        for ix, layer in enumerate(self.bck_transition_head):
            bck_transition_ = layer(edge_embeddings_)
        bck_transition_= bck_transition_.squeeze(-1)  # (B, max_nodes*max_nodes)
        bck_transition_probs = F.softmax(bck_transition_, dim=-1).view(adj.shape[0], adj.shape[1], adj.shape[2])  # (B, max_nodes, max_nodes)

        # stop probability, aggregate at the last layer of the stop head
        for ix, layer in enumerate(self.stop_head):
            stop_ = layer(edge_embeddings_)
        stop_ = stop_.sum(dim=1)  # (B, 1)
        stop_prob = F.sigmoid(stop_)  # (B, 1)
        # same for flow
        for ix, layer in enumerate(self.flow_head):
            flow_ = layer(edge_embeddings_)
        flow = flow_.mean(dim=1)  # (B, 1)

        # get new node embeddings
        for ix, layer in enumerate(self.node_embedding_head):
            node_embeddings = layer(edge_embeddings_)
        # aggregate over the edge sequence from their appearance in the source positions and the target positions in edge embeddings
        node_embeddings = node_embeddings.view(adj.shape[0], adj.shape[1], adj.shape[2], -1).sum(dim=2)  # (B, max_nodes, node_embedding_dim)
        # mask nodes that are not in the graph (as they are not in the adjacency matrix)
        not_adj_mask = (adj.sum(dim=-1) == 0).unsqueeze(-1).repeat(1, 1, node_embeddings.shape[-1])
        node_embeddings = node_embeddings.masked_fill(not_adj_mask, 0)
        return bck_transition_probs, fwd_transition_probs, stop_prob, flow, node_embeddings

class ContrastiveScorer(nn.Module):
    # Learns to score pairs of graphs, whether they represent the same episode or not
    def __init__(self, net_config):
        super(ContrastiveScorer, self).__init__()
        self.net_config = net_config
        # Graph encoder
        layers = []
        for i in range(net_config['num_transformer_blocks']):
            if i == 0:
                input_dim = 2*net_config['node_embedding_dim']
            else:
                input_dim = net_config['embedding_dim']
            layers.append(TransformerBlock(input_dim=input_dim,
                                embedding_dim=net_config['embedding_dim'],
                                num_heads=net_config['num_heads'],
                                out_dim=net_config['embedding_dim'],
                                mix_dim=net_config['embedding_dim'],
                                output_layer='linear',
                                dropout=net_config['dropout'],
                                residual=True))
        layers.append(nn.Linear(net_config['embedding_dim'], net_config['graph_embedding_dim']))
        self.transformer_blocks = nn.ModuleList(layers)
        
        self.contrastive_head = nn.ModuleList(
            [nn.Linear(2*net_config['num_graph_per_obs']*net_config['graph_embedding_dim'], 2*net_config['graph_embedding_dim']),
            nn.ReLU()]
            + [nn.Linear(2*net_config['graph_embedding_dim'], 2*net_config['graph_embedding_dim']), nn.ReLU()]
               * net_config['num_contrastive_layers'] + [nn.Linear(2*net_config['graph_embedding_dim'], 1), nn.Sigmoid()]
        )
    def forward(self, adj1, adj2, node_embeddings1, node_embeddings2):
        
        edge_embeddings1 = torch.cat([node_embeddings1.unsqueeze(2).repeat(1, 1, self.net_config['max_nodes'], 1),
                                    node_embeddings1.unsqueeze(1).repeat(1, self.net_config['max_nodes'], 1, 1)], 
                                    dim=-1) # [batch_size*num_graph_per_obs, max_nodes, max_nodes, 2*node_embedding_dim]
        edge_embeddings2 = torch.cat([node_embeddings2.unsqueeze(2).repeat(1, 1, self.net_config['max_nodes'], 1),
                                    node_embeddings2.unsqueeze(1).repeat(1, self.net_config['max_nodes'], 1, 1)],
                                    dim=-1)
        edge_embeddings1 = edge_embeddings1.view(adj1.shape[0],  self.net_config['max_nodes']*self.net_config['max_nodes'], edge_embeddings1.shape[3] )  # (B*K, max_nodes*max_nodes, 2*node_embedding_dim
        # set non-edges to 0

        edge_embeddings1 = edge_embeddings1 * adj1.view(-1,  self.net_config['max_nodes']*self.net_config['max_nodes'], 1)
        edge_embeddings2 = edge_embeddings2.view(adj1.shape[0],  self.net_config['max_nodes']*self.net_config['max_nodes'], edge_embeddings2.shape[3] )  # (B*K, max_nodes*max_nodes, 2*node_embedding_dim)
        edge_embeddings2 = edge_embeddings2 * adj2.view(-1,  self.net_config['max_nodes']*self.net_config['max_nodes'], 1)

        # embed the edge sequence with transformer blocks
        for tb in self.transformer_blocks:
            edge_embeddings1 = tb(edge_embeddings1)
            edge_embeddings2 = tb(edge_embeddings2)
        # aggregate over the edge sequence
        graph1_x = edge_embeddings1.sum(dim=1) # [batch_size*num_graph_per_obs, graph_embedding_dim]
        graph2_x = edge_embeddings2.sum(dim=1)
        # reshape to [batch_size, num_graph_per_obs*graph_embedding_dim]
        graph1_x = graph1_x.view(-1, self.net_config['num_graph_per_obs']*self.net_config['graph_embedding_dim'])
        graph2_x = graph2_x.view(-1, self.net_config['num_graph_per_obs']*self.net_config['graph_embedding_dim'])
        # concatenate graphs embeddings
        x = torch.cat([graph1_x, graph2_x], dim=-1) # [batch_size, 2*num_graph_per_obs*embedding_dim]
        # apply contrastive head
        for layer in self.contrastive_head:
            x = layer(x)

        return x
    
    def _contrastive_loss(self, adj1, adj2, nd_ebd1, nd_ebd2, labels, device):
        # computes the contrastive loss for batch of graphs sequences 
        # adj1, adj2: lists of adjacency matrices of shape [batch_size, K, max_nodes, max_nodes], each K matrix is a graph representing the same observation
        # nd_ebd1, nd_ebd2: lists of node embeddings of shape [batch_size, K, max_nodes, node_embedding_dim]
        # labels: list of labels of shape [batch_size, 1] that indicate whether a graph sequence in adj1 and adj2 represent the same observation
        # device: torch.device object

        B, K, N = (self.net_config['batch_size'], self.net_config['num_graph_per_obs'], self.net_config['max_nodes'])
        E = self.net_config['node_embedding_dim']

        # reshape to [batch_size*K, max_nodes, max_nodes]
        adj1 = adj1.view(B*K, N, N).to(device)
        adj2 = adj2.view(B*K, N, N).to(device)
        # reshape to [batch_size*K, max_nodes, node_embedding_dim]
        nd_ebd1 = nd_ebd1.view(B*K, N, E).to(device)
        nd_ebd2 = nd_ebd2.view(B*K, N, E).to(device)

        scores = self.forward(adj1, adj2, nd_ebd1.clone(), nd_ebd2.clone()) # [batch_size, 1]
        labels = labels.view(-1)

        loss = F.binary_cross_entropy(scores.view(-1), labels, reduction='none')
        return loss
    
class ContrastGFN(nn.Module):
    # combined Generative Flow Network and Contrastive Network
    def __init__(self, net_config_cont, net_config_gfn, device):
        super(ContrastGFN, self).__init__()
        self.gfn = ConditionalFlowModel(net_config_gfn)
        self.contrast = ContrastiveScorer(net_config_cont)
        self.device = device
        self.net_config_gfn = net_config_gfn
        self.net_config_cont = net_config_cont
    

    def train_step(self, obs1, obs2, adj1, adj2, node_embeddings1, node_embeddings2, labels):
        # performs a transition step, computes the gfn loss and the contrastive loss for that transition
        # obs1, obs2: lists of observations of shape [batch_size*K, C, H, W]
        # adj1, adj2: lists of adjacency matrices of shape [batch_size, K, max_nodes, max_nodes], each K matrix is a graph representing the same observation
        # nd_ebd1, nd_ebd2: lists of node embeddings of shape [batch_size, K, max_nodes, node_embedding_dim]
        B, K, N = (self.net_config_gfn['batch_size'], self.net_config_gfn['num_graph_per_obs'], self.net_config_gfn['max_nodes'])
        C, H, W = (self.net_config_gfn['obs_channels'], self.net_config_gfn['height'], self.net_config_gfn['width'])
        E = self.net_config_gfn['embedding_dim']
        adj1 = adj1.view(B, K, N, N)
        adj2 = adj2.view(B, K, N, N)
        contrastive_loss = self.contrast._contrastive_loss(adj1, adj2, node_embeddings1.detach(), node_embeddings2.detach(), labels, self.device)

        r_contrastive_loss = contrastive_loss.mean()
        
        # create big batch of adj1 and adj2 of size [2*batch_size*num_graph_per_obs, max_nodes, max_nodes]
        adj = torch.cat([adj1, adj2], dim=0).view(2*B*K, N, N).to(self.device)
        node_ebd = torch.cat([node_embeddings1, node_embeddings2], dim=0).view(2*B*K, N, E).to(self.device)
        obs = torch.cat([obs1, obs2], dim=0).view(2*B*K, C, H, W).to(self.device)

        bck_tr_probs, fwd_tr_probs, stop_prob, flow, node_ebds_ = self.gfn(adj, node_ebd, obs, get_mask(adj))

        # sample a stop transition
        stop = torch.bernoulli(stop_prob).to(self.device)
        # for every that hasn't stopped, sample an edge to toggle from the forward transition probabilities
        fwd_tr_probs = fwd_tr_probs * (1-stop).view(-1, 1, 1)
        fwd_tr_probs = fwd_tr_probs / fwd_tr_probs.sum(dim=1, keepdim=True)

        # sample an edge to toggle, should be of shape [batch_size*K, 1] where elements are int in [0, max_nodes**2-1]
        fwd_tr = torch.multinomial(fwd_tr_probs.view(2*B*K, N*N), 1)


        # check right shape
        assert fwd_tr.squeeze().shape == (2*B*K,)
        
        # update adjacency matrices for not stopped transitions otherwise keep them the same
        adj_ = adj.clone()
        for i in range(adj.shape[0]):
            if stop[i] == 0:
                # toggle the edge
                adj_[i, fwd_tr[i]//N, fwd_tr[i]%N] = 1 - adj_[i, fwd_tr[i]//N, fwd_tr[i]%N]
                adj_[i, fwd_tr[i]%N, fwd_tr[i]//N] = 1 - adj_[i, fwd_tr[i]%N, fwd_tr[i]//N]
        
        # compute for next step
        bck_tr_probs_, fwd_tr_probs_, stop_prob_, flow_, node_ebds__ = self.gfn(adj, node_ebds_, obs, get_mask(adj))

        # contrastive loss for the next step
        # split the batch into the two batches of the original size
        adj1_, adj2_ = adj_[:B*K], adj_[B*K:]
        # reshape to [batch_size, K, max_nodes, max_nodes]
        adj1_ = adj1_.view(B, K, N, N)
        adj2_ = adj2_.view(B, K, N, N)
        node_ebds_1, node_ebds_2 = node_ebds__[:B*K], node_ebds__[B*K:]
        contrastive_loss_ = self.contrast._contrastive_loss(adj1_, adj2_, node_ebds_1.detach(), node_ebds_2.detach(), labels, self.device)

        dE = contrastive_loss_ - contrastive_loss # B, 1
        dE = dE.repeat(1, K).view(-1, 1) # B*K, 1
        
    
        # forward transition probabilities of the selected transitions
        selected_fwd_tr_probs = fwd_tr_probs.view(2*B*K, N*N)[torch.arange(2*B*K), fwd_tr.squeeze()]
        # backward transition probabilities of the selected transitions
        selected_bck_tr_probs = bck_tr_probs_.view(2*B*K, N*N)[torch.arange(2*B*K), fwd_tr.squeeze()]
        # compute the energy difference between the two steps from the contrastive 

        eps = 1e-8

        if self.net_config_gfn['log_flow']:
            gfn_loss = (flow.squeeze() + torch.log(selected_fwd_tr_probs + eps) - flow_.squeeze() - torch.log(selected_bck_tr_probs + eps) + dE.detach())**2
        else:
            gfn_loss = (torch.log(flow.squeeze()+ eps) + torch.log(selected_fwd_tr_probs + eps) - torch.log(flow_.squeeze() + eps) - torch.log(selected_bck_tr_probs + eps) + dE.detach())**2
        
        return adj_, node_ebds__, stop, gfn_loss.mean() + contrastive_loss + contrastive_loss_, r_contrastive_loss, gfn_loss.mean(), dE.mean()
    
    def forward(self, obs1, obs2, adj1, adj2, node_embeddings1, node_embeddings2):
        # just computes the contrastive loss for the given observations and the forward transition probabilities
        # obs1, obs2: lists of observations of shape [batch_size, C, H, W]
        # adj1, adj2: lists of adjacency matrices of shape [batch_size, K, max_nodes, max_nodes], each K matrix is a graph representing the same observation
        # nd_ebd1, nd_ebd2: lists of node embeddings of shape [batch_size, K, max_nodes, node_embedding_dim]

        adj = torch.cat([adj1, adj2], dim=0).view(-1, adj1.shape[2], adj1.shape[3]).to(self.device)
        node_ebd = torch.cat([node_embeddings1, node_embeddings2], dim=0).view(-1, node_embeddings1.shape[2], node_embeddings1.shape[3]).to(self.device)

        contrast = self.contrast(adj1, adj2, node_embeddings1, node_embeddings2)

        bck_tr_probs, fwd_tr_probs, stop_prob, flow, node_ebds_ = self.gfn(adj, node_ebd, obs1, get_mask(adj))

        return contrast, fwd_tr_probs, stop_prob, flow, node_ebds_