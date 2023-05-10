import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from torch_geometric.nn import GATConv, GPSConv, GINEConv, global_mean_pool
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data, Batch
from util import get_mask
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
            x = self.mix_layer(torch.cat([x, mix.unsqueeze(1).repeat(1, x.shape[1], 1)], dim=-1))
        
        x = self.activation(self.lin1(x))
        x = self.norm1(x)
        x = self.activation(self.lin2(x))
        x = self.attention(x, mask)
        x = self.norm2(x)
        x = self.activation(self.lin3(x))
        if self.residual:
            x = x + res
        return x 

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
                layers.append(nn.ReLU())
                layers.append(nn.Conv2d(net_config['obs_channels'], net_config['obs_channels'], 3, padding=1, stride=2))
                layers.append(nn.ReLU())
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
                                for _ in range(net_config['num_flow_heads'])])
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
        edge_embeddings = edge_embeddings * adj.view(adj.shape[0], -1, 1)
        # embed observation
        obs_embedding = obs
        for layer in self.obs_encoder:
            obs_embedding = layer(obs_embedding)

        
        # embed the edge sequence with transformer blocks
        for tb in self.transformer_blocks:
            edge_embeddings = tb(edge_embeddings, obs_embedding, mask=None)
        # forward transition probabilities, one value per element in the edge sequence
        for ix, layer in enumerate(self.fwd_transition_head):
            fwd_transition_ = layer(edge_embeddings)
        fwd_transition_= fwd_transition_.squeeze(-1)  # (B, max_nodes*max_nodes)
        # set bad transition probabilities to 0 
        fwd_transition_ = fwd_transition_.masked_fill(mask.view(adj.shape[0], -1) == 0, -1e9)
        # softmax over the edge sequence
        fwd_transition_probs = F.softmax(fwd_transition_, dim=-1).view(adj.shape[0], adj.shape[1], adj.shape[2])  # (B, max_nodes, max_nodes)

        # same for backward transition probabilities
        for ix, layer in enumerate(self.bck_transition_head):
            bck_transition_ = layer(edge_embeddings)
        bck_transition_= bck_transition_.squeeze(-1)  # (B, max_nodes*max_nodes)
        bck_transition_probs = F.softmax(bck_transition_, dim=-1).view(adj.shape[0], adj.shape[1], adj.shape[2])  # (B, max_nodes, max_nodes)

        # stop probability, aggregate at the last layer of the stop head
        for ix, layer in enumerate(self.stop_head):
            stop_ = layer(edge_embeddings)
        stop_ = stop_.sum(dim=1)  # (B, 1)
        stop_prob = F.sigmoid(stop_)  # (B, 1)
        # same for flow
        for ix, layer in enumerate(self.flow_head):
            flow_ = layer(edge_embeddings)
        flow = flow_.mean(dim=1)  # (B, 1)

        # get new node embeddings
        for ix, layer in enumerate(self.node_embedding_head):
            node_embeddings = layer(edge_embeddings)
        # aggregate over the edge sequence from their appearance in the source positions and the target positions in edge embeddings
        node_embeddings = node_embeddings.view(adj.shape[0], adj.shape[1], adj.shape[2], -1).sum(dim=2)  # (B, max_nodes, node_embedding_dim)
        # mask nodes that are not in the graph (as they are not in the adjacency matrix)
        not_adj_mask = (adj.sum(dim=-1) == 0).unsqueeze(-1).repeat(1, 1, node_embeddings.shape[-1])
        node_embeddings = node_embeddings.masked_fill(not_adj_mask, 0)
        return bck_transition_probs, fwd_transition_probs, stop_prob, flow, node_embeddings


class GAT(torch.nn.Module):
    # Graph attention block
    def __init__(self, in_dim, out_dim, hid, in_head, out_head, dropout):
        super(GAT, self).__init__()
        self.in_head = in_head
        self.out_head = out_head
        self.hid = hid
        self.dropout = dropout
        self.conv1 = GATConv(in_dim, self.hid, heads=self.in_head, dropout=dropout)
        self.conv2 = GATConv(self.hid*self.in_head, out_dim, concat=False,
                             heads=self.out_head, dropout=dropout)
    def forward(self,x, edge_index):
        # x: node feature matrix of shape [batch_size, num_nodes, in_channels]
        # edge_index: graph connectivity in COO format with shape [batch_size, 2, num_edges]
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class ContrastiveScorer(nn.Module):
    # Learns to score pairs of graphs, whether they represent the same episode or not
    def __init__(self, net_config):
        super(ContrastiveScorer, self).__init__()
        self.net_config = net_config
        # Graph encoder with Graph Attention Transformer
        self.graph_encoder_gat = nn.ModuleList(
            [GAT(net_config['num_node_features'], net_config['hidden_dim'], net_config['hidden_dim'], net_config['num_heads'], net_config['num_heads'], net_config['dropout']),
            nn.LeakyReLU(inplace=True)]
            + [GAT(net_config['hidden_dim'], net_config['hidden_dim'], net_config['hidden_dim'], net_config['num_heads'], net_config['num_heads'], net_config['dropout']),
            nn.LeakyReLU(inplace=True)]*net_config['num_gat_layers'] +
            [nn.Flatten(), nn.Linear(net_config['hidden_dim'], net_config['graph_embedding_dim'])]
        )
        self.contrastive_head = nn.ModuleList(
            [nn.Linear(2*net_config['num_graph_per_obs']*net_config['graph_embedding_dim'], 2*net_config['graph_embedding_dim']),
            nn.ReLU()] * net_config['num_contrastive_layers'] + [nn.Linear(2*net_config['graph_embedding_dim'], 1), nn.Sigmoid()]
        )
    def forward(self, graph_batch_1, graph_batch_2):
        # graphs_batch is a list of are torch_geometric.data.Batch objects of length  true_batch_size*num_graph_per_obs
        
        graph1_x = graph_batch_1.x
        graph2_x = graph_batch_2.x
        graph1_num_nodes = graph1_x.shape[0]
        graph2_num_nodes = graph2_x.shape[0]
        batch_size = graph_batch_1.batch.max().item() + 1
        # encode graphs
        for layer in self.graph_encoder_gat:
            
            if isinstance(layer, GAT):
                graph1_x = layer(graph1_x.view(graph1_num_nodes, -1), graph_batch_1.edge_index)
                graph2_x = layer(graph2_x.view(graph2_num_nodes, -1), graph_batch_2.edge_index)
            elif isinstance(layer, nn.Flatten):
                # aggregate node embeddings with mean pooling before flattening
                graph1_x = global_mean_pool(graph1_x.view(graph1_num_nodes, -1), graph_batch_1.batch)
                graph2_x = global_mean_pool(graph2_x.view(graph2_num_nodes, -1), graph_batch_2.batch)

                graph1_x = layer(graph1_x)
                graph2_x = layer(graph2_x)
            else:
                # have to change view to include batch dimension
                graph1_x = layer(graph1_x)
                graph2_x = layer(graph2_x)
        # reshape to [batch_size, num_graph_per_obs*graph_embedding_dim]
        graph1_x = graph1_x.view(batch_size, -1)
        # concatenate graphs embeddings
        x = torch.cat([graph1_x, graph2_x], dim=-1) # [batch_size, 2*num_graph_per_obs*graph_embedding_dim]
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
        batch_size = adj1.shape[0]
        K = adj1.shape[1]
        # reshape to [batch_size*K, max_nodes, max_nodes]
        adj1 = adj1.view(batch_size*K, adj1.shape[2], adj1.shape[3]).to(device)
        adj2 = adj2.view(batch_size*K, adj2.shape[2], adj2.shape[3]).to(device)
        # reshape to [batch_size*K, max_nodes, node_embedding_dim]
        nd_ebd1 = nd_ebd1.view(batch_size*K, nd_ebd1.shape[2], nd_ebd1.shape[3]).to(device)
        nd_ebd2 = nd_ebd2.view(batch_size*K, nd_ebd2.shape[2], nd_ebd2.shape[3]).to(device)
        # create torch_geometric.data.Batch objects
        graph_batch_1 = []
        graph_batch_2 = []
        for i in range(batch_size):
            for k in range(K):
                graph_batch_1.append(Data(x=nd_ebd1[i, k], edge_index=adj1[i, k].nonzero().t(), device=device))
                graph_batch_2.append(Data(x=nd_ebd2[i, k], edge_index=adj2[i, k].nonzero().t(), device=device))
        graph_batch_1 = Batch.from_data_list(graph_batch_1, device=device)
        graph_batch_2 = Batch.from_data_list(graph_batch_2, device=device)

        # compute contrastive loss
        scores = self.forward(graph_batch_1, graph_batch_2)
        labels = labels.view(-1)

        loss = F.binary_cross_entropy(scores.view(-1), labels, reduction='none')
        return loss