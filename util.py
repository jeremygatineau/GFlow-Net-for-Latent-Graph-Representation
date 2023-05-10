import torch
import numpy as np

def is_cycle(edges, edge):
    # check if adding edge creates a cycle
    # edges is a list of edges, edge is a new edge
    # returns True if cycle is created
    if len(edges) == 0:
        return False
    edges = edges + [edge]
    num_nodes = max([max(edge) for edge in edges]) + 1
    adj = torch.zeros((num_nodes, num_nodes))
    for edge in edges:
        adj[edge[0], edge[1]] = 1
    return (torch.trace(torch.matrix_power(adj, 5*num_nodes)) > 0).item()

def _get_mask(graph_adj):
    # mask for the available actions of a single adjacency matrix state
    # 0 means not available, 1 means available
    mask = torch.ones(graph_adj.shape)
    mask[graph_adj > 0] = 0 # already connected
    mask[torch.eye(graph_adj.shape[0]) > 0] = 0 # self loop
    # check if adding edge creates a cycle
    for i in range(graph_adj.shape[0]):
        for j in range(i+1, graph_adj.shape[0]):
            if is_cycle(graph_adj.nonzero().tolist(), [i, j]):
                mask[i, j] = 0
                mask[j, i] = 0
    return mask

def get_mask(adjs):
    # mask but for a batch of adjacency matrices
    # adjs: (batch_size, num_nodes, num_nodes)
    # mask: (batch_size, num_nodes, num_nodes)
    mask = torch.stack([_get_mask(adj) for adj in adjs])
    return mask
