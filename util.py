import torch
import numpy as np
import pickle
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
def _unit_test__get_mask():
    adj = torch.zeros((5, 5))
    adj[0, 1] = 1
    adj[1, 4] = 1

    mask = _get_mask(adj)
    print(mask)
    print(mask.nonzero().tolist())


def get_mask(adjs):
    # mask but for a batch of adjacency matrices
    # adjs: (batch_size, num_nodes, num_nodes)
    # mask: (batch_size, num_nodes, num_nodes)
    mask = torch.stack([_get_mask(adj) for adj in adjs])
    return mask

class ContrastiveDataset(torch.utils.data.IterableDataset):
    'Contrastive dataset of pairs of observations with labels 0 if they are from the same episode and 1 if they are from different episodes'
    def __init__(self, file_name):
        super(ContrastiveDataset).__init__()
        'Initialization'
        with open(file_name, "rb") as f:
            self.dataset = pickle.load(f)
            # remove alpha channel
            self.dataset = [(obs1[:,:,:3], obs2[:,:,:3], label) for obs1, obs2, label in self.dataset]
    def __iter__(self):
        # get info on current worker process
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            return iter(self.dataset)
        else:  # in a worker process
            # split workload
            per_worker = int(len(self.dataset) / float(worker_info.num_workers))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.dataset))
            return iter(self.dataset[iter_start:iter_end])
    """
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.dataset)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        obs1, obs2, label = self.dataset[index]
        # reshape images to (H, W, C)
        # convert images to RGB (saved as RGBA)
        obs1 = obs1[:,:,:3]
        obs2 = obs2[:,:,:3]

        return obs1, obs2, label"""
    
if __name__ == "__main__":
    _unit_test__get_mask()