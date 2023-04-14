import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical



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
  # Modified GFLowNet model for DAG generation that generates a DAG conditioned on a given observation
  def __init__(self, input_dim, embedding_dim, num_layers, dropout=0.0):
    super(ConditionalFlowModel).__init__()
    