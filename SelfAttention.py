import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size # the embedding size initialization
        self.heads = heads  # The number of heads in the multi-head attention
        self.head_dim = embed_size // heads  # the dimension of each head
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads" # Checking if the embedding size is divisible by the number of heads
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False) # The Linear layer for the values 
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False) # The Linear layer for the keys
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False) # The linear layer of the queries
        self.fc_out = nn.Linear(embed_size, embed_size)
    
    def forward(self, values, keys, query, mask):
        N = query.shape[0]  # Batch size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1] # The length of the values, keys and queries
        values = values.view(N, value_len, self.heads, self.head_dim) # Reshaping the values
        keys = keys.view(N, key_len, self.heads, self.head_dim) # Reshaping the keys
        query = query.view(N, query_len, self.heads, self.head_dim) # Reshaping the queries
        values = self.values(values) # the values after the linear layer
        keys = self.keys(keys) # the keys after the linear layer
        queries = self.queries(query) # the queries after the linear layer
        # computing the scaled dot product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) # The energy calculation, the dot product of the queries and keys
        energy = energy / math.sqrt(self.head_dim) # Scaling the energy
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-inf")) # Applying the mask for the padding 
        attention = torch.softmax(energy, dim=-1) # applying the softmax function
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]) # The output of the attention, the dot product of the attention and values   
        out = out.reshape(N, query_len, self.head_dim * self.heads) # Reshaping the output
        return self.fc_out(out) # The output after the linear layer