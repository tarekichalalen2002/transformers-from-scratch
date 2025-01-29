import torch.nn as nn
import torch.nn.functional as F
from SelfAttention import SelfAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.self_attention = SelfAttention(embed_size, heads) # The self attention layer
    def forward(self, values, keys, query, mask):
        return self.self_attention(values, keys, query, mask) # The output of the self attention layer