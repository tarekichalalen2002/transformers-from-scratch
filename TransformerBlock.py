from torch import nn
from torch.nn import functional as F
from MultiHeadAttention import MultiHeadAttention
from PositionalEncoding import PositionalEncoding
from ForwardFeed import ForwardFeed


class TransformerBlock(nn.Model):
    def __init__(self, embed_size, heads, hidden_dim, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.forward_feed = ForwardFeed(embed_size, hidden_dim, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.forward_feed(x)
        return self.dropout(self.norm2(forward + x))