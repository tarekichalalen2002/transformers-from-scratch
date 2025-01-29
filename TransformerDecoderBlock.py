from torch import nn
from torch.nn import functional as F
from MultiHeadAttention import MultiHeadAttention
from ForwardFeed import ForwardFeed

class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, hidden_size, dropout=0.1):
        super(TransformerDecoderBlock, self).__init__()
        self.masked_attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        self.cross_attention = MultiHeadAttention(embed_size, heads)
        self.forward_feed = ForwardFeed(embed_size, hidden_size, dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        attention = self.masked_attention(x, x, x, trg_mask)
        x = self.dropout(self.norm1(attention+x))
        attention = self.cross_attention(x, enc_out, enc_out, src_mask)
        x = self.dropout(self.norm2(attention+x))
        forward = self.forward_feed(x)
        return self.dropout(self.norm3(forward+x))
