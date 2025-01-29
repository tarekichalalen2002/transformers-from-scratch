from torch import nn
from torch.nn import functional as F
from PositionalEncoding import PositionalEncoding
from TransformerBlock import TransformerBlock

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, hidden_dim, max_len, dropout):
        super(TransformerEncoder, self).__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, max_len, dropout)
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.word_embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, x, x, mask)
        return x