import torch
from torch import nn
from torch.nn import functional as F
from TransformerDecoderBlock import TransformerDecoderBlock
from PositionalEncoding import PositionalEncoding

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, hidden_dim, max_len, dropout):
        super(TransformerDecoder, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, max_len, dropout)
        self.layers = nn.ModuleList(
            [TransformerDecoderBlock(embed_size, heads, hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        x = self.word_embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, trg_mask)
        return torch.softmax(self.fc_out(x), dim=-1) 