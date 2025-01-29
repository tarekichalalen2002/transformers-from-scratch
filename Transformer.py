from torch import nn
from torch.nn import functional as F
from TransformerEncoder import TransformerEncoder
from TransformerDecoder import TransformerDecoder

class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, hidden_dim, max_len, dropout):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(vocab_size, embed_size, num_layers, heads, hidden_dim, max_len, dropout)
        self.decoder = TransformerDecoder(vocab_size, embed_size, num_layers, heads, hidden_dim, max_len, dropout)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x, trg, src_mask, trg_mask):
        encoder_out = self.encoder(x, src_mask)
        out = self.decoder(trg, encoder_out, src_mask, trg_mask)
        return self.fc_out(out)