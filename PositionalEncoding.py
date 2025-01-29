import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len = 100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_size) # Initializing the positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # The position tensor initialization, the range of the possible positions
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size)) # the division term for frequency
        pe[:, 0::2] = torch.sin(position * div_term) # The positional encoding for % 2 == 0 indices (les nombres pairs)
        pe[:, 1::2] = torch.cos(position * div_term) # The positional encoding for % 2 == 1 indices (les nombres impairs)
        pe = pe.unsqueeze(0) # adding a new dimension to the positional encoding, the batch dimension
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :] # Adding the positional encoding to the input tensor