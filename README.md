# Transformers from Scratch

A PyTorch implementation of the encoder–decoder Transformer from [Attention Is All You Need](https://arxiv.org/abs/1706.03762), built as small modules you can read one file at a time. There is no training loop or dataset here: the repo is the model itself, from scaled dot-product attention up to the full sequence-to-sequence network.

Each class is a `torch.nn.Module`. Token ids go in, and the model returns logits over the target vocabulary.

## Architecture

```
source tokens                         target tokens
      |                                      |
 word embedding                        word embedding
      |                                      |
 sinusoidal positional encoding        sinusoidal positional encoding
      |                                      |
 N × TransformerBlock                  N × TransformerDecoderBlock
   • multi-head self-attention           • masked multi-head self-attention
   • position-wise feed-forward          • cross-attention over encoder states
      |                                  • position-wise feed-forward
      |                                      |
 encoder hidden states  ---------------->  |
                                             |
                                      linear projection
                                             |
                                      vocabulary logits
```

Attention is the scaled dot-product from the paper:

\[
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
\]

Queries, keys, and values are split across `heads` so that `embed_size` is divisible by the number of heads. A mask of zeros is filled with \(-\infty\) before the softmax, which is how padding and the causal (look-ahead) mask drop those positions.

Positional encodings are the fixed sinusoids

\[
PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d}), \quad PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})
\]

registered as a buffer and added to the token embeddings. The feed-forward block is two linear layers with a ReLU and dropout in between, applied independently at each position.

## Modules

| File | Class | Role |
| --- | --- | --- |
| `SelfAttention.py` | `SelfAttention` | Splits the embedding into heads, projects Q, K, and V, applies scaled dot-product attention and the mask, then projects the concatenated heads back to `embed_size`. |
| `MultiHeadAttention.py` | `MultiHeadAttention` | Thin wrapper around `SelfAttention`. The same module is used for self-attention and for cross-attention; what changes is which tensors are passed as query, key, and value. |
| `PositionalEncoding.py` | `PositionalEncoding` | Adds fixed sinusoidal positional encodings up to `max_len`. |
| `ForwardFeed.py` | `ForwardFeed` | Position-wise feed-forward network: `Linear → ReLU → Dropout → Linear`. |
| `TransformerBlock.py` | `TransformerBlock` | One encoder layer: multi-head attention, residual connection, layer norm, then the feed-forward block with its own residual and layer norm. |
| `TransformerDecoderBlock.py` | `TransformerDecoderBlock` | One decoder layer: masked self-attention, cross-attention over the encoder output, then the feed-forward block. Each sub-layer has a residual connection, dropout, and layer norm. |
| `TransformerEncoder.py` | `TransformerEncoder` | Token embedding, positional encoding, and a stack of `num_layers` encoder blocks. |
| `TransformerDecoder.py` | `TransformerDecoder` | Token embedding, positional encoding, and a stack of `num_layers` decoder blocks. |
| `Transformer.py` | `Transformer` | Encoder, decoder, and a final linear layer from `embed_size` to `vocab_size`. |

## Forward pass

`Transformer.forward(x, trg, src_mask, trg_mask)` does three things:

1. The encoder embeds `x`, adds positions, and runs every encoder block with `src_mask`. It returns contextual states of shape `(batch, src_len, embed_size)`.
2. The decoder embeds `trg`, adds positions, and runs every decoder block. Masked self-attention uses `trg_mask`. Cross-attention uses the encoder states as key and value, and `src_mask` to ignore source padding. The decoder then projects those states with its own linear layer and applies a softmax over the vocabulary.
3. `Transformer` applies a second linear layer, `fc_out`, mapping from `embed_size` to `vocab_size`.

Masks are compared against zero inside attention (`mask == 0` becomes \(-\infty\)). They need to broadcast to `(batch, heads, query_len, key_len)`.

- `src_mask` covers the source sequence. Shape `(batch, 1, 1, src_len)` broadcasts over heads and query positions.
- `trg_mask` is the causal mask, usually combined with a padding mask. A lower-triangular matrix of ones keeps each target position from attending to future tokens.

## Requirements

Dependencies are pinned in `requiremenets.txt`. PyTorch is the only library the model imports.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requiremenets.txt
```

## Usage

Run this from the `transformers-from-scratch` directory so the bare imports resolve.

```python
import torch
from Transformer import Transformer

vocab_size = 10000
embed_size = 512
num_layers = 6
heads = 8
hidden_dim = 2048
max_len = 100
dropout = 0.1

model = Transformer(
    vocab_size, embed_size, num_layers, heads, hidden_dim, max_len, dropout
)

batch, src_len, trg_len = 2, 20, 15
src = torch.randint(0, vocab_size, (batch, src_len))
trg = torch.randint(0, vocab_size, (batch, trg_len))

src_mask = torch.ones(batch, 1, 1, src_len)
trg_mask = torch.tril(torch.ones(trg_len, trg_len)).view(1, 1, trg_len, trg_len)
trg_mask = trg_mask.expand(batch, 1, trg_len, trg_len)

logits = model(src, trg, src_mask, trg_mask)
```

`embed_size` must be divisible by `heads`, and both sequences must be shorter than `max_len`.

## Layout

```
transformers-from-scratch/
├── SelfAttention.py
├── MultiHeadAttention.py
├── PositionalEncoding.py
├── ForwardFeed.py
├── TransformerBlock.py
├── TransformerDecoderBlock.py
├── TransformerEncoder.py
├── TransformerDecoder.py
├── Transformer.py
└── requiremenets.txt
```
