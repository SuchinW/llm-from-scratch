# embeddings

Giving the model a sense of position.

## Read in this order

1. **`positional.py`** - learned absolute positional embeddings (GPT-2 style). A `(max_seq_len, d_model)` lookup table, added to the token embeddings at the input.
2. **`rope.py`** - Rotary Position Embeddings. Instead of adding a position vector at the input, RoPE rotates Q and K inside each attention layer by an angle that depends on position. Used by LLaMA, DeepSeek, and most modern models.

## What changes

Learned positional embeddings are fixed to a maximum sequence length and don't generalize beyond it. RoPE is applied per-layer to Q and K (not to the input), encodes *relative* position naturally, and extrapolates to longer sequences than seen during training.

DeepSeek uses a *decoupled* RoPE - a separate, smaller RoPE-encoded component in addition to the compressed latent K. That lives in `attention/multi_head_latent_attention.py`, not here.
