# attention

Attention mechanisms, from the simplest to the most modern.

## Read in this order

1. **`self_attention.py`** - single-head scaled dot-product attention. The core idea: query, key, value projections and `softmax(QK^T / sqrt(d)) V`. No mask, no heads, no tricks.
2. **`causal_self_attention.py`** - adds the triangular causal mask so tokens can't attend to future positions. This is what makes it a *decoder* attention.
3. **`multi_head_attention.py`** - runs multiple attention heads in parallel, each on a lower-dimensional slice, then concatenates. The standard GPT-2 attention.
4. **`multi_query_attention.py` (MQA)** - all heads share a single K and V. Dramatically smaller KV cache at inference time, small quality hit.
5. **`grouped_query_attention.py` (GQA)** - the middle ground: groups of heads share K and V. Used by LLaMA 2/3.
6. **`multi_head_latent_attention.py` (MLA)** - DeepSeek's innovation. Compresses K and V into a low-rank latent space, reducing KV cache size further while preserving quality. Uses decoupled RoPE for position encoding.

## What changes between each step

Each file adds exactly one concept to the previous one. If you diff file N against file N-1, you should see exactly the change the name suggests - nothing more.

| From | To | What's new |
|---|---|---|
| self_attention | causal_self_attention | triangular mask |
| causal_self_attention | multi_head_attention | head dimension, reshape, concat |
| multi_head_attention | multi_query_attention | shared K, V across heads |
| multi_query_attention | grouped_query_attention | groups instead of single shared K, V |
| grouped_query_attention | multi_head_latent_attention | low-rank K/V compression, decoupled RoPE |
