# inference

Generation and the machinery that makes it fast.

## Read in this order

1. **`generate.py`** - basic autoregressive generation. Greedy decode, top-k, top-p (nucleus), temperature. No caching - each new token re-runs attention over the entire sequence from scratch.
2. **`kv_cache.py`** - the KV cache. Cache the K and V projections for past tokens so each new step is O(1) in context length instead of O(N). Bolt onto a model by threading a cache object through the attention layers.

## Why this matters for DeepSeek

The KV cache is what motivates MLA. Standard MHA caches a full-dim K and V for every head at every position - the cache grows linearly with context and becomes the memory bottleneck at long contexts. MLA reduces this by storing only a compressed latent vector. Before implementing MLA, implement the plain KV cache here so you feel the problem MLA is solving.
