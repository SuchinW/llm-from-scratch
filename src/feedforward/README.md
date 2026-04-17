# feedforward

The position-wise MLP that sits after attention in every transformer block.

## Read in this order

1. **`mlp.py`** - standard two-layer MLP with GELU activation. `Linear → GELU → Linear`. The GPT-2 FFN.
2. **`swiglu.py`** - SwiGLU. A gated variant with three linear layers: `(SiLU(W1 x) * W3 x) W2`. Used by LLaMA, DeepSeek. Higher quality for the same parameter budget.
3. **`moe.py`** - Mixture of Experts. Instead of one FFN, have N experts and a router that picks the top-k per token. Only the chosen experts run, so compute stays bounded while parameter count explodes. DeepSeek uses a fine-grained MoE with many small experts and shared always-on experts - this file implements that variant.

## What changes

| From | To | What's new |
|---|---|---|
| mlp | swiglu | gating (third linear layer multiplying element-wise) + SiLU activation |
| swiglu | moe | N copies of the FFN, a router that picks top-k, shared experts, load balancing loss |
