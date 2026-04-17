# notes

Your own notes on each concept, in your own words. One markdown file per hard idea.

## Why this folder matters

Explaining something in writing is how you find the gaps in your understanding. A concept isn't really yours until you can write about it without looking at the book. These notes are also what turn a learning repo into a portfolio - a reviewer reading your notes learns something, which is a rare quality bar.

## Suggested topics (add more as you go)

- attention from scratch - the three projections and what the softmax really does
- why causal masking works and how it's implemented
- how multi-head attention differs from running N single-head attentions
- the KV cache and why it exists
- RoPE - why rotations, not additions, and why it generalizes
- RMSNorm vs LayerNorm - what gets dropped and why
- SwiGLU - what the gate is doing
- MoE routing and the load balancing loss
- MLA - the intuition for compressing K/V and why RoPE has to be decoupled
- GRPO vs PPO - why no value model
- multi-token prediction - how auxiliary heads help the main objective
