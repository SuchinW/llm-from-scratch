# llm-from-scratch

**From GPT to DeepSeek - a reference implementation of every major component in a modern LLM, built one file at a time.**

This repo is a learning project. Each file implements one concept and builds on the last. The goal is to understand how modern LLMs work by coding every piece from scratch - not to train a production model. Training is an optional last step.

## The arc

The project progresses from the simplest possible self-attention up to the full DeepSeek architecture. Each new component adds exactly one idea to the previous one:

- **Attention**: single-head → causal → multi-head → MQA → GQA → MLA
- **Normalization**: LayerNorm → RMSNorm
- **Positional encoding**: learned → RoPE
- **Feedforward**: MLP → SwiGLU → Mixture-of-Experts
- **Models**: GPT-2 style → DeepSeek style (composed from the components above)
- **Training**: pretraining → supervised fine-tuning → GRPO
- **Inference**: vanilla generation → KV cache

## How to read this repo

Start at `src/attention/self_attention.py` and read in order. Each folder has its own README explaining the progression within that concept. Each file ends with a small `__main__` block so you can run it standalone and see what it produces.

```bash
python src/attention/self_attention.py
```

## Structure

```
src/
├── attention/        # self-attention → MLA
├── tokenizer/        # simple tokenizer → BPE
├── embeddings/       # learned positional → RoPE
├── normalization/    # LayerNorm → RMSNorm
├── feedforward/      # MLP → SwiGLU → MoE
├── models/           # GPT → DeepSeek (composed from the above)
├── training/         # pretrain → SFT → GRPO
├── inference/        # KV cache, generation
└── utils/            # shared helpers
tests/                # shape tests, equivalence tests, gradient tests
notebooks/            # visualizations, sanity checks
docs/
├── notes/            # my notes on each concept, in my own words
└── diagrams/         # architecture sketches
```

## Setup

```bash
git clone <repo>
cd llm-from-scratch
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Editable install means `from src.attention.multi_head_attention import MultiHeadAttention` works from anywhere in the repo.

## Progress

| Component | File | Status |
|---|---|---|
| Self-attention | `src/attention/self_attention.py` | ☐ |
| Causal self-attention | `src/attention/causal_self_attention.py` | ☐ |
| Multi-head attention | `src/attention/multi_head_attention.py` | ☐ |
| Multi-query attention (MQA) | `src/attention/multi_query_attention.py` | ☐ |
| Grouped-query attention (GQA) | `src/attention/grouped_query_attention.py` | ☐ |
| Multi-head latent attention (MLA) | `src/attention/multi_head_latent_attention.py` | ☐ |
| Simple tokenizer | `src/tokenizer/simple_tokenizer.py` | ☐ |
| BPE tokenizer | `src/tokenizer/bpe.py` | ☐ |
| Learned positional embeddings | `src/embeddings/positional.py` | ☐ |
| Rotary position embeddings (RoPE) | `src/embeddings/rope.py` | ☐ |
| LayerNorm | `src/normalization/layer_norm.py` | ☐ |
| RMSNorm | `src/normalization/rms_norm.py` | ☐ |
| MLP feedforward | `src/feedforward/mlp.py` | ☐ |
| SwiGLU | `src/feedforward/swiglu.py` | ☐ |
| Mixture-of-Experts | `src/feedforward/moe.py` | ☐ |
| GPT model | `src/models/gpt.py` | ☐ |
| DeepSeek model | `src/models/deepseek.py` | ☐ |
| KV cache | `src/inference/kv_cache.py` | ☐ |
| Generation loop | `src/inference/generate.py` | ☐ |
| Pretraining loop | `src/training/pretrain.py` | ☐ |
| Supervised fine-tuning | `src/training/sft.py` | ☐ |
| GRPO | `src/training/grpo.py` | ☐ |

Tick boxes as you go. When a component is done, link its file and its tests.

## Notes

Concept notes in my own words live in `docs/notes/`. These are the artifact of actually understanding something, not just typing it.
