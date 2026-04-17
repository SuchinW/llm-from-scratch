# models

Full models, composed from the components in sibling folders.

## Read in this order

1. **`gpt.py`** - GPT-2 style decoder-only transformer. Uses `attention.multi_head_attention`, `embeddings.positional`, `normalization.layer_norm`, `feedforward.mlp`. This is the baseline you verify against everything else.
2. **`deepseek.py`** - DeepSeek-style model. Uses `attention.multi_head_latent_attention`, `embeddings.rope`, `normalization.rms_norm`, `feedforward.moe`. Same overall shape as GPT (stacked decoder blocks) - every internal component is swapped for its modern counterpart.

## The point of this folder

The components in the other folders are the interesting part. This folder is just wiring them together into a working model. A model file here should be *short* - mostly `__init__` with the right imports and a `forward` that composes them. If a model file gets long, the complexity probably belongs in a component, not here.
