# normalization

Keeping activations well-scaled across layers.

## Read in this order

1. **`layer_norm.py`** - standard LayerNorm. Subtracts the mean, divides by standard deviation, applies learnable scale and shift. Used by GPT-2.
2. **`rms_norm.py`** - RMSNorm. Drops the mean-subtraction and the shift parameter. Just divides by root-mean-square and applies a learnable scale. Cheaper, works as well or better in practice. Used by LLaMA, DeepSeek, and most modern models.

## What changes

RMSNorm removes two things from LayerNorm: the centering (mean subtraction) and the bias term. The hypothesis is that re-centering isn't necessary - rescaling alone is enough. The empirical result: same quality, faster.
