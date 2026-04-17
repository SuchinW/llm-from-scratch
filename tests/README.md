# tests

The feedback loop for a code-first repo. Without "loss went down" as a signal, these tests are how you know your implementations are correct.

## Three kinds of tests per component

1. **Shape tests** - pass a dummy input, assert the output shape is what the math says it should be. Catches 50% of bugs instantly.
2. **Equivalence tests** - where possible, check your implementation against a reference. `MultiHeadAttention` should match `torch.nn.MultiheadAttention` (or `F.scaled_dot_product_attention`) on identical inputs up to floating-point tolerance. Your BPE should match `tiktoken` on the same text. RMSNorm should match `torch.nn.RMSNorm` (PyTorch 2.4+).
3. **Gradient tests** - run `loss.backward()` and assert gradients exist and are non-zero on all trainable params. Catches detached tensors, frozen params, dtype mismatches.

## Running tests

```bash
pytest                          # run everything
pytest tests/test_attention.py  # one file
pytest -k "mla"                 # tests matching a substring
pytest -x                       # stop on first failure
pytest -v                       # verbose
```
