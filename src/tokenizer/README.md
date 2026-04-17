# tokenizer

Turning text into integer token IDs.

## Read in this order

1. **`simple_tokenizer.py`** - word-level or character-level tokenization. Split on whitespace/punctuation, build a vocab, map to IDs. Good enough to understand the abstraction, hopeless on real text.
2. **`bpe.py`** - Byte-Pair Encoding. Starts from bytes, iteratively merges the most frequent pair. This is what GPT-2, GPT-3, and most modern LLMs use. GPT-2 ships with `tiktoken` - we implement it, then verify against `tiktoken` in tests.

## What changes

From `simple` to `bpe`: the vocabulary is no longer fixed to full words. Common substrings become tokens, rare words get broken into pieces. Handles out-of-vocabulary gracefully because the fallback is always bytes.
