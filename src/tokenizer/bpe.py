class TiktokenBPE:
    """Thin adapter around tiktoken's pretrained encodings (default: gpt2)."""

    def __init__(self, encoding_name="gpt2"):
        import tiktoken
        self._enc = tiktoken.get_encoding(encoding_name)

    def encode(self, text):
        return self._enc.encode(text)

    def decode(self, ids):
        return self._enc.decode(ids)


if __name__ == "__main__":
    text = (
        "the quick brown fox jumps over the lazy dog. "
        "the quick brown fox is quick and brown."
    )

    tik = TiktokenBPE("gpt2")
    ids = tik.encode(text)
    print(f"[tiktoken] {len(text)} chars -> {len(ids)} tokens")
    assert tik.decode(ids) == text, "tiktoken round-trip failed"
