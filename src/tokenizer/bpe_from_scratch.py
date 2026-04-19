from collections import Counter


class BPETokenizer:
    """Byte-level BPE trained from scratch. Follows the minbpe algorithm:
    start from raw UTF-8 bytes (vocab 0..255), greedily merge the most
    frequent adjacent pair, assign it a new id, repeat until vocab_size."""

    def __init__(self):
        self.merges = {}  # (int, int) -> int
        self.vocab = {i: bytes([i]) for i in range(256)}

    def train(self, text, vocab_size):
        assert vocab_size >= 256, "vocab_size must be >= 256 (byte alphabet)"
        num_merges = vocab_size - 256
        ids = list(text.encode("utf-8"))

        for i in range(num_merges):
            pair_counts = Counter(zip(ids, ids[1:]))
            if not pair_counts:
                break
            pair = pair_counts.most_common(1)[0][0]
            new_id = 256 + i
            ids = self._merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            # pick the pair with the lowest merge index (earliest learned)
            pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = self._merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")

    @staticmethod
    def _merge(ids, pair, new_id):
        out = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out


if __name__ == "__main__":
    text = (
        "the quick brown fox jumps over the lazy dog. "
        "the quick brown fox is quick and brown."
    )

    bpe = BPETokenizer()
    bpe.train(text, vocab_size=300)
    ids = bpe.encode(text)
    print(f"[scratch] vocab: {len(bpe.vocab)}  merges: {len(bpe.merges)}")
    print(f"[scratch] {len(text)} chars -> {len(ids)} tokens")
    assert bpe.decode(ids) == text, "round-trip failed"
