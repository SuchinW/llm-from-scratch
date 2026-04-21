import pytest

from src.tokenizer.bpe_from_scratch import BPETokenizer


def test_initial_vocab_is_byte_alphabet():
    bpe = BPETokenizer()

    assert len(bpe.vocab) == 256
    assert bpe.merges == {}
    assert all(bpe.vocab[i] == bytes([i]) for i in range(256))


def test_encode_without_training_returns_raw_bytes():
    bpe = BPETokenizer()
    text = "hello"

    assert bpe.encode(text) == list(text.encode("utf-8"))


def test_train_then_round_trip():
    bpe = BPETokenizer()
    text = "the quick brown fox jumps over the lazy dog. " \
           "the quick brown fox is quick and brown."

    bpe.train(text, vocab_size=300)

    assert bpe.decode(bpe.encode(text)) == text


def test_train_compresses_repetitive_text():
    bpe = BPETokenizer()
    text = "ababababababababab"

    bpe.train(text, vocab_size=260)
    ids = bpe.encode(text)

    assert len(ids) < len(text.encode("utf-8"))


def test_train_grows_vocab_and_merges():
    bpe = BPETokenizer()
    text = "abcabcabcabcabcabc"

    bpe.train(text, vocab_size=270)

    assert len(bpe.merges) <= 270 - 256
    assert len(bpe.vocab) == 256 + len(bpe.merges)


def test_train_rejects_vocab_smaller_than_byte_alphabet():
    bpe = BPETokenizer()

    with pytest.raises(AssertionError):
        bpe.train("hello", vocab_size=100)


def test_decode_handles_unicode_round_trip():
    bpe = BPETokenizer()
    text = "café naïve café naïve"

    bpe.train(text, vocab_size=280)

    assert bpe.decode(bpe.encode(text)) == text


def test_merge_static_method_replaces_pair():
    out = BPETokenizer._merge([1, 2, 3, 1, 2, 4], (1, 2), 99)

    assert out == [99, 3, 99, 4]


def test_train_stops_when_no_pairs_left():
    bpe = BPETokenizer()
    bpe.train("a", vocab_size=300)

    assert len(bpe.merges) == 0
