import pytest

pytest.importorskip("tiktoken")

from src.tokenizer.bpe import TiktokenBPE


def test_encode_returns_list_of_ints():
    tok = TiktokenBPE("gpt2")
    ids = tok.encode("hello world")

    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert len(ids) > 0


def test_round_trip_preserves_text():
    tok = TiktokenBPE("gpt2")
    text = "the quick brown fox jumps over the lazy dog."

    assert tok.decode(tok.encode(text)) == text


def test_round_trip_with_unicode():
    tok = TiktokenBPE("gpt2")
    text = "café — naïve façade 🚀"

    assert tok.decode(tok.encode(text)) == text


def test_empty_string_round_trips():
    tok = TiktokenBPE("gpt2")

    assert tok.encode("") == []
    assert tok.decode([]) == ""


def test_different_encoding_name_works():
    tok = TiktokenBPE("cl100k_base")
    text = "hello"

    assert tok.decode(tok.encode(text)) == text
