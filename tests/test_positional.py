import pytest
import torch

from src.embeddings.positional import PositionalEmbedding


def test_output_shape_matches_input():
    B, N, d_model = 2, 4, 8
    pos = PositionalEmbedding(max_seq_len=16, d_model=d_model)
    x = torch.randn(B, N, d_model)

    out = pos(x)

    assert out.shape == (B, N, d_model)


def test_adds_same_position_vector_across_batch():
    B, N, d_model = 3, 5, 8
    pos = PositionalEmbedding(max_seq_len=16, d_model=d_model)
    x = torch.zeros(B, N, d_model)

    out = pos(x)

    for b in range(1, B):
        assert torch.allclose(out[0], out[b])


def test_different_positions_get_different_embeddings():
    N, d_model = 6, 8
    pos = PositionalEmbedding(max_seq_len=16, d_model=d_model)
    x = torch.zeros(1, N, d_model)

    out = pos(x)[0]

    for i in range(N):
        for j in range(i + 1, N):
            assert not torch.allclose(out[i], out[j])


def test_handles_sequence_shorter_than_max_seq_len():
    B, N, d_model = 2, 4, 8
    pos = PositionalEmbedding(max_seq_len=32, d_model=d_model)
    x = torch.randn(B, N, d_model)

    out = pos(x)

    assert out.shape == (B, N, d_model)


def test_sequence_longer_than_max_seq_len_raises():
    B, N, d_model = 2, 10, 8
    pos = PositionalEmbedding(max_seq_len=8, d_model=d_model)
    x = torch.randn(B, N, d_model)

    with pytest.raises(AssertionError):
        pos(x)


def test_gradients_flow_to_all_parameters():
    B, N, d_model = 2, 4, 8
    pos = PositionalEmbedding(max_seq_len=16, d_model=d_model)
    x = torch.randn(B, N, d_model)

    out = pos(x)
    out.sum().backward()

    for name, param in pos.named_parameters():
        assert param.grad is not None, f"{name} has no grad"
        assert param.grad.abs().sum() > 0, f"{name} grad is all zero"
