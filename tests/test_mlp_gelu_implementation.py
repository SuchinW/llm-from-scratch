import torch
import torch.nn as nn

from src.feedforward.mlp_gelu_implementation import GELU, FeedForward


def test_gelu_matches_torch_tanh_approximation():
    gelu = GELU()
    ref = nn.GELU(approximate="tanh")
    x = torch.randn(4, 16)

    assert torch.allclose(gelu(x), ref(x), atol=1e-5)


def test_gelu_zero_at_zero():
    gelu = GELU()
    x = torch.zeros(3, 3)

    assert torch.allclose(gelu(x), torch.zeros(3, 3))


def test_gelu_preserves_shape():
    gelu = GELU()
    x = torch.randn(2, 5, 8)

    assert gelu(x).shape == x.shape


def test_feedforward_output_shape_matches_input():
    B, N, d_model = 2, 4, 16
    ff = FeedForward(d_model)
    x = torch.randn(B, N, d_model)

    out = ff(x)

    assert out.shape == (B, N, d_model)


def test_feedforward_default_d_ff_is_4x():
    d_model = 8
    ff = FeedForward(d_model)

    assert ff.layers[0].out_features == d_model * 4


def test_feedforward_uses_custom_gelu():
    ff = FeedForward(8)

    assert isinstance(ff.layers[1], GELU)


def test_feedforward_gradients_flow():
    ff = FeedForward(8)
    x = torch.randn(2, 3, 8)

    ff(x).sum().backward()

    for name, param in ff.named_parameters():
        assert param.grad is not None, f"{name} has no grad"
        assert param.grad.abs().sum() > 0, f"{name} grad is all zero"
