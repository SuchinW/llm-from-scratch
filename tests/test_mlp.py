import torch

from src.feedforward.mlp import FeedForward


def test_output_shape_matches_input():
    B, N, d_model = 2, 5, 16
    ff = FeedForward(d_model)
    x = torch.randn(B, N, d_model)

    out = ff(x)

    assert out.shape == (B, N, d_model)


def test_default_d_ff_is_4x_d_model():
    d_model = 8
    ff = FeedForward(d_model)

    first_linear = ff.layers[0]
    second_linear = ff.layers[2]

    assert first_linear.out_features == d_model * 4
    assert second_linear.in_features == d_model * 4


def test_custom_d_ff_is_respected():
    d_model, d_ff = 8, 32
    ff = FeedForward(d_model, d_ff=d_ff)

    assert ff.layers[0].out_features == d_ff
    assert ff.layers[2].in_features == d_ff


def test_gradients_flow_to_all_parameters():
    ff = FeedForward(8)
    x = torch.randn(2, 4, 8)

    out = ff(x)
    out.sum().backward()

    for name, param in ff.named_parameters():
        assert param.grad is not None, f"{name} has no grad"
        assert param.grad.abs().sum() > 0, f"{name} grad is all zero"


def test_dropout_zero_is_deterministic():
    torch.manual_seed(0)
    ff = FeedForward(8, dropout=0.0)
    x = torch.randn(2, 3, 8)

    a = ff(x)
    b = ff(x)

    assert torch.allclose(a, b)
