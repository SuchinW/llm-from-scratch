import torch
import torch.nn as nn

from src.normalization.layer_norm import LayerNorm


def test_output_shape_matches_input():
    B, N, d_model = 2, 4, 8
    ln = LayerNorm(d_model)
    x = torch.randn(B, N, d_model)

    out = ln(x)

    assert out.shape == (B, N, d_model)


def test_normalizes_to_zero_mean_unit_var():
    B, N, d_model = 2, 4, 16
    ln = LayerNorm(d_model)
    x = torch.randn(B, N, d_model) * 5 + 3

    out = ln(x)

    assert torch.allclose(out.mean(dim=-1), torch.zeros(B, N), atol=1e-5)
    assert torch.allclose(out.var(dim=-1, unbiased=False), torch.ones(B, N), atol=1e-4)


def test_default_params_are_identity_after_normalization():
    d_model = 8
    ln = LayerNorm(d_model)

    assert torch.allclose(ln.scale, torch.ones(d_model))
    assert torch.allclose(ln.shift, torch.zeros(d_model))


def test_scale_and_shift_are_applied():
    B, N, d_model = 2, 3, 8
    ln = LayerNorm(d_model)
    with torch.no_grad():
        ln.scale.fill_(2.0)
        ln.shift.fill_(1.0)

    x = torch.randn(B, N, d_model)
    out = ln(x)

    assert torch.allclose(out.mean(dim=-1), torch.ones(B, N), atol=1e-5)
    assert torch.allclose(out.var(dim=-1, unbiased=False), torch.full((B, N), 4.0), atol=1e-3)


def test_matches_torch_layernorm():
    B, N, d_model = 2, 4, 16
    ln = LayerNorm(d_model)
    ref = nn.LayerNorm(d_model, eps=1e-5)
    x = torch.randn(B, N, d_model)

    ours = ln(x)
    theirs = ref(x)

    assert torch.allclose(ours, theirs, atol=1e-5)


def test_handles_constant_input_without_nan():
    B, N, d_model = 2, 4, 8
    ln = LayerNorm(d_model)
    x = torch.full((B, N, d_model), 3.14)

    out = ln(x)

    assert torch.all(torch.isfinite(out))


def test_gradients_flow_to_all_parameters():
    B, N, d_model = 2, 4, 8
    ln = LayerNorm(d_model)
    x = torch.randn(B, N, d_model)

    out = ln(x)
    out.sum().backward()

    for name, param in ln.named_parameters():
        assert param.grad is not None, f"{name} has no grad"
        assert param.grad.abs().sum() > 0, f"{name} grad is all zero"
