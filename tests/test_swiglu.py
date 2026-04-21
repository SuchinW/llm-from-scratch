import torch
import torch.nn.functional as F

from src.feedforward.swiglu import SwiGLU


def test_output_shape_matches_input():
    B, N, d_model, d_hidden = 2, 4, 16, 32
    swiglu = SwiGLU(d_model, d_hidden)
    x = torch.randn(B, N, d_model)

    out = swiglu(x)

    assert out.shape == (B, N, d_model)


def test_no_bias_in_projections():
    swiglu = SwiGLU(d_model=8, d_hidden=16)

    assert swiglu.w_gate_up.bias is None
    assert swiglu.w_down.bias is None


def test_gate_up_projection_doubles_hidden():
    d_model, d_hidden = 8, 16
    swiglu = SwiGLU(d_model, d_hidden)

    assert swiglu.w_gate_up.out_features == d_hidden * 2
    assert swiglu.w_down.in_features == d_hidden
    assert swiglu.w_down.out_features == d_model


def test_matches_manual_swiglu_computation():
    torch.manual_seed(0)
    d_model, d_hidden = 4, 8
    swiglu = SwiGLU(d_model, d_hidden)
    x = torch.randn(2, 3, d_model)

    out = swiglu(x)

    gate, up = swiglu.w_gate_up(x).chunk(2, dim=-1)
    expected = swiglu.w_down(F.silu(gate) * up)

    assert torch.allclose(out, expected)


def test_gradients_flow_to_all_parameters():
    swiglu = SwiGLU(d_model=8, d_hidden=16)
    x = torch.randn(2, 3, 8)

    swiglu(x).sum().backward()

    for name, param in swiglu.named_parameters():
        assert param.grad is not None, f"{name} has no grad"
        assert param.grad.abs().sum() > 0, f"{name} grad is all zero"
