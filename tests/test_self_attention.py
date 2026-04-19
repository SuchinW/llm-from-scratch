import torch

from src.attention.self_attention import SelfAttention


def test_output_shape_matches_input():
    B, N, d_in, d_out = 2, 4, 8, 8
    attn = SelfAttention(d_in=d_in, d_out=d_out, context_len=N)
    x = torch.randn(B, N, d_in)

    out = attn(x)

    assert out.shape == (B, N, d_out)


def test_attention_weights_sum_to_one():
    B, N, d_in, d_out = 2, 5, 8, 8
    attn = SelfAttention(d_in=d_in, d_out=d_out, context_len=N)
    x = torch.randn(B, N, d_in)

    qkv = attn.qkv(x)
    q, k, _ = qkv.chunk(3, dim=-1)
    scores = q @ k.transpose(-2, -1) / attn.scale
    weights = torch.softmax(scores, dim=-1)

    assert torch.allclose(weights.sum(dim=-1), torch.ones(B, N), atol=1e-6)


def test_gradients_flow_to_all_parameters():
    B, N, d_in, d_out = 2, 4, 8, 8
    attn = SelfAttention(d_in=d_in, d_out=d_out, context_len=N)
    x = torch.randn(B, N, d_in)

    out = attn(x)
    out.sum().backward()

    for name, param in attn.named_parameters():
        assert param.grad is not None, f"{name} has no grad"
        assert param.grad.abs().sum() > 0, f"{name} grad is all zero"
