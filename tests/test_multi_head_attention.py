import pytest
import torch
import torch.nn.functional as F

from src.attention.multi_head_attention import MultiHeadAttention


def test_output_shape_matches_input():
    B, N, d_in, d_out, num_heads = 2, 6, 12, 12, 4
    attn = MultiHeadAttention(d_in=d_in, d_out=d_out, num_heads=num_heads, context_len=N)
    x = torch.randn(B, N, d_in)

    out = attn(x)

    assert out.shape == (B, N, d_in)


def test_d_out_not_divisible_by_num_heads_raises():
    with pytest.raises(AssertionError):
        MultiHeadAttention(d_in=12, d_out=10, num_heads=4, context_len=6)


def test_matches_scaled_dot_product_attention():
    torch.manual_seed(0)
    B, N, d_in, d_out, num_heads = 2, 5, 16, 16, 4
    attn = MultiHeadAttention(
        d_in=d_in, d_out=d_out, num_heads=num_heads, context_len=N, dropout=0.0
    )
    attn.eval()
    x = torch.randn(B, N, d_in)

    qkv = attn.qkv(x)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(B, N, num_heads, attn.head_dim).transpose(1, 2)
    k = k.view(B, N, num_heads, attn.head_dim).transpose(1, 2)
    v = v.view(B, N, num_heads, attn.head_dim).transpose(1, 2)

    reference = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    reference = reference.transpose(1, 2).contiguous().view(B, N, d_out)
    reference = attn.out_proj(reference)

    ours = attn(x)

    assert torch.allclose(ours, reference, atol=1e-6)


def test_attention_is_causal():
    B, N, d_in, d_out, num_heads = 2, 5, 16, 16, 4
    attn = MultiHeadAttention(
        d_in=d_in, d_out=d_out, num_heads=num_heads, context_len=N, dropout=0.0
    )
    x = torch.randn(B, N, d_in)

    qkv = attn.qkv(x)
    q, k, _ = qkv.chunk(3, dim=-1)
    q = q.view(B, N, num_heads, attn.head_dim).transpose(1, 2)
    k = k.view(B, N, num_heads, attn.head_dim).transpose(1, 2)
    scores = q @ k.transpose(-2, -1) / attn.scale
    scores = scores.masked_fill(attn.mask[:N, :N], -torch.inf)
    weights = torch.softmax(scores, dim=-1)

    upper_triangle = torch.triu(weights, diagonal=1)
    assert torch.all(upper_triangle == 0)


def test_handles_sequence_shorter_than_context_len():
    context_len = 16
    B, N, d_in, d_out, num_heads = 2, 4, 12, 12, 4
    attn = MultiHeadAttention(
        d_in=d_in, d_out=d_out, num_heads=num_heads, context_len=context_len
    )
    x = torch.randn(B, N, d_in)

    out = attn(x)

    assert out.shape == (B, N, d_in)


def test_gradients_flow_to_all_parameters():
    B, N, d_in, d_out, num_heads = 2, 4, 12, 12, 4
    attn = MultiHeadAttention(d_in=d_in, d_out=d_out, num_heads=num_heads, context_len=N)
    x = torch.randn(B, N, d_in)

    out = attn(x)
    out.sum().backward()

    for name, param in attn.named_parameters():
        assert param.grad is not None, f"{name} has no grad"
        assert param.grad.abs().sum() > 0, f"{name} grad is all zero"
