import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_len, dropout=0.0, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.qkv = nn.Linear(d_in, d_out * 3, bias=qkv_bias)

        self.scale = self.head_dim ** 0.5

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_len, context_len, dtype=torch.bool), diagonal=1)
        )
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_out, d_in)

    def forward(self, x):
        B, N, _ = x.shape           #B, N, d_in

        qkv = self.qkv(x)           #B, N, d_out * 3
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)            #B, H, N, head_dim
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)            #B, H, N, head_dim
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)            #B, H, N, head_dim

        attn_score = q @ k.transpose(-2, -1) / self.scale                          #B, H, N, N
        attn_score = attn_score.masked_fill(self.mask[:N, :N], -torch.inf)         #B, H, N, N

        attn_weight = torch.softmax(attn_score, dim=-1)                            #B, H, N, N
        attn_weight = self.dropout(attn_weight)                                    #B, H, N, N

        context_vec = attn_weight @ v                                              #B, H, N, head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(B, N, self.d_out)  #B, N, d_out

        return self.out_proj(context_vec)                                          #B, N, d_in
