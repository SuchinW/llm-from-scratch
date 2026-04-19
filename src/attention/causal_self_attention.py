import torch
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    def __init__(self, d_in, d_out, context_len, dropout = 0.0, qkv_bias = False):
        super().__init__()
        self.d_out = d_out
        self.d_in = d_in
        self.scale = d_out ** 0.5

        #fused qkv
        self.qkv = nn.Linear(d_in, d_out * 3, bias = qkv_bias)

        #causal masking
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_len, context_len, dtype=torch.bool), diagonal = 1)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, _ = x.shape

        qkv = self.qkv(x)                           #B, N, d_out * 3
        q, k, v = qkv.chunk(3, dim=-1)              #B, N, d_out

        attn_score = q @ k.transpose(-2, -1) / self.scale                   #B, N, N
        attn_score = attn_score.masked_fill(self.mask[:N, :N], -torch.inf)  #B, N, N

        attn_weight = torch.softmax(attn_score, dim=-1)                     #B, N, N
        attn_weight = self.dropout(attn_weight)                             #B, N, N

        context_vec = attn_weight @ v                                       #B, N, d_out

        return context_vec                                                  #B, N, d_out

