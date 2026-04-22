import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA), as used in DeepSeek-V2/V3.
    Instead of caching full K and V per head, MLA compresses the KV
    signal into a low-rank latent vector c_kv of size d_c_kv, which is
    the only thing that needs to be cached at inference. K and V are
    reconstructed from c_kv via up-projections. Queries are optionally
    compressed through a separate latent c_q.

    This is a simplified version without the decoupled RoPE branch used
    in the full DeepSeek paper; positional information is handled by the
    learned positional embedding at the GPT level, consistent with the
    sibling attention implementations in this repo.
    """
    def __init__(self, d_in, d_out, n_heads, context_len,
                 d_c_kv=None, d_c_q=None,
                 dropout=0.0, qkv_bias=False):
        super().__init__()

        assert d_out % n_heads == 0, 'd_out should be able to divide by n_heads'
        self.d_out = d_out
        self.n_heads = n_heads
        self.d_head = d_out // n_heads
        self.scale = self.d_head ** 0.5

        # defaults roughly follow DeepSeek-V2: d_c_kv ~ 4 * d_head, d_c_q ~ 12 * d_head
        self.d_c_kv = d_c_kv if d_c_kv is not None else max(self.d_head * 4, 1)
        self.d_c_q = d_c_q if d_c_q is not None else max(self.d_head * 12, d_out)

        # Q path: down-project to latent c_q, then up-project to full multi-head Q
        self.W_DQ = nn.Linear(d_in, self.d_c_q, bias=qkv_bias)
        self.W_UQ = nn.Linear(self.d_c_q, d_out, bias=qkv_bias)

        # KV path: down-project once to latent c_kv, up-project separately to K and V
        self.W_DKV = nn.Linear(d_in, self.d_c_kv, bias=qkv_bias)
        self.W_UK = nn.Linear(self.d_c_kv, d_out, bias=qkv_bias)
        self.W_UV = nn.Linear(self.d_c_kv, d_out, bias=qkv_bias)

        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_len, context_len, dtype=torch.bool), diagonal=1)
        )
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_out, d_in)


    def forward(self, x):
        B, N, C = x.shape

        # compress then expand Q
        c_q = self.W_DQ(x)              # B, N, d_c_q
        q = self.W_UQ(c_q)              # B, N, d_out

        # compress KV once (this c_kv is what would be cached at inference)
        c_kv = self.W_DKV(x)            # B, N, d_c_kv
        k = self.W_UK(c_kv)             # B, N, d_out
        v = self.W_UV(c_kv)             # B, N, d_out

        # B, H, N, d_head
        q = q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        attn_score = (q @ k.transpose(-2, -1)) / self.scale
        attn_score = attn_score.masked_fill(self.mask[:N, :N], -torch.inf)
        attn_weight = torch.softmax(attn_score, dim=-1)
        attn_weight = self.dropout(attn_weight)

        context_vec = (attn_weight @ v).transpose(1, 2).contiguous().view(B, N, self.d_out)

        return self.out_proj(context_vec)


class LayerNormalization(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(cfg['d_model']))
        self.shift = nn.Parameter(torch.zeros(cfg['d_model']))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        return ((x - mean) / (torch.sqrt(var + self.eps))) * self.scale + self.shift


class FeedForwardNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(cfg['d_model'], cfg['d_model'] * 4),
            nn.GELU(approximate='tanh'),
            nn.Linear(cfg['d_model'] * 4, cfg['d_model'])
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.ln1 = LayerNormalization(cfg)
        self.attn = MultiHeadLatentAttention(
            d_in=cfg['d_model'],
            d_out=cfg['d_model'],
            n_heads=cfg['n_heads'],
            context_len=cfg['context_len'],
            d_c_kv=cfg.get('d_c_kv'),
            d_c_q=cfg.get('d_c_q'),
            dropout=cfg['dropout'],
            qkv_bias=cfg['qkv_bias']
        )
        self.dropout = nn.Dropout(cfg['dropout'])

        self.ln2 = LayerNormalization(cfg)
        self.ffn = FeedForwardNetwork(cfg)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.context_len = cfg['context_len']
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['d_model'])
        self.pos_emb = nn.Embedding(cfg['context_len'], cfg['d_model'])

        self.dropout = nn.Dropout(cfg['dropout'])

        self.layers = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )

        self.ln = LayerNormalization(cfg)

        self.out_head = nn.Linear(cfg['d_model'], cfg['vocab_size'], bias=False)

        self.apply(self._init_weights)
        residual_std = 0.02 / (2 * cfg['n_layers']) ** 0.5
        for name, p in self.named_parameters():
            if name.endswith('attn.out_proj.weight') or name.endswith('ffn.layers.2.weight'):
                nn.init.normal_(p, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx_s):
        B, N = idx_s.shape

        assert N <= self.context_len, "Input sequence length should be smaller than the model maximum context length"

        tok_emb = self.tok_emb(idx_s)
        pos_emb = self.pos_emb(torch.arange(N, device=idx_s.device))

        x = self.dropout(tok_emb + pos_emb)

        x = self.layers(x)

        return self.out_head(x)
