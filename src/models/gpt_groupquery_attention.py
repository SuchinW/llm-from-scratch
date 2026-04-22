import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA): query heads are split into n_kv_heads groups,
    each group shares one K and V head. Interpolates between full MHA
    (n_kv_heads == n_heads) and MQA (n_kv_heads == 1).
    Used in LLaMA 2/3, Mistral, etc.
    """
    def __init__(self, d_in, d_out, n_heads, n_kv_heads, context_len, dropout = 0.0, qkv_bias = False):
        super().__init__()

        assert d_out % n_heads == 0, 'd_out should be able to divide by n_heads'
        assert n_heads % n_kv_heads == 0, 'n_heads should be divisible by n_kv_heads'
        self.d_out = d_out
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.group_size = n_heads // n_kv_heads
        self.d_head = d_out // n_heads
        self.scale = self.d_head ** 0.5

        self.q_proj = nn.Linear(d_in, n_heads * self.d_head, bias = qkv_bias)
        self.k_proj = nn.Linear(d_in, n_kv_heads * self.d_head, bias = qkv_bias)
        self.v_proj = nn.Linear(d_in, n_kv_heads * self.d_head, bias = qkv_bias)

        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_len, context_len, dtype = torch.bool), diagonal=1)
        )
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_out, d_in)


    def forward(self, x):
        B, N, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # B, H,    N, d_head
        q = q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        # B, H_kv, N, d_head
        k = k.view(B, N, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.n_kv_heads, self.d_head).transpose(1, 2)

        # expand KV heads to match Q heads: repeat each KV head `group_size` times
        # B, H, N, d_head
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

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
        mean = x.mean(dim=-1, keepdim = True)
        var = x.var(dim=-1, unbiased = False, keepdim = True)
        return ((x - mean) / (torch.sqrt(var + self.eps))) * self.scale + self.shift


class FeedForwardNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(cfg['d_model'], cfg['d_model'] * 4),
            nn.GELU(approximate = 'tanh'),
            nn.Linear(cfg['d_model'] * 4, cfg['d_model'])
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.ln1 = LayerNormalization(cfg)
        self.attn = GroupedQueryAttention(
            d_in=cfg['d_model'],
            d_out=cfg['d_model'],
            n_heads=cfg['n_heads'],
            n_kv_heads=cfg['n_kv_heads'],
            context_len=cfg['context_len'],
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
        pos_emb = self.pos_emb(torch.arange(N, device = idx_s.device))

        x = self.dropout(tok_emb + pos_emb)

        x = self.layers(x)

        return self.out_head(x)
