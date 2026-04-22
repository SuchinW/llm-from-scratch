import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    """
    Multi-head attention with an in-module KV cache for efficient
    autoregressive decoding. During generation, only the newly produced
    tokens are fed in each step; keys/values for past tokens are read
    from the cache instead of being recomputed.
    """
    def __init__(self, d_in, d_out, n_heads, context_len, dropout = 0.0, qkv_bias = False):
        super().__init__()

        assert d_out % n_heads == 0, 'd_out should be able to divide by n_heads'
        self.qkv = nn.Linear(d_in, d_out * 3, bias = qkv_bias)
        self.d_out = d_out
        self.n_heads = n_heads
        self.d_head = d_out // n_heads
        self.scale = self.d_head ** 0.5

        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_len, context_len,  dtype = torch.bool), diagonal=1)
        )
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_out, d_in)

        self.k_cache = None
        self.v_cache = None


    def reset_cache(self):
        self.k_cache = None
        self.v_cache = None


    def forward(self, x, use_cache = False):
        B, N, C  = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # B, H, N, d_head
        q = q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        if use_cache:
            if self.k_cache is not None:
                # append new K/V to the cache along the sequence dimension
                k = torch.cat([self.k_cache, k], dim=2)
                v = torch.cat([self.v_cache, v], dim=2)
            self.k_cache = k
            self.v_cache = v

        # total key length (may be > N when cache is populated)
        T_k = k.size(2)
        # current queries correspond to absolute positions [q_start, q_start + N)
        q_start = T_k - N

        attn_score = (q @ k.transpose(-2, -1)) / self.scale
        # slice the causal mask so rows align with current queries and cols with all keys
        attn_score = attn_score.masked_fill(self.mask[q_start:q_start + N, :T_k], -torch.inf)
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
        return ((x - mean) / (torch.sqrt(var + self.eps))) * self.scale +  self.shift


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
        self.attn = MultiheadAttention(
            d_in=cfg['d_model'],
            d_out=cfg['d_model'],
            n_heads=cfg['n_heads'],
            context_len=cfg['context_len'],
            dropout=cfg['dropout'],
            qkv_bias=cfg['qkv_bias']
        )
        self.dropout = nn.Dropout(cfg['dropout'])

        self.ln2 = LayerNormalization(cfg)
        self.ffn = FeedForwardNetwork(cfg)

    def forward(self, x, use_cache = False):
        x = x + self.dropout(self.attn(self.ln1(x), use_cache=use_cache))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.context_len = cfg['context_len']
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['d_model'])
        self.pos_emb = nn.Embedding(cfg['context_len'], cfg['d_model'])

        self.dropout = nn.Dropout(cfg['dropout'])

        # ModuleList (not Sequential) so we can thread use_cache through each block
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )

        self.ln =  LayerNormalization(cfg)

        self.out_head = nn.Linear(cfg['d_model'], cfg['vocab_size'], bias=False)

        # tracks the next position index when decoding with cache
        self._cache_pos = 0

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

    def reset_cache(self):
        self._cache_pos = 0
        for block in self.layers:
            block.attn.reset_cache()

    def forward(self, idx_s, use_cache = False):
        B, N = idx_s.shape

        pos_start = self._cache_pos if use_cache else 0
        assert pos_start + N <= self.context_len, "Input sequence length should be smaller than the model maximum context length"

        tok_emb = self.tok_emb(idx_s)
        pos_emb = self.pos_emb(torch.arange(pos_start, pos_start + N, device = idx_s.device))

        x = self.dropout(tok_emb + pos_emb)

        for block in self.layers:
            x = block(x, use_cache=use_cache)

        if use_cache:
            self._cache_pos = pos_start + N

        return self.out_head(x)

    @torch.no_grad()
    def generate(self, idx_s, max_new_tokens, temperature = 1.0, top_k = None):
        """
        Greedy/temperature sampling loop using the KV cache: the full prompt
        is fed once to prime the cache, then each new token is fed one at a
        time so attention only recomputes for that single position.
        """
        self.eval()
        self.reset_cache()

        # prime the cache with the whole prompt
        logits = self(idx_s, use_cache=True)
        out = idx_s

        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(next_logits, top_k)
                next_logits = next_logits.masked_fill(next_logits < v[:, [-1]], -torch.inf)
            probs = torch.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)

            out = torch.cat([out, next_tok], dim=1)
            if self._cache_pos >= self.context_len:
                break

            # feed only the new token; cache supplies the rest
            logits = self(next_tok, use_cache=True)

        return out
