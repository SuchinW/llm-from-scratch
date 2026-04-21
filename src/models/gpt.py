import torch
import torch.nn as nn

from src.normalization.layer_norm import LayerNorm
from src.attention.multi_head_attention import MultiHeadAttention
from src.feedforward.mlp import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, context_len, dropout=0.0):
        super().__init__()

        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, d_model, n_head, context_len, dropout)
        self.drop = nn.Dropout(dropout)

        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x

class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, context_len, n_blocks, dropout=0.0):
        super().__init__()

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.trn_blocks = nn.Sequential(
            *[TransformerBlock(d_model, n_head, context_len, dropout) for _ in range(n_blocks)]
        )
        self.ln = LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias = False)

    def forward(self, x):
        B, N = x.shape
        x = self.tok_emb(x)
        x = x + self.pos_emb(torch.arange(N, device = x.device))
        x = self.drop(x)
        x = self.trn_blocks(x)

        return self.output_head(self.ln(x))


