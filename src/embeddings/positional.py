import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.emd = nn.Embedding(max_seq_len, d_model)

    def forward(self, x):
        B, N, _ = x.shape

        assert N <= self.max_seq_len, "Input sequence should be smaller than the maximum sequence length"

        pos = self.emd(torch.arange(N))
        print(pos)

        return x + pos
