import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()

        self.w_gate_up = nn.Linear(d_model, d_hidden * 2, bias =False)
        self.w_down =  nn.Linear(d_hidden, d_model, bias= False)

    def forward(self, x):
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)

        return self.w_down(F.silu(gate) * up)

class FeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()

        d_hidden = int(d_model * 8 /3)
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            SwiGLU(d_model, d_hidden),
            nn.Linear(d_hidden, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.layers(x)