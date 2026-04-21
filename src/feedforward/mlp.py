import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x):
        return self.layers(x)
