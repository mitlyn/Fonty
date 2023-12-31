import torch.nn as nn

# *----------------------------------------------------------------------------* Residual Trait

class Residual(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()

        self.block = block

    def forward(self, X):
        return X + self.block(X)

# *----------------------------------------------------------------------------* Normalized Trait

class Normal(nn.Module):
    def __init__(self, dim: int, block: nn.Module):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.block = block

    def forward(self, X, **kwargs):
        return self.block(self.norm(X), **kwargs)

