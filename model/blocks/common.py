import torch.nn as nn

# *----------------------------------------------------------------------------* Residual Decorator

class Residual(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()

        self.block = block

    def forward(self, X):
        return X + self.block(X)


# *----------------------------------------------------------------------------* PreNorm Decorator


class Normalized(nn.Module):
    def __init__(self, dim: int, block: nn.Module):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.block = block

    def forward(self, X, **kwargs):
        return self.block(self.norm(X), **kwargs)

