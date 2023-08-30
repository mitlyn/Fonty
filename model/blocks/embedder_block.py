import torch.nn as nn

# *----------------------------------------------------------------------------*


class EmbedderBlock(nn.Module):
    def __init__(self, dim: int):
        super(EmbedderBlock, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(inplace=True) # ?? do we need leaky relu if the network is residual?
        )

    def forward(self, X):
        return self.model(X)