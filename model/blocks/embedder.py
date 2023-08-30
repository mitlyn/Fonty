import torch.nn as nn
from torch import tensor, cat, int64, float32
from torch.nn.functional import one_hot

from model.blocks import *

# TODO: Panose Batches
# TODO: CLS Token

# *----------------------------------------------------------------------------*

# Number of categories for each panose feature.
classes = tensor((6, 16, 12, 10, 10, 11, 12, 16, 14, 8), dtype=int64)

# *----------------------------------------------------------------------------* Linear Embedder

class LinearEmbedder(nn.Module):
    def __init__(self, filters=64, blocks=2):
        super(LinearEmbedder, self).__init__()

        model = [Residual(EmbedderBlock(115)) for _ in range(blocks)]

        model += [
            nn.Linear(115, filters * 4), # { filters**3 // 4 | filters * 4}
            nn.LeakyReLU(inplace=True),
            # nn.Unflatten(0, (1, filters * 4, filters // 4, filters // 4))
        ]

        self.model = nn.Sequential(*model)


    def forward(self, panose):
        encoded = cat([
            one_hot(
                panose[i],
                num_classes=classes[i]
            ) for i in range(10)
        ]).to(float32)

        return self.model(encoded)


# *----------------------------------------------------------------------------* Attention Embedder

""" [Work In Progress]

class FeedForward(nn.Module):
    def __init__(self, dim: int = 10, latent: int = 64, dropout=0.):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(dim, latent),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.model(x)


class Attention(nn.Module):
    def __init__(self, dim: int = 10, heads: int = 2, dropout: float = 0.1):
        super(Attention, self).__init__()

        self.Q = nn.Linear(dim, dim)
        self.K = nn.Linear(dim, dim)
        self.V = nn.Linear(dim, dim)

        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout)

    def forward(self, X):
        Q = self.Q(X)
        K = self.K(X)
        V = self.V(X)

        Y, W = self.attention(Q, K, V)

        return Y


class AttentionEmbedder(nn.Module):
    def __init__(self, filters=64, layers=2):
        super(AttentionEmbedder, self).__init__()

        model = []

        for _ in range(layers):
            model += [
                Residual(Normal(10, Attention(10))),
                Residual(Normal(10, FeedForward(10, filters))),
            ]

        model += [
            nn.LayerNorm(10),
            nn.Linear(10, 64)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, panose):
        X = panose / classes

        # cls
        # X = cat([cls, X])
        # X += pos encoding

        X = self.model(X)
"""