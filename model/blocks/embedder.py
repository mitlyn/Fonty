import torch.nn as nn
from torch import tensor, cat, int64, float32
from torch.nn.functional import one_hot

# *----------------------------------------------------------------------------*


class LinearEmbedder(nn.Module):
    def __init__(self, filters=64):
        super(LinearEmbedder, self).__init__()

        self.filters = filters

        self.model = nn.Sequential(
            nn.Linear(114, filters**3 // 4),
            nn.LeakyReLU(inplace=True),
            nn.Unflatten(0, (1, filters * 4, filters // 4, filters // 4)),
        )

        self.num_classes = tensor((6, 16, 12, 10, 10, 11, 12, 16, 14, 8), dtype=int64)


    def forward(self, panose):
        encoded = cat([
            one_hot(panose[i], num_classes=self.num_classes[i])
            for i in range(10)
        ]).to(float32)

        return self.model(encoded)


# *----------------------------------------------------------------------------*


class AttentionEmbedder(nn.Module):
    def __init__(self, filters=64):
        super(AttentionEmbedder, self).__init__()

        self.filters = filters

        self.model = nn.Sequential(
            nn.Linear(114, filters**3 // 4),
            nn.LeakyReLU(inplace=True),
            nn.Unflatten(0, (1, filters * 4, filters // 4, filters // 4)),
        )

        self.num_classes = tensor((6, 16, 12, 10, 10, 11, 12, 16, 14, 8), dtype=int64)


    def forward(self, panose):
        encoded = cat([
            one_hot(panose[i], num_classes=self.num_classes[i])
            for i in range(10)
        ]).to(float32)

        return self.model(encoded)