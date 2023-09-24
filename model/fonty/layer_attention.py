import torch.nn as nn
from torch import mean, randn

# *----------------------------------------------------------------------------*

class LayerAttention(nn.Module):
    def __init__(self, filters: int = 64):
        super(LayerAttention, self).__init__()
        self.filters = filters

        self.linear = nn.Linear(4096, 3)
        self.softmax  = nn.Softmax(dim=1)

    def forward(self, features, features_1, features_2, features_3, B, K):
        features = mean(features.view(B, K, self.filters * 4, 4, 4), dim=1)
        features = features.view(B, -1)

        weight = self.softmax(self.linear(features))

        features_1 = mean(features_1.view(B, K, self.filters * 4), dim=1)
        features_2 = mean(features_2.view(B, K, self.filters * 4), dim=1)
        features_3 = mean(features_3.view(B, K, self.filters * 4), dim=1)

        features = (
            features_1 * weight.narrow(1, 0, 1) +
            features_2 * weight.narrow(1, 1, 1) +
            features_3 * weight.narrow(1, 2, 1)
        ).view(B, self.filters * 4, 1, 1) + randn([B, self.filters * 4, 16, 16], device='cuda') * 0.02

        return features