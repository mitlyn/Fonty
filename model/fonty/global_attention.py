import torch.nn as nn
from torch import mean, randn

# *----------------------------------------------------------------------------*

class GlobalAttention(nn.Module):
    def __init__(self, filters: int = 64):
        super(GlobalAttention, self).__init__()
        self.filters = filters

    def forward(self, features, B, K):
        features = features.view(B, K, self.filters * 4)

        features = mean(features, dim=1).view(B, self.filters * 4, 1, 1)

        features = features + randn([B, self.filters * 4, 16, 16], device='cuda') * 0.02

        return features
