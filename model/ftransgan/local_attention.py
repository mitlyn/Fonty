import torch.nn as nn
from torch import tanh, sum

from model.ftransgan import SelfAttention

# *----------------------------------------------------------------------------*

class LocalAttention(nn.Module):
    def __init__(self, filters: int = 64):
        super(LocalAttention, self).__init__()

        self.self_attention = SelfAttention(filters * 4)
        self.attention = nn.Linear(filters * 4, 100)
        self.context = nn.Linear(100, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features):
        B, C, H, W = features.shape

        h = self.self_attention(features)
        h = h.permute(0, 2, 3, 1).reshape(-1, C)
        h = tanh(self.attention(h))
        h = self.context(h)

        attention_map = self.softmax(h.view(B, H * W)).view(B, 1, H, W)

        return sum(features * attention_map, dim=[2, 3])
