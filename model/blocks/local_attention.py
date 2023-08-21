import torch
import torch.nn as nn

from model.blocks import *

# *----------------------------------------------------------------------------*


class LocalAttention(nn.Module):
    def __init__(self, filters: int = 64):
        super(LocalAttention, self).__init__()
        self.SA = SelfAttention(filters * 4)
        self.attention = nn.Linear(filters * 4, 100)
        self.context_vec = nn.Linear(100, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features):
        B, C, H, W = features.shape

        h = self.SA(features)
        h = h.permute(0, 2, 3, 1).reshape(-1, C)
        h = torch.tanh(self.attention(h))
        h = self.context_vec(h)

        attention_map = self.softmax(h.view(B, H * W)).view(B, 1, H, W)

        return torch.sum(features * attention_map, dim=[2, 3])
