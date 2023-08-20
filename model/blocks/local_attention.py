import torch
import torch.nn as nn

from model.blocks import *

# *----------------------------------------------------------------------------*


class LocalAttention(nn.Module):
    def __init__(self, ngf=64):
        super(LocalAttention, self).__init__()
        self.SA = SelfAttention(ngf*4)
        self.attention = nn.Linear(ngf*4, 100)
        self.context_vec = nn.Linear(100, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, style_features):
        B, C, H, W = style_features.shape

        h = self.SA(style_features)
        h = h.permute(0, 2, 3, 1).reshape(-1, C)
        h = torch.tanh(self.attention(h))                               # (B*H*W, 100)
        h = self.context_vec(h)                                         # (B*H*W, 1)
        attention_map = self.softmax(h.view(B, H*W)).view(B, 1, H, W)   # (B, 1, H, W)

        return torch.sum(style_features*attention_map, dim=[2, 3])
