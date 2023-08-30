import torch.nn as nn
from torch import zeros, bmm

from model.blocks import *

# *----------------------------------------------------------------------------*


class SelfAttention(nn.Module):
    def __init__(self, dim: int):
        super(SelfAttention, self).__init__()

        self.query_conv = nn.Conv2d(dim, dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(dim, dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(dim, dim, kernel_size=1)

        self.gamma = nn.Parameter(zeros(1))
        self.softmax  = nn.Softmax(-1)

    def forward(self, X):
        """
            inputs :
                x : input feature maps(B × C × H × W)
            returns :
                out : self attention value + input feature
                attention: B × N × N (N is Width * Height)
        """
        B, C, H, W = X.size()
        N = H * W

        query = self.query_conv(X).view(B, -1, N).permute(0, 2, 1)
        key = self.key_conv(X).view(B, -1, N)
        energy = bmm(query, key)
        attention = self.softmax(energy)
        value = self.value_conv(X).view(B, -1, N)

        out = bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        out = X + out * self.gamma

        return out
