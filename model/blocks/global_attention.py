import torch, torch.nn as nn

from model.blocks import *

# *----------------------------------------------------------------------------*


class GlobalAttention(nn.Module):
    def __init__(self, ngf=64):
        super(GlobalAttention, self).__init__()
        self.ngf = ngf

    def forward(self, style_features, B, K):
        style_features = style_features.view(B, K, self.ngf * 4)

        style_features = torch.mean(style_features, dim=1).view(B, self.ngf * 4, 1, 1) # TBD

        style_features = style_features + torch.randn([B, self.ngf * 4, 16, 16], device='cuda') * 0.02

        return style_features
