import torch, torch.nn as nn

from model.blocks import *

# *----------------------------------------------------------------------------*


class LayerAttention(nn.Module):
    def __init__(self, ngf=64):
        super(LayerAttention, self).__init__()

        self.ngf = ngf

        self.FC = nn.Linear(4096, 3)
        self.SM  = nn.Softmax(dim=1)

    def forward(self, style_features, style_features_1, style_features_2, style_features_3, B, K):
        style_features = torch.mean(style_features.view(B, K, self.ngf * 4, 4, 4), dim=1)
        style_features = style_features.view(B, -1)

        weight = self.SM(self.FC(style_features))

        style_features_1 = torch.mean(style_features_1.view(B, K, self.ngf * 4), dim=1)
        style_features_2 = torch.mean(style_features_2.view(B, K, self.ngf * 4), dim=1)
        style_features_3 = torch.mean(style_features_3.view(B, K, self.ngf * 4), dim=1)

        style_features = (
            style_features_1 * weight.narrow(1, 0, 1) +
            style_features_2 * weight.narrow(1, 1, 1) +
            style_features_3 * weight.narrow(1, 2, 1)
        ).view(B, self.ngf * 4, 1, 1) + torch.randn([B, self.ngf * 4, 16, 16], device='cuda') * 0.02

        return style_features