import torch
import torch.nn as nn

from model.blocks import *

# *----------------------------------------------------------------------------*


class Generator(nn.Module):
    def __init__(self, filters: int = 64, blocks: int = 6, dropout: bool = False):
        super(Generator, self).__init__()

        self.Es = Encoder(dim=1, filters=filters)
        self.Ec = Encoder(dim=1, filters=filters)
        self.D = Decoder(filters=filters, blocks=blocks, dropout=dropout)

        self.A1 = LocalAttention(filters=filters)
        self.A2 = LocalAttention(filters=filters)
        self.A3 = LocalAttention(filters=filters)
        self.LA = LayerAttention(filters=filters)

        self.downsample_1 = nn.Sequential(
            nn.Conv2d(filters * 4, filters * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(filters * 4),
            nn.ReLU(True)
        )

        self.downsample_2 = nn.Sequential(
            nn.Conv2d(filters * 4, filters * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(filters * 4),
            nn.ReLU(True)
        )

    def forward(self, X):
        content_image, style_images = X
        B, K, _, _ = style_images.shape

        content_feature = self.Ec(content_image)

        style_features = self.Es(style_images.view(-1, 1, 64, 64))
        style_features_1 = self.A1(style_features)

        style_features = self.downsample_1(style_features)
        style_features_2 = self.A2(style_features)

        style_features = self.downsample_2(style_features)
        style_features_3 = self.A3(style_features)

        style_features = self.LA(style_features, style_features_1, style_features_2, style_features_3, B, K)

        features = torch.cat([content_feature, style_features], dim=1)

        return self.D(features)