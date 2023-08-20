import torch, torch.nn as nn

from model.blocks import *

# *----------------------------------------------------------------------------*


class Generator(nn.Module):
    def __init__(self, ngf=64, n_blocks=6, dropout=False):
        super(Generator, self).__init__()

        self.Es = Encoder(input_nc=1, ngf=ngf)
        self.Ec = Encoder(input_nc=1, ngf=ngf)
        self.D = Decoder(ngf=ngf, n_blocks=n_blocks, dropout=dropout)
        self.A1 = LocalAttention(ngf=ngf)
        self.A2 = LocalAttention(ngf=ngf)
        self.A3 = LocalAttention(ngf=ngf)
        self.LA = LayerAttention(ngf=ngf)

        self.downsample_1 = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )

        self.downsample_2 = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )

    def forward(self, inp):
        content_image, style_images = inp
        B, K, _, _ = style_images.shape

        content_feature = self.Ec(content_image)

        style_features = self.Es(style_images.view(-1, 1, 64, 64))
        style_features_1 = self.A1(style_features)

        style_features = self.downsample_1(style_features)
        style_features_2 = self.A2(style_features)

        style_features = self.downsample_2(style_features)
        style_features_3 = self.A3(style_features)

        style_features = self.LA(style_features, style_features_1, style_features_2, style_features_3, B, K)

        feature = torch.cat([content_feature, style_features], dim=1)

        return self.D(feature)