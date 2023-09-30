import torch.nn as nn
from torch import cat

from model.ftransgan import Encoder, Decoder, LocalAttention, LayerAttention, FeatureScaler

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
        self.Al = LayerAttention(filters=filters)

        self.S1 = FeatureScaler(filters)
        self.S2 = FeatureScaler(filters)


    def forward(self, content_image, style_images):
        B, K, N, _ = style_images.shape

        content_features = self.Ec(content_image)

        style_features = self.Es(style_images.view(-1, 1, N, N))
        style_features_1 = self.A1(style_features)

        style_features = self.S1(style_features)
        style_features_2 = self.A2(style_features)

        style_features = self.S2(style_features)
        style_features_3 = self.A3(style_features)

        style_features = self.Al(style_features, style_features_1, style_features_2, style_features_3, B, K)

        features = cat([content_features, style_features], dim=1)

        return self.D(features)