import torch.nn as nn
from torch import cat

from model.fonty import LinearEmbedder
from model.ftransgan import Encoder, Decoder, LocalAttention, LayerAttention, FeatureScaler

# *----------------------------------------------------------------------------*

class Generator(nn.Module):
    def __init__(self, filters: int = 64, blocks: int = 6, dropout: bool = False):
        super(Generator, self).__init__()

        self.Es = Encoder(dim=1, filters=filters)
        self.Ec = Encoder(dim=1, filters=filters)
        self.Ep = LinearEmbedder(filters=filters)

        self.D = Decoder(filters=filters, blocks=blocks, dropout=dropout)

        self.A1 = LocalAttention(filters=filters)
        self.A2 = LocalAttention(filters=filters)
        self.A3 = LocalAttention(filters=filters)
        self.Al = LayerAttention(filters=filters)

        self.S1 = FeatureScaler(filters)
        self.S2 = FeatureScaler(filters)

        self.bilinear = nn.Bilinear(filters * 4, 1, filters * 4)


    def forward(self, content_image, style_images, panose):
        B, K, N, _ = style_images.shape

        content_features = self.Ec(content_image)

        panose_features = self.Ep(panose)

        style_features = self.Es(style_images.view(-1, 1, N, N))
        style_features_1 = self.A1(style_features)

        style_features = self.S1(style_features)
        style_features_2 = self.A2(style_features)

        style_features = self.S2(style_features)
        style_features_3 = self.A3(style_features)

        style_features = self.Al(style_features, style_features_1, style_features_2, style_features_3, B, K)

        # *--------------------------------------------------------------------* Option 1: Concatenate
        # # Batch × 3 * (Filters * 4) × (Size / 4) × (Size / 4)
        # #   1   ×      3 * 256      ×     16     ×     16

        # # Simply concatenate content, style, and Panose-1 features.
        # features = cat([content_features, style_features, panose_features], dim=1)

        # *--------------------------------------------------------------------* Option 2: Bilinear Feature Fusion
        # Batch × 2 * (Filters * 4) × (Size / 4) × (Size / 4)
        #   1   ×      2 * 256      ×     16     ×     16

        style_features = style_features.view(N * 4, -1)
        panose_features = panose_features.view(N * 4, -1)
        # Perform bilinear feature fusion of style and Panose-1 features.
        font_features = self.bilinear(style_features, panose_features)
        font_features = font_features.view(1, N * 4, N // 4, N // 4)

        features = cat([content_features, font_features], dim=1)

        return self.D(features)