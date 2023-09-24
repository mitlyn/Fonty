import torch.nn as nn

from model.emd import ContentEncoder, StyleEncoder, Decoder, Mixer

class Generator(nn.Module):
    def __init__(self, style_channels, content_channels=1):
        super(Generator, self).__init__()

        self.content_encoder = ContentEncoder(content_channels)
        self.style_encoder = StyleEncoder(style_channels)
        self.decoder = Decoder()
        self.mixer = Mixer()

    def forward(self, content, style):
        style_features = self.style_encoder(style)
        content_features = self.content_encoder(content)
        mixed = self.mixer(content_features[-1], style_features)

        return self.decoder(mixed, content_features)
