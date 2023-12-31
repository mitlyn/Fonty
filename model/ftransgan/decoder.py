import torch.nn as nn

from model.share import Residual
from model.ftransgan import DecoderBlock

# *----------------------------------------------------------------------------*

class Decoder(nn.Module):
    def __init__(self, filters: int = 64, blocks: int = 6, dropout: bool = False):
        super(Decoder, self).__init__()

        # Residual Convolution Blocks
        model = [
            Residual(DecoderBlock(
                filters * 8, # { 12 | 8 }
                padding='reflect',
                norm_layer=nn.BatchNorm2d,
                dropout=dropout,
                bias=False
            )) for _ in range(blocks)
        ]

        # Upsampling Layers
        for mul in [8, 4]: # { [12, 6] | [8, 4] }
            model += [
                nn.ConvTranspose2d(
                    filters * mul,
                    filters * mul // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(filters * mul // 2),
                nn.ReLU(True)
            ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(filters * 2, 1, kernel_size=7, padding=0), # { 3 | 2 }
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, X):
        return self.model(X)