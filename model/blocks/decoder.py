import torch.nn as nn

from model.blocks import *

# *----------------------------------------------------------------------------*


class Decoder(nn.Module):
    def __init__(self, filters: int = 64, blocks: int = 6, dropout: bool = False):
        super(Decoder, self).__init__()

        model = []

        for i in range(blocks):       # add ResNet blocks
            model += [ResNetBlock(
                filters * 8,
                padding='reflect',
                norm_layer=nn.BatchNorm2d,
                dropout=dropout,
                bias=False
            )]

        for i in range(2):  # add upsampling layers
            mul = 2 ** (3 - i)

            model += [
                nn.ConvTranspose2d(
                    filters * mul,
                    int(filters * mul / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(int(filters * mul / 2)),
                nn.ReLU(True)
            ]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(filters * 2, 1, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, X):
        return self.model(X)