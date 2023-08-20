import torch.nn as nn

from model.blocks import *

# *----------------------------------------------------------------------------*


class Decoder(nn.Module):
    def __init__(self, ngf=64, n_blocks=6, dropout=False):
        super(Decoder, self).__init__()

        model = []

        for i in range(n_blocks):       # add ResNet blocks
            model += [ResNetBlock(
                ngf * 8,
                padding_type='reflect',
                norm_layer=nn.BatchNorm2d,
                use_dropout=dropout,
                use_bias=False
            )]

        for i in range(2):  # add upsampling layers
            mul = 2 ** (3 - i)

            model += [
                nn.ConvTranspose2d(
                    ngf * mul,
                    int(ngf * mul / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(int(ngf * mul / 2)),
                nn.ReLU(True)
            ]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf * 2, 1, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, inp):
        return self.model(inp)