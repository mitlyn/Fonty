import torch.nn as nn

from model.blocks import *

# *----------------------------------------------------------------------------*


class Encoder(nn.Module):
    def __init__(self, input_nc, ngf=64):
        super(Encoder, self).__init__()

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        ]

        for i in range(2):  # add downsampling layers
            mul = 2 ** i
            model += [
                nn.Conv2d(ngf * mul, ngf * mul * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ngf * mul * 2),
                nn.ReLU(True)
            ]

        self.model = nn.Sequential(*model)

    def forward(self, inp):
        return self.model(inp)