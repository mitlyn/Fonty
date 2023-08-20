import functools
import torch.nn as nn
from torch.nn.utils import spectral_norm

from model.blocks import *

# *----------------------------------------------------------------------------*


class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(Discriminator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        pad = 1
        kernel = 4
        nf_mul = 1
        nf_mul_prev = 1

        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kernel, stride=2, padding=pad),
            nn.LeakyReLU(0.2, True),
        ]

        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mul_prev = nf_mul
            nf_mul = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mul_prev,
                    ndf * nf_mul,
                    kernel_size=kernel,
                    stride=2,
                    padding=pad,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mul),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mul_prev = nf_mul
        nf_mul = min(2**n_layers, 8)

        sequence += [
            nn.Conv2d(
                ndf * nf_mul_prev,
                ndf * nf_mul,
                kernel_size=kernel,
                stride=1,
                padding=pad,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mul),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mul, 1, kernel_size=kernel, stride=1, padding=pad)
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


# *----------------------------------------------------------------------------*


class SpectralDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(SpectralDiscriminator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        pad = 1
        kernel = 4
        nf_mul = 1
        nf_mul_prev = 1

        sequence = [
            spectral_norm(
                nn.Conv2d(input_nc, ndf, kernel_size=kernel, stride=2, padding=pad)
            ),
            nn.LeakyReLU(0.2, True),
        ]

        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mul_prev = nf_mul
            nf_mul = min(2**n, 8)
            sequence += [
                spectral_norm(
                    nn.Conv2d(
                        ndf * nf_mul_prev,
                        ndf * nf_mul,
                        kernel_size=kernel,
                        stride=2,
                        padding=pad,
                        bias=use_bias,
                    )
                ),
                norm_layer(ndf * nf_mul),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mul_prev = nf_mul
        nf_mul = min(2**n_layers, 8)

        sequence += [
            spectral_norm(
                nn.Conv2d(
                    ndf * nf_mul_prev,
                    ndf * nf_mul,
                    kernel_size=kernel,
                    stride=1,
                    padding=pad,
                    bias=use_bias,
                )
            ),
            norm_layer(ndf * nf_mul),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            spectral_norm(
                nn.Conv2d(ndf * nf_mul, 1, kernel_size=kernel, stride=1, padding=pad)
            )
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
