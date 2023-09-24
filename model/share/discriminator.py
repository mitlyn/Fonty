import torch.nn as nn
from torch.nn.utils import spectral_norm as spectral

# *----------------------------------------------------------------------------* Regular Discriminator

class Discriminator(nn.Module):
    def __init__(self, dim: int, filters: int = 64, layers: int = 3, norm = nn.InstanceNorm2d):
        super(Discriminator, self).__init__()

        mul = 1
        mul_prev = 1

        sequence = [
            nn.Conv2d(dim, filters, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(0.2, True),
        ]

        for n in range(1, layers):
            mul_prev = mul
            mul = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    filters * mul_prev,
                    filters * mul,
                    kernel_size=4,
                    padding=1,
                    stride=2,
                    bias=False,
                ),
                norm(filters * mul),
                nn.LeakyReLU(0.2, True),
            ]

        mul_prev = mul
        mul = min(2**layers, 8)

        sequence += [
            nn.Conv2d(
                filters * mul_prev,
                filters * mul,
                kernel_size=4,
                padding=1,
                stride=1,
                bias=False,
            ),
            norm(filters * mul),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [nn.Conv2d(filters * mul, 1, kernel_size=4, padding=1, stride=1)]

        self.model = nn.Sequential(*sequence)

    def forward(self, X):
        return self.model(X)


# *----------------------------------------------------------------------------* Spectral Discriminator


class SpectralDiscriminator(nn.Module):
    def __init__(self, dim: int, filters: int = 64, layers: int = 3, norm_layer = nn.InstanceNorm2d):
        super(SpectralDiscriminator, self).__init__()

        mul = 1
        mul_prev = 1

        sequence = [
            spectral(
                nn.Conv2d(dim, filters, kernel_size=4, padding=1, stride=2)
            ),
            nn.LeakyReLU(0.2, True),
        ]

        for n in range(1, layers):
            mul_prev = mul
            mul = min(2**n, 8)
            sequence += [
                spectral(
                    nn.Conv2d(
                        filters * mul_prev,
                        filters * mul,
                        kernel_size=4,
                        padding=1,
                        stride=2,
                        bias=False,
                    )
                ),
                norm_layer(filters * mul),
                nn.LeakyReLU(0.2, True),
            ]

        mul_prev = mul
        mul = min(2**layers, 8)

        sequence += [
            spectral(
                nn.Conv2d(
                    filters * mul_prev,
                    filters * mul,
                    kernel_size=4,
                    padding=1,
                    stride=1,
                    bias=False,
                )
            ),
            norm_layer(filters * mul),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [spectral(nn.Conv2d(filters * mul, 1, kernel_size=4, padding=1, stride=1))]

        self.model = nn.Sequential(*sequence)

    def forward(self, X):
        return self.model(X)
