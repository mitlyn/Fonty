import torch.nn as nn

# *----------------------------------------------------------------------------* Decoder Block

class DecoderBlock(nn.Module):
    def __init__(self, dim: int, padding: str, norm_layer, dropout: bool, bias: bool):
        super(DecoderBlock, self).__init__()

        block = []

        p = 0
        if padding == 'reflect':
            block += [nn.ReflectionPad2d(1)]
        elif padding == 'replicate':
            block += [nn.ReplicationPad2d(1)]
        else:
            p = 1

        block += [
            nn.Conv2d(
                dim, dim,
                kernel_size=3,
                padding=p,
                bias=bias
            ),
            norm_layer(dim),
            nn.ReLU(True)
        ]

        if dropout:
            block += [nn.Dropout(0.5)]

        p = 0
        if padding == 'reflect':
            block += [nn.ReflectionPad2d(1)]
        elif padding == 'replicate':
            block += [nn.ReplicationPad2d(1)]
        else:
            p = 1

        block += [
            nn.Conv2d(
                dim, dim,
                kernel_size=3,
                padding=p,
                bias=bias
            ),
            norm_layer(dim)
        ]

        self.block = nn.Sequential(*block)

    def forward(self, X):
        return self.block(X)

# *----------------------------------------------------------------------------* Downsampling Block For Layer Attention

class FeatureScaler(nn.Module):
    def __init__(self, dim: int):
        super(FeatureScaler, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                dim * 4, dim * 4,
                kernel_size=3,
                bias=False,
                padding=1,
                stride=2,
            ),
            nn.BatchNorm2d(dim * 4),
            nn.ReLU(True)
        )

    def forward(self, X):
        return self.block(X)



