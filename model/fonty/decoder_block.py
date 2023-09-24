import torch.nn as nn

# *----------------------------------------------------------------------------*

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