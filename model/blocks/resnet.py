import torch.nn as nn

# *----------------------------------------------------------------------------*

"""Convolutional block with skip connection.

    Parameters:
        dim (int)       - the number of channels in the conv layer
        padding (str)   - padding type: reflect | replicate | zero
        norm_layer      - normalization layer
        dropout (bool)  - use dropout layers?
        bias (bool)     - uses bias?
"""

class ResNetBlock(nn.Module):
    def __init__(self, dim: int, padding: str, norm_layer, dropout: bool, bias: bool):
        super(ResNetBlock, self).__init__()

        conv_block = []

        p = 0
        if padding == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        else:
            p = 1

        conv_block += [
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
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        else:
            p = 1

        conv_block += [
            nn.Conv2d(
                dim, dim,
                kernel_size=3,
                padding=p,
                bias=bias
            ),
            norm_layer(dim)
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, X):
        return X + self.conv_block(X)