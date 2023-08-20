import torch.nn as nn

# *----------------------------------------------------------------------------*

"""Construct a convolutional block.

    Parameters:
        dim (int)           -- the number of channels in the conv layer.
        padding_type (str)  -- the name of padding layer: reflect | replicate | zero
        norm_layer          -- normalization layer
        use_dropout (bool)  -- if use dropout layers.
        use_bias (bool)     -- if the conv layer uses bias or not

    Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
"""

class ResNetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResNetBlock, self).__init__()

        conv_block = []

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        else:
            p = 1

        conv_block += [
            nn.Conv2d(
                dim, dim,
                kernel_size=3,
                padding=p,
                bias=use_bias
            ),
            norm_layer(dim),
            nn.ReLU(True)
        ]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        else:
            p = 1

        conv_block += [
            nn.Conv2d(
                dim, dim,
                kernel_size=3,
                padding=p,
                bias=use_bias
            ),
            norm_layer(dim)
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)