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

class DecoderBlock(nn.Module):
    def __init__(self, dim: int, padding: str, norm_layer, dropout: bool, bias: bool):
        super(DecoderBlock, self).__init__()

        model = []

        p = 0
        if padding == 'reflect':
            model += [nn.ReflectionPad2d(1)]
        elif padding == 'replicate':
            model += [nn.ReplicationPad2d(1)]
        else:
            p = 1

        model += [
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
            model += [nn.Dropout(0.5)]

        p = 0
        if padding == 'reflect':
            model += [nn.ReflectionPad2d(1)]
        elif padding == 'replicate':
            model += [nn.ReplicationPad2d(1)]
        else:
            p = 1

        model += [
            nn.Conv2d(
                dim, dim,
                kernel_size=3,
                padding=p,
                bias=bias
            ),
            norm_layer(dim)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, X):
        return self.model(X)