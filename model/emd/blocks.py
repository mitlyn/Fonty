import torch.nn as nn

# *----------------------------------------------------------------------------* Encoder Block

class EncoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride, padding):
        super(EncoderBlock, self).__init__()

        self.block = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.Conv2d(
                in_dim,
                out_dim,
                kernel_size=kernel,
                padding=padding,
                stride=stride,
                bias=False,
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )


    def forward(self, X):
        return self.block(X)

# *----------------------------------------------------------------------------* Decoder Block

class DecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride, padding, add, inner=True):
        super(DecoderBlock, self).__init__()

        if inner:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(
                    in_dim,
                    out_dim,
                    output_padding=add,
                    kernel_size=kernel,
                    padding=padding,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.block = nn.ConvTranspose2d(
                in_dim,
                out_dim,
                output_padding=add,
                kernel_size=kernel,
                padding=padding,
                stride=stride,
                bias=False,
            )


    def forward(self, X):
        return self.block(X)
