import torch.nn as nn
from torch import cat
from model.emd import DecoderBlock

# *----------------------------------------------------------------------------* Variables

index = range(7)
kernels = [3, 3, 3, 3, 3, 3, 5]
strides = [2, 2, 2, 2, 2, 2, 1]
dims = [512, 512, 512, 256, 128, 64, 1]

# *----------------------------------------------------------------------------* Decoder

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList(
            DecoderBlock(
                (dims[i - 1] if i > 0 else 512) * 2,
                dims[i],
                kernels[i],
                strides[i],
                kernels[i] // 2,
                strides[i] // 2,
                inner=(i < 6),
            ) for i in index
        )

        self.th = nn.Tanh()

    def forward(self, X, layers):
        for i in index:
            X = cat([X, layers[-i - 1]], 1)
            X = self.layers[i](X)

        return self.th(X)