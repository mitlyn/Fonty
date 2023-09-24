import torch.nn as nn
from model.emd import EncoderBlock

# *----------------------------------------------------------------------------* Variables

index = range(7)
kernels = [5, 3, 3, 3, 3, 3, 3]
strides = [1, 2, 2, 2, 2, 2, 2]
dims = [64, 128, 256, 512, 512, 512, 512]

# *----------------------------------------------------------------------------* Content Encoder

class ContentEncoder(nn.Module):
    def __init__(self, channels=1):
        super(ContentEncoder, self).__init__()

        self.layers = nn.ModuleList(
            EncoderBlock(
                dims[i - 1] if i > 0 else channels,
                dims[i],
                kernels[i],
                strides[i],
                kernels[i] // 2
            ) for i in index
        )


    def forward(self, X):
        out = [X]

        for i in index:
            out.append(self.layers[i](out[-1]))

        return out

# *----------------------------------------------------------------------------* Style Encoder

class StyleEncoder(nn.Module):
    def __init__(self, channels):
        super(StyleEncoder, self).__init__()

        self.layers = nn.ModuleList(
            EncoderBlock(
                dims[i - 1] if i > 0 else channels,
                dims[i],
                kernels[i],
                strides[i],
                kernels[i] // 2
            ) for i in index
        )

    def forward(self, X):
        for i in index:
            X = self.layers[i](X)

        return X
