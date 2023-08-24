import torch.nn as nn
import torch.nn.functional as F

# TODO: Work In Progress

# *----------------------------------------------------------------------------*

# nn.Embedding

class Embedder(nn.Module):
    def __init__(self, dim, filters=64):
        super(Embedder, self).__init__()

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(dim, filters, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(True),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, inp):
        return self.model(inp)
