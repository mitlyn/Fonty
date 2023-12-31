import torch.nn as nn

# *----------------------------------------------------------------------------*

class Encoder(nn.Module):
    def __init__(self, dim: int, filters: int = 64):
        super(Encoder, self).__init__()

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(dim, filters, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(True)
        ]

        # Downsampling Layers
        for mul in [1, 2]:
            model += [
                nn.Conv2d(filters * mul, filters * mul * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(filters * mul * 2),
                nn.ReLU(True)
            ]

        self.model = nn.Sequential(*model)

    def forward(self, X):
        return self.model(X)