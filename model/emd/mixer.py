import torch, torch.nn as nn
from torch import squeeze, unsqueeze

# *----------------------------------------------------------------------------*

class Mixer(nn.Module):
    def __init__(self):
        super(Mixer, self).__init__()
        self.mixer = nn.Bilinear(512, 512, 512)

    def forward(self, content, style):
        content = squeeze(squeeze(content, -1), -1)
        style = squeeze(squeeze(style, -1), -1)
        mixed = self.mixer(content, style)

        return unsqueeze(unsqueeze(mixed, -1), -1)