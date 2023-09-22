from torch import nn
from typing import Literal

Inits = Literal['normal', 'xavier', 'kaiming', 'orthogonal']

# *----------------------------------------------------------------------------*

def setInit(model: nn.Module, method: Inits = 'normal', gain: float = 0.02):
    def init(node):
        classname = node.__class__.__name__
        if hasattr(node, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if method == 'normal':
                nn.init.normal_(node.weight.data, 0.0, gain)
            elif method == 'xavier':
                nn.init.xavier_normal_(node.weight.data, gain)
            elif method == 'orthogonal':
                nn.init.orthogonal_(node.weight.data, gain)
            elif method == 'kaiming':
                nn.init.kaiming_normal_(node.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError(f'No such Init method: {method}')

            if hasattr(node, 'bias') and node.bias is not None:
                nn.init.constant_(node.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(node.weight.data, 1.0, gain)
            nn.init.constant_(node.bias.data, 0.0)

    model.apply(init)