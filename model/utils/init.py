from torch import nn

# *----------------------------------------------------------------------------*


def setInit(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)       - network to be initialized
        init_type (str)     - initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)   - scaling factor for normal, xavier and orthogonal
    """
    def init(n):
        classname = n.__class__.__name__
        if hasattr(n, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(n.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(n.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(n.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(n.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'No such Init method: {init_type}')

            if hasattr(n, 'bias') and n.bias is not None:
                nn.init.constant_(n.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(n.weight.data, 1.0, init_gain)
            nn.init.constant_(n.bias.data, 0.0)

    net.apply(init)