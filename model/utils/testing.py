from torch import Tensor, no_grad
from model.types import Bundle

# *----------------------------------------------------------------------------*


def apply(net, data: Bundle) -> Tensor:
    """Apply font style transfer to given data."""
    content = data.content.cuda().view(1, -1, 64, 64)
    style = data.style.cuda().view(1, -1, 64, 64)
    net.cuda()

    net.eval()

    with no_grad():
        result = net.G((content, style))

    net.train()

    return result.cpu().view(-1, 64, 64)