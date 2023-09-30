from torch import Tensor, no_grad
from torch.nn import Module, init
from typing import Any, List, Literal

from fonts.types.mongo import MongoGlyphs
from model.types import Bundle, TrainBundle

# *----------------------------------------------------------------------------* Apply Style Transfer

def __call(model: Module, *args: Any) -> Tensor:
    model.cuda()
    model.eval()

    with no_grad():
        result = model.G(*args)

    model.train()

    return result.cpu().view(-1, 64, 64)


def apply(model: Module, data: Bundle) -> Tensor:
    """Apply font style transfer to given data."""
    content = data.content.view(1, -1, 64, 64).cuda()
    style = data.style.view(1, -1, 64, 64).cuda()
    return __call(model, content, style)


def applyPanose(model: Module, data: Bundle) -> Tensor:
    """Apply panose-conditioned font style transfer."""
    content = data.content.view(1, -1, 64, 64).cuda()
    style = data.style.view(1, -1, 64, 64).cuda()
    panose = data.panose.view(-1).cuda()
    return __call(model, content, style, panose)

# *----------------------------------------------------------------------------* Create Bundles

def bundled(data: List[MongoGlyphs], content: Tensor, train: bool = False) -> List[Bundle]:
    result: List[Bundle] = []

    if train:
        for item in data:
            result.extend(
                TrainBundle(
                    target=t,
                    content=c,
                    style=item.en,
                    panose=item.panose,
                ) for c, t in zip(content, item.ua)
            )
    else:
        for item in data:
            result.extend(
                Bundle(
                    content=c,
                    style=item.en,
                    panose=item.panose,
                ) for c in content
            )

    return result

# *----------------------------------------------------------------------------* Initialize Weights

Inits = Literal['normal', 'xavier', 'kaiming', 'orthogonal']

def setInit(model: Module, method: Inits = 'normal', gain: float = 0.02) -> None:
    def func(node):
        classname = node.__class__.__name__
        if hasattr(node, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if method == 'normal':
                init.normal_(node.weight.data, 0.0, gain)
            elif method == 'xavier':
                init.xavier_normal_(node.weight.data, gain)
            elif method == 'orthogonal':
                init.orthogonal_(node.weight.data, gain)
            elif method == 'kaiming':
                init.kaiming_normal_(node.weight.data)
            else:
                raise NotImplementedError(f'No Init: {method}')

            if hasattr(node, 'bias') and node.bias is not None:
                init.constant_(node.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(node.weight.data, 1.0, gain)
            init.constant_(node.bias.data, 0.0)

    model.apply(func)

# *----------------------------------------------------------------------------* Toggle Gradients

def setGrads(state: bool, *nets: Any):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = state
