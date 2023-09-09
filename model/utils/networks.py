from typing import List, Union
from torch import Tensor, no_grad

from fonts.types.mongo import MongoGlyphs
from model.types import Bundle, TrainBundle

# *----------------------------------------------------------------------------* Apply Style Transfer

def apply(net, data: Bundle) -> Tensor:
    """Apply font style transfer to given data."""
    content = data.content.view(1, -1, 64, 64).cuda()
    style = data.style.view(1, -1, 64, 64).cuda()
    panose = data.panose.view(-1).cuda()
    net.cuda()

    net.eval()

    with no_grad():
        result = net.G((content, style, panose))

    net.train()

    return result.cpu().view(-1, 64, 64)

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