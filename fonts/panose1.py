from typing import Dict, Tuple
from torch import Tensor, tensor, int64

PANOSE_FEATURES = (
    'family', 'serif_style', 'weight', 'proportion', 'contrast',
    'stroke_variation', 'arm_style', 'letterform', 'midline', 'xheight'
)


def digits_to_features(digits: Tuple[int]) -> Dict[str, int]:
    return dict(zip(PANOSE_FEATURES, digits))


def features_to_digits(panose: Dict[str, int]) -> Tensor:
    return tensor([panose[x] for x in PANOSE_FEATURES], dtype=int64)
