from torch import Tensor
from torch.nn import Module
from model.types import Bundle
from typing import Any, List, Dict, Union

from torch import stack, uint8
from torchmetrics import MetricCollection, image as mi

from model.utils.nets import apply

# *----------------------------------------------------------------------------* Metrics

def Metrics() -> MetricCollection:
    return MetricCollection({
        "PSNR": mi.PeakSignalNoiseRatio(),
        "RASE": mi.RelativeAverageSpectralError(),
        "SSIM": mi.StructuralSimilarityIndexMeasure(),
        "RMSE-SW": mi.RootMeanSquaredErrorUsingSlidingWindow(),
        "ERGAS": mi.ErrorRelativeGlobalDimensionlessSynthesis(),
    })

# *----------------------------------------------------------------------------* Evaluation

def eval(model: Module, data: List[Bundle]) -> Dict[str, Any]:
    fake: Union[List[Tensor], Tensor] = []
    real: Union[List[Tensor], Tensor] = []
    metrics = Metrics()

    # Collecting & Preparing Data

    for item in data:
        fake.append(apply(model, item))
        real.append(item.target)

    fake = stack(fake).view(-1, 1, 64, 64)
    real = stack(real).view(-1, 1, 64, 64)

    # Regular Metrics

    metrics.update(fake, real)
    result = metrics.compute()

    result = {k: v.item() for k, v in result.items()}

    # Casting to 3-Channel Images

    fake = (fake * 255).to(uint8).repeat(1, 3, 1, 1)
    real = (real * 255).to(uint8).repeat(1, 3, 1, 1)

    # Frechet Inception Distance

    FID = mi.FrechetInceptionDistance(64)

    FID.update(real, True)
    FID.update(fake, False)

    result["FID"] = FID.compute().item()

    # Memorization Informed Frechet Inception Distance

    mFID = mi.MemorizationInformedFrechetInceptionDistance(64)
    mFID.update(fake, False)
    mFID.update(real, True)

    result["mFID"] = mFID.compute().item()

    # Learned Perceptual Image Patch Similarity

    # TODO: [0, 255] -> [-1, 1]

    # LPIPS = mi.LearnedPerceptualImagePatchSimilarity()
    # LPIPS.update(real, fake)

    # result["LPIPS"] = LPIPS.compute().item()

    return result
