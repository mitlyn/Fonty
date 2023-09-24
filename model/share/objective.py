import torch.nn as nn
from torch import Tensor, tensor

# *----------------------------------------------------------------------------*

class Objective(nn.Module):
    """Base GAN loss objective function."""

    def __init__(self, real_label: float = 1.0, fake_label: float = 0.0):
        super(Objective, self).__init__()

        self.register_buffer('real_label', tensor(real_label))
        self.register_buffer('fake_label', tensor(fake_label))

    def mimic(self, result: Tensor, real: bool) -> Tensor:
        """Create label tensor with the same size as the input."""
        target = self.real_label if real else self.fake_label
        return target.expand_as(result)

# *----------------------------------------------------------------------------* Vanilla Objective

class Vanilla(Objective):
    """Vanilla GAN loss objective function."""

    def __init__(self, real_label: float = 1.0, fake_label: float = 0.0):
        super(Vanilla, self).__init__(real_label, fake_label)
        self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, result: Tensor, real: bool) -> Tensor:
        target = self.mimic(result, real)
        return self.loss(result, target)

# *----------------------------------------------------------------------------* LSGAN Objective

class LSGAN(Objective):
    """Least Squares GAN loss objective function."""

    def __init__(self, real_label: float = 1.0, fake_label: float = 0.0):
        super(LSGAN, self).__init__(real_label, fake_label)
        self.loss = nn.MSELoss()

    def __call__(self, result: Tensor, real: bool) -> Tensor:
        target = self.mimic(result, real)
        return self.loss(result, target)

# *----------------------------------------------------------------------------* WGAN-GP Objective

class WGANGP(Objective):
    """Wasserstein GAN with gradient penalty loss objective function."""

    def __init__(self, real_label: float = 1.0, fake_label: float = 0.0):
        super(WGANGP, self).__init__(real_label, fake_label)

    def __call__(self, result: Tensor, real: bool) -> Tensor:
        return -result.mean() if real else result.mean()

# *----------------------------------------------------------------------------* Hinge Objective

class Hinge(Objective):
    """Hinge GAN loss objective function."""

    def __init__(self, real_label: float = 1.0, fake_label: float = 0.0):
        super(Hinge, self).__init__(real_label, fake_label)
        self.act = nn.ReLU()

    def __call__(self, result: Tensor, real: bool) -> Tensor:
        return self.act(1.0 - result).mean() if real else self.act(1.0 + result).mean()