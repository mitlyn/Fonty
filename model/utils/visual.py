import pylab as pl
from os import mkdir
from cv2 import imwrite
from torch import Tensor
from typing import Iterable

# *----------------------------------------------------------------------------*

def iShow(X: Tensor) -> None:
    image = X.reshape(64, 64)
    pl.imshow(image, cmap="gray")
    pl.axis("off")


def iShowMany(*Xs: Iterable[Tensor]) -> None:
    N = len(Xs)
    M = 1

    if N > 4:
        M = N // 4 + 1
        N = 4

    for i, X in enumerate(Xs):
        pl.subplot(M, N, i + 1)
        iShow(X)

    pl.axis("off")
    pl.show()

# *----------------------------------------------------------------------------*

def iSave(X: Tensor, path: str) -> None:
    image = X.reshape(64, 64).numpy()
    imwrite(f"{path}.png", image * 255.0)


def iSaveMany(*Xs: Iterable[Tensor], folder: str = "out") -> None:
    mkdir(folder)
    for i in range(len(Xs)):
        iSave(Xs[i], f"{folder}/{i}")