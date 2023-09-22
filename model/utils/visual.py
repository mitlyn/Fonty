import pylab as pl
from os import mkdir
from torch import Tensor
from cv2 import imwrite, resize
from typing import Iterable, List

from numpy import zeros, hstack, vstack

arrow = zeros((8,20))

arrow[0, 14] = 1
arrow[1, 15:17] = 1
arrow[2, 16:19] = 1
arrow[3, :] = 1
arrow[4, :] = 1
arrow[5, 16:19] = 1
arrow[6, 15:17] = 1
arrow[7, 14] = 1

arrow = vstack([zeros((28,20)), arrow, zeros((28,20))])
arrow = 1 - arrow

plus = zeros((20,20))

plus[9:12, :] = 1
plus[:, 9:12] = 1

plus = vstack([zeros((22,20)), plus, zeros((22,20))])
plus = 1 - plus

"""
0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9
                            x
                              x x
                                x x x
x x x x x x x x x x x x x x x x x x x x
x x x x x x x x x x x x x x x x x x x x
                                x x x
                              x x
                            x

"""
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

    for m in range(M):
        for n in range(N):
            pl.subplot(1, N, (n % 4) + 1)
            iShow(Xs[m * N + n])

        pl.axis("off")
        pl.show()

# *----------------------------------------------------------------------------*

def iSave(X: Tensor, path: str) -> None:
    image = X.reshape(64, 64).numpy()
    imwrite(f"{path}.png", image * 255.0)


def i2Save(X: Tensor, Y:Tensor, path: str) -> None:
    content = X.reshape(64, 64).numpy()
    result = Y.reshape(64, 64).numpy()

    image = hstack([content, arrow, result])
    imwrite(f"{path}.png", image * 255.0)


def i3Save(X: Tensor, S: Tensor, Y:Tensor, path: str) -> None:
    content = X.reshape(64, 64).numpy()
    result = Y.reshape(64, 64).numpy()
    style = S.reshape(64, 64).numpy()

    image = hstack([content, plus, style, arrow, result])
    imwrite(f"{path}.png", image * 255.0)


def is3Save(Bs: List[dict], path: str) -> None:
    images = []

    for B in Bs:
        content = B["X"].reshape(64, 64).numpy()
        result = B["Y"].reshape(64, 64).numpy()
        style = B["S"].reshape(64, 64).numpy()

        images.append(hstack([content, plus, style, arrow, result]))

    imwrite(f"{path}.png", vstack(images) * 255.0)


def iSaveMany(*Xs: Iterable[Tensor], folder: str = "out") -> None:
    mkdir(folder)
    for i in range(len(Xs)):
        iSave(Xs[i], f"{folder}/{i}")