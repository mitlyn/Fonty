import pylab as pl

# *----------------------------------------------------------------------------*


def show(I):
    O = I.reshape(64, 64)
    pl.imshow(O, cmap="gray")
    pl.axis("off")


# *----------------------------------------------------------------------------*


def showMany(*Is):
    N = len(Is)
    M = 1

    if N > 4:
        M = N // 4 + 1
        N = 4

    for i, I in enumerate(Is):
        pl.subplot(M, N, i + 1)
        show(I)

    pl.axis("off")
    pl.show()
