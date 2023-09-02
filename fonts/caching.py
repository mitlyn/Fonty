from typing import Tuple, List
from pickle import dump, load

from fonts.types.mongo import MongoGlyphs


def toCache(base, train, test, dir: str = ".") -> None:
    with open(f"{dir}/base.pkl", "wb") as O:
        dump(base, O)

    with open(f"{dir}/train.pkl", "wb") as O:
        dump(train, O)

    with open(f"{dir}/test.pkl", "wb") as O:
        dump(test, O)


def fromCache(dir: str = ".") -> Tuple[MongoGlyphs, List[MongoGlyphs], List[MongoGlyphs]]:
    with open(f"{dir}/base.pkl", "rb") as I:
        base = load(I)

    with open(f"{dir}/train.pkl", "rb") as I:
        train = load(I)

    with open(f"{dir}/test.pkl", "rb") as I:
        test = load(I)

    return base, train, test
