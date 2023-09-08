from torch import tensor
from pickle import dump, load, loads
from typing import Any, List, Dict, Tuple, Iterable

from fonts.clients import LocalClient
from fonts.types.mongo import MongoGlyphs

# TODO: batching, panose filtering, etc.


class FontLoader():
    def __init__(self, localDB: LocalClient):
        self.cache = localDB.col

    def _get_glyph(self, name: str, size: int = 64) -> Dict[str, Any]:
        return self.cache.find_one({"name": name, "size": size}, {"_id": 0})


    def _get_glyphs(self, size: int = 64) -> Dict[str, Any]:
        return self.cache.find({"size": size}, {"_id": 0})


    def _decode(self, item: dict) -> MongoGlyphs:
        return MongoGlyphs(
            name=item["name"],
            size=item["size"],
            panose=item["panose"],
            en=tensor(loads(item["en"])).view(-1, item["size"], item["size"]),
            ua=tensor(loads(item["ua"])).view(-1, item["size"], item["size"]),
        )


    def loadMany(self, size: int = 64, *names: Iterable[str]) -> Dict[str, MongoGlyphs]:
        return {name: self._decode(self._get_glyph(name, size)) for name in names}


    def loadAll(self, size: int = 64) -> Dict[str, MongoGlyphs]:
        return {font["name"]: self._decode(font) for font in self._get_glyphs(size)}


    # *------------------------------------------------------------------------*  Loading Data


    def load(self, size: int = 64, base_font: str = "Open Sans Light"):
        """Loads all rendered data and splits it into train and test sets."""

        data = self.loadAll(size)
        base = data.pop(base_font)

        train = [] # fonts with Latin and Cyrillic glyphs
        test = [] # fonts with only Latin glyphs

        for name in data:
            if (len(data[name].en) == 0):
                continue

            if (len(data[name].ua) > 0):
                train.append(data[name])
            else:
                test.append(data[name])

        return base, train, test

# *----------------------------------------------------------------------------* Data Caching

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
