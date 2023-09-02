from torch import tensor
from pickle import loads
from typing import Any, Dict, Iterable

from fonts.client import Client
from fonts.types.mongo import MongoGlyphs


class FontLoader(Client):

    def _get_glyph(self, name: str, size: int = 64) -> Dict[str, Any]:
        return self.glyphs.find_one({"name": name, "size": size}, {"_id": 0})


    def _get_glyphs(self, size: int = 64) -> Dict[str, Any]:
        return self.glyphs.find({"size": size}, {"_id": 0})


    def _decode(self, item: dict) -> MongoGlyphs:
        return MongoGlyphs(
            name=item["name"],
            size=item["size"],
            panose=item["panose"],
            lat=tensor(loads(item["lat"])).view(-1, item["size"], item["size"]),
            cyr=tensor(loads(item["cyr"])).view(-1, item["size"], item["size"]),
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
            if (len(data[name].lat) == 0):
                continue

            if (len(data[name].cyr) > 0):
                train.append(data[name])
            else:
                test.append(data[name])

        return base, train, test
