import pickle as pkl
from torch import tensor
from typing import List, Union
from pymongo.cursor import Cursor

from fonts.clients import LocalClient
from fonts.types.mongo import MongoGlyphs

# TODO: batching, panose filtering, etc.


class FontLoader():
    def __init__(self, localDB: LocalClient):
        self.cache = localDB.col

    # *------------------------------------------------------------------------* Retrieval

    def getFont(self, name: str, size: int = 64) -> dict:
        return self.cache.find_one({"name": name, "size": size}, {"_id": 0})


    def getFonts(self, size: int = 64) -> Cursor:
        return self.cache.find({"size": size}, {"_id": 0})

    # *------------------------------------------------------------------------* Decoding

    def decode(self, item: dict) -> MongoGlyphs:
        return MongoGlyphs(
            name=item["name"],
            size=item["size"],
            panose=tensor(item["panose"]),
            en=tensor(pkl.loads(item["en"])).view(-1, item["size"], item["size"]),
            ua=tensor(pkl.loads(item["ua"])).view(-1, item["size"], item["size"]),
        )

    # *------------------------------------------------------------------------*  Loading & Saving

    def save(self, base, train, test, dir: str = "./data") -> None:
        with open(f"{dir}/base.pkl", "wb") as O:
            pkl.dump(base, O)

        with open(f"{dir}/train.pkl", "wb") as O:
            pkl.dump(train, O)

        with open(f"{dir}/test.pkl", "wb") as O:
            pkl.dump(test, O)


    def download(self, size: int = 64, base_font: str = "Open Sans Light", dir: str = "./data") -> None:
        """Downloads data, splits it into train and test and saves it locally"""

        base = None # single content reference font
        train = [] # fonts with Latin and Cyrillic
        test = [] # Latin-only fonts

        data: Cursor = self.getFonts(size)

        for item in data:
            font = self.decode(item)
            print(f"{font.name} →", end=" ")

            if font.name == base_font:
                base = font
                print("Base")
            elif len(font.en) == 0:
                print("Skipped")
            elif len(font.ua) > 0:
                train.append(font)
                print("Train")
            else:
                test.append(font)
                print("Test")

        self.save(base, train, test, dir)

# *----------------------------------------------------------------------------* Data Caching


def load(path: str, dir: str = "./data") -> Union[MongoGlyphs, List[MongoGlyphs]]:
    with open(f"{dir}/{path}.pkl", "rb") as I:
        return pkl.load(I)

