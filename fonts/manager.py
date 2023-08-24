from pymongo import MongoClient

from torch import tensor
from pickle import loads

from typing import Any, List, Dict
from fonts.types.mongo import Glyphs

# TODO: visualization ?

# *----------------------------------------------------------------------------*

MONGO_HOST = "mongodb+srv://fonty.hquocfa.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority"
MONGO_KEY_PATH = "auth/X509-cert-6658948524510096383.pem"
MONGO_TLS = True

MONGO_DATABASE = "fonty"
MONGO_GLYPHS = "glyphs"
# MONGO_FONTS = "fonts"

# *----------------------------------------------------------------------------*


class Manager:
    def __init__(self):
        self.client = MongoClient(
            MONGO_HOST, tls=MONGO_TLS, tlsCertificateKeyFile=MONGO_KEY_PATH
        )

        self.db = self.client[MONGO_DATABASE]
        self.col = self.db[MONGO_GLYPHS]

        self.projection = {
            "_id": 0,
            "name": 1,
            "size": 1,
            "panose": 1,
            "lat": 1,
            "cyr": 1,
        }

    def getNames(self):
        return self.col.distinct("name")

    def getOne(self, name: str, size: int = 64) -> Dict[str, Any]:
        return self.col.find_one({"name": name, "size": size}, self.projection)

    def getAll(self, size: int = 64) -> Dict[str, Any]:
        return self.col.find({"size": size}, self.projection)

    def toGlyphs(self, item: dict) -> Glyphs:
        return Glyphs(
            item["name"],
            item["size"],
            item["panose"],
            tensor(loads(item["lat"])).view(-1, item["size"], item["size"]),
            tensor(loads(item["cyr"])).view(-1, item["size"], item["size"]),
        )

    def load(self, size: int = 64, *names: List[str]) -> Dict[str, Glyphs]:
        return {name: self.toGlyphs(self.getOne(name, size)) for name in names}

    def loadAll(self, size: int = 64) -> Dict[str, Glyphs]:
        return {font["name"]: self.toGlyphs(font) for font in self.getAll(size)}
