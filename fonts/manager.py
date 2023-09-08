from bson import Binary
from pickle import dumps
from typing import Iterable, List, Set
from torch import Tensor, tensor, empty

from pymongo.results import InsertManyResult, InsertOneResult, DeleteResult

from fonts.font import Font
from fonts.symbols import get_symbols
from fonts.processor import FontProcessor
from fonts.clients import RemoteClient, LocalClient


class FontManager():
    def __init__(self, remoteDB: RemoteClient, localDB: LocalClient):
        self.fonts = remoteDB.col
        self.cache = localDB.col

    # *------------------------------------------------------------------------* Font Management (MongoDB)

    def dropFonts(self) -> DeleteResult:
        return self.fonts.delete_many({})


    def addFonts(self, *urls: Iterable[str]) -> InsertManyResult:
        processors = (FontProcessor.fromUrl(url, flip_v=True) for url in urls)

        glyphsets = (
            processor.glyphset_from_unicode_subset(get_symbols("l"))
            for processor in processors
        )

        payload = (glyphset.to_json() for glyphset in glyphsets)

        return self.fonts.insert_many(payload)


    def addFontFile(self, file: str, drop: bool = False) -> InsertManyResult:
        if drop:
            self.dropFonts()

        with open(file, "r") as I:
            urls = I.readlines()

        return self.addFonts(*urls)

    # *------------------------------------------------------------------------* Payload Construction

    def _encode(
        self, font: Font, en: Tensor, ua: Tensor, size: int = 64
    ) -> dict:
        return {
            "size": size,
            "panose": font["panose"],
            "name": font["font_name"],
            "en": Binary(dumps(en, protocol=2)),
            "ua": Binary(dumps(ua, protocol=2)),
        }


    def _font_to_payload(self, font: Font, size: int = 64) -> dict:
        try:
            en = []
            en.extend(font.render("en"))
            en = tensor(en)[:, :, :, 0].reshape(-1)
        except:
            en = empty(0)

        try:
            ua = []
            ua.extend(font.render("ua"))
            ua = tensor(ua)[:, :, :, 0].reshape(-1)
        except:
            ua = empty(0)

        return self._encode(font, en, ua, size)


    def _fonts_to_payload(self, fonts: Iterable[Font]) -> Iterable[dict]:
        return (self._font_to_payload(font) for font in fonts)

    # *------------------------------------------------------------------------* Rendered Glyphs Management

    def dropGlyphs(self, size: int) -> DeleteResult:
        return self.cache.delete_many({"size": size})


    def getNames(self, size: int = 64) -> Set[str]:
        return set(self.cache.find({"size": size}, {"_id": 0, "name": 1}))


    def render(self, document: dict, size: int = 64) -> InsertOneResult:
        font = Font.fromDocument(document, image_w=size, image_h=size)
        payload = self._font_to_payload(font)
        return self.cache.insert_one(payload)


    def renderMany(self, documents: Iterable[dict], size: int = 64) -> InsertManyResult:
        fonts = (
            Font.fromDocument(document, image_w=size, image_h=size)
            for document in documents
        )

        payload = self._fonts_to_payload(fonts)
        return self.cache.insert_many(payload)


    def renderAll(self, size: int = 64) -> List[str]:
        existing: Set[str] = self.getNames(size)
        loaded: List[str] = []

        documents = filter(
            lambda x: x["font_name"] not in existing,
            self.fonts.find({})
        )

        try:
            for document in documents:
                self.render(document, size)
                loaded.append(document["font_name"])
        except:
            pass
        finally:
            return loaded
