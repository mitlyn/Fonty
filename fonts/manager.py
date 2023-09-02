from bson import Binary
from pickle import dumps
from typing import Generator, Iterable
from torch import Tensor, tensor, empty

from fonts.font import Font
from fonts.client import Client
from fonts.symbols import get_symbols
from fonts.processor import FontProcessor
from fonts.types.mongo import MongoGlyphs


class FontManager(Client):

    # *------------------------------------------------------------------------* Font Source Management

    def addFonts(self, *urls: Iterable[str]):
        processors = (FontProcessor.fromUrl(url, flip_v=True) for url in urls)

        glyphsets = (
            processor.glyphset_from_unicode_subset(get_symbols("l"))
            for processor in processors
        )

        payload = (glyphset.to_json() for glyphset in glyphsets)

        self.fonts.insert_many(payload)


    def addFontFile(self, file: str):
        with open(file, "r") as I:
            urls = I.readlines()

        self.addFonts(*urls)


    # *------------------------------------------------------------------------* Payload Construction


    def _encode(
        self, font: Font, lat: Tensor, cyr: Tensor, size: int = 64
    ) -> MongoGlyphs:
        MongoGlyphs(
            size=size,
            panose=font.panose,
            name=font.font_name,
            lat=Binary(dumps(lat, protocol=2)),
            cyr=Binary(dumps(cyr, protocol=2)),
        )

    def _font_to_payload(self, font: Font, size: int = 64) -> MongoGlyphs:
        try:
            lat = []
            lat.extend(font.render("en"))
            lat = tensor(lat)[:, :, :, 0].reshape(-1)
        except:
            lat = empty(0)

        try:
            cyr = []
            cyr.extend(font.render("ua"))
            cyr = tensor(cyr)[:, :, :, 0].reshape(-1)
        except:
            cyr = empty(0)

        return self._encode(font, lat, cyr, size)

    def _fonts_to_payload(self, fonts: Iterable[Font]) -> Generator[MongoGlyphs]:
        for font in fonts:
            yield self._font_to_payload(font)


    # *------------------------------------------------------------------------* Rendering Of Glyphs


    def renderOne(self, document: dict, size: int = 64):
        font = Font.fromDocument(document, image_w=size, image_h=size)
        payload = self._font_to_payload(font)
        self.glyphs.insert_one(payload)


    def renderMany(self, documents: Iterable[dict], size: int = 64):
        fonts = (
            Font.fromDocument(document, image_w=size, image_h=size)
            for document in documents
        )

        payload = self._fonts_to_payload(fonts)
        self.glyphs.insert_many(payload)


    def renderAll(self, size: int = 64):
        self.glyphs.delete_many({"size": size})

        documents = self.fonts.find({})

        self.renderMany(documents, size)
