import os
import tempfile

from collections import namedtuple

import fontforge

from bs4 import BeautifulSoup


Glyph = namedtuple('Glyph', ['d', 'glyph_name', 'unicode'])


def ttf2tuples(input_address: str):

    with tempfile.TemporaryDirectory() as dir:
        font = fontforge.open(input_address)
        file_address = os.path.join(dir, input_address + '._generated.svg')
        font.generate(file_address)
        font.close()

        with open(file_address, mode='r') as fp:
            xml_data = fp.read()
            glyphs = BeautifulSoup(xml_data, 'xml').find_all('glyph')

    return [
        Glyph(item.get('d'), item.get('glyph_name'), item.get('unicode'))
        for item in glyphs
    ]
