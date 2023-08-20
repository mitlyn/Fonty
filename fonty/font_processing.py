import os
import shutil
import tempfile

import requests

from collections import namedtuple

import fontforge
import cairosvg
import svgpathtools

import matplotlib.pyplot as plt

from bs4 import BeautifulSoup

import csv

import errors
import panose1


Glyph = namedtuple('Glyph', ['d', 'glyph_name', 'unicode', 'attrs'])
GLYPH_FETCH_ATTRS = ['d', 'glyph-name', 'unicode']


ALPHABETS = {
    'ua': [*'АаБбВвГгҐґДдЕеЄєЖжЗзИиІіЇїЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЬьЮюЯя'],
    'en': [*'AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz']
}


class EmptyPathError(errors.FontyError):
    pass


class FontProcessor:

    def __init__(self, input_address: str, flip_v: bool = False, flip_h: bool = False, temp_dir=None):
        """Opens font and reads all the glyphs in it.

        Arguments:
            flip_v, flip_h: flips glyphs vertically or horizontally
        """
        self._flip_glyphs_v = flip_v
        self._flip_glyphs_h = flip_h

        self.temp_dir = temp_dir or tempfile.TemporaryDirectory()
        self._svg_font_address = os.path.join(self.temp_dir.name, input_address + '._generated.svg')

        font = fontforge.open(input_address)
        font.generate(self._svg_font_address)

        self._font = font

        self._cap_height = font.capHeight

        # might fetch baselines from font
        self._baseline_y = 0
        self._baseline_x = 0

        with open(self._svg_font_address, mode='r') as fp:
            xml_data = fp.read()
            glyphs = BeautifulSoup(xml_data, 'xml').find_all('glyph')

        def _convert_to_Glyph(item):
            fetched_attributes = [item.get(attr) for attr in GLYPH_FETCH_ATTRS]
            all_attribute_keys = set(item.attrs.keys())
            rest_attributes = {
                key: item.attrs[key]
                for key in all_attribute_keys - set(GLYPH_FETCH_ATTRS)
            }

            return Glyph(*fetched_attributes, rest_attributes)

        self.glyphs = [_convert_to_Glyph(item) for item in glyphs]

    @property
    def features(self):
        features = {
            feature[4:]: getattr(self._font, feature)
            for feature
            in dir(self._font)
            if feature[:4] == 'os2_' and feature != 'os2_panose'
        }

        return {
            **features,
            'panose': panose1.digits_to_features(self._font.os2_panose),
            'family_name': self._font.familyname,
            'font_name': self._font.fullname
        }

    def __del__(self):
        self._font.close()
        self.temp_dir.close()

    def save_svg_font(self, dst: str = None):
        """Saves a generated SVG font file to a given destination. Original file name will
        be remained of `dst` is None.
        """
        dst = dst or self._font.path.split('/')[-1]
        shutil.copy(self._svg_font_address, dst)

    @classmethod
    def fromUrl(cls, url: str, extension: str = None, *args, **kwargs):
        """Opens font by URL and reads all the glyphs in it. Extension of the font
        will be identified automatically from the URL. You can type it manually otherwise.
        """
        response = requests.get(url)

        extension = extension or url.split('.')[-1]

        temp_dir = tempfile.TemporaryDirectory()
        fname = os.path.join(temp_dir.name, 'font.' + extension)
        with open(fname, 'wb') as fp:
            fp.write(response.content)

        instance = cls(fname, temp_dir=temp_dir, *args, **kwargs)

        return instance

    def _calculate_transform(
        self,
        vector_width: int,
        image_h: int, image_w: int,
        glyph_size_proportion: int
    ):
        vector_ymin, vector_ymax = self._baseline_y, self._cap_height

        glyph_height = (glyph_size_proportion * image_h) / (glyph_size_proportion + 2)
        padding_y = (image_h - glyph_height) / 2

        glyph_ymin, glyph_ymax = padding_y, padding_y + glyph_height

        translating_y = glyph_ymin - vector_ymin
        scaling = (glyph_ymax - glyph_ymin) / (vector_ymax - vector_ymin)

        padding_x = (image_w - vector_width * scaling) / 2

        return (
            (padding_x, image_h - translating_y),
            (
                scaling if not self._flip_glyphs_h else -scaling,
                scaling if not self._flip_glyphs_v else -scaling
            )
        )

    def _get_svg_boilerplate(
        self,
        path: str,
        glyph_size_proportion: int = 1,
        bounding_box: tuple = None,
        image_w: str = 128, image_h: str = 128,
        background: str = None, fill: str = None
    ):
        """Generates SVG code for a given glyph.
        """
        image_size = f'width=\'{image_w}\' height=\'{image_h}\''

        if bounding_box:
            vector_xmax, vector_xmin, vector_ymax, vector_ymin = bounding_box
            translation, scaling = self._calculate_transform(
                vector_width=(vector_xmax - vector_xmin),
                image_w=image_w, image_h=image_h,
                glyph_size_proportion=glyph_size_proportion
            )
        else:
            translation, scaling = (0, 0), (1, 1)

        def transform(fname, args):
            args = map(str, args)
            return fname + '(' + ','.join(args) + ')'

        return f'''
            <svg
                xmlns='http://www.w3.org/2000/svg'
                xmlns:xlink='http://www.w3.org/1999/xlink'
                {image_size}
                viewPort='0 0 {image_w} {image_h}'
            >
                <rect {image_size} fill='{background or '#fff'}' />
                <path
                    d='{path}'
                    fill='{fill or '#000'}'
                    transform='{transform('translate', translation)} {transform('scale', scaling)}'
                />
            </svg>
        '''

    def _svg2png(self, path: str, svg_text: str, image_w: int, image_h: int, dst_file: str):
        """Converts SVG path to PNG data and writes to destination file.
        """
        png_data = cairosvg.svg2png(
            bytestring=svg_text.encode(),
            output_width=image_w,
            output_height=image_h
        )
        with open(dst_file, mode='wb+') as fp:
            fp.write(png_data)

    def _get_glyph_bbox(self, path: str, fname: str) -> tuple:

        def get_paths_bounding_box(paths):
            for i, path in enumerate(paths):
                try:
                    if i == 0:
                        # Initialise the overall min-max with the first path
                        xmin, xmax, ymin, ymax = path.bbox()
                    else:
                        # Expand bounds to match path bounds if needed
                        p_xmin, p_xmax, p_ymin, p_ymax = path.bbox()
                        xmin = p_xmin if p_xmin < xmin else xmin
                        xmax = p_xmax if p_xmax > xmax else xmax
                        ymin = p_ymin if p_ymin < ymin else ymin
                        ymax = p_ymax if p_ymax > ymax else ymax
                except ValueError:
                    raise EmptyPathError

            return xmax, xmin, ymax, ymin

        with open(fname, mode='w+') as f_glyph:
            content = self._get_svg_boilerplate(path)
            f_glyph.write(content)

        paths, _ = svgpathtools.svg2paths(fname)

        return get_paths_bounding_box(paths)

    def glyph2png(
        self,
        glyph: Glyph, fname: str, image_w: int = 128, image_h: int = 128, glyph_size_proportion: int = 1
    ):
        """Renders a glyph to PNG image with the given size.
        """
        bounding_box = self._get_glyph_bbox(glyph.d, os.path.join(self.temp_dir.name, fname + '.svg'))
        svg_text = self._get_svg_boilerplate(
            path=glyph.d, bounding_box=bounding_box,
            image_w=image_w, image_h=image_h,
            glyph_size_proportion=glyph_size_proportion
        )
        self._svg2png(glyph.d, svg_text, image_w, image_h, fname)

    def glyph2array(self, glyph: Glyph, *args, **kwargs):
        """Converts a given glyph to a pyplot image array with given size.
        Arguments is the same as in Font.glyph2png.
        """
        glyph_fname = (glyph.unicode or '?').encode().hex() + '.png'
        glyph_path = os.path.join(self.temp_dir.name, glyph_fname)
        self.glyph2png(glyph, fname=glyph_path, *args, **kwargs)
        return plt.imread(glyph_path)

    def glyphset_from_unicode_subset(self, unicode_subset: list):
        """Generates new glyphset on the given unicode letters.
        """
        glyphs = [
            glyph
            for glyph in self.glyphs
            if glyph.unicode in set(unicode_subset)
        ]

        return Glyphset(glyphs, unicode_subset, self)


class Glyphset:

    def __init__(self, glyphs, letters, font_processor):
        self.glyphs = glyphs
        self.letters = letters
        self.font_processor = font_processor

    def to_dataset(self, one_channel_image: bool = True, *args, **kwargs):
        """Returns array of dataset for the glyphset. All the parameters will be
        passed to FontProcessor.glyph2png.
        """
        def to_image(glyph):
            array = self.font_processor.glyph2array(glyph, *args, **kwargs)
            return array[:, :, 1] if one_channel_image else array

        return [
            {
                'glyph': glyph,
                'letter': letter,
                'image': to_image(glyph)
            }
            for glyph, letter
            in zip(self.glyphs, self.letters)
        ]

    def to_csv(
        self, faddr: int, include_parameters: list = None, headers: bool = True,
        *args, **kwargs
    ):
        """Stores rendered glyphset to CSV file.

        Arguments:
            include_parameters: specify list of parameters of Glyph that will be
                stored in CSV.
            headers (bool)
            ... all other arguments will be passed into Glyphset.to_dataset
        """
        df = self.to_dataset(*args, **kwargs)
        image_length = len(df[0]['image'].reshape(-1))

        with open(faddr, mode='w') as fp:

            writer = csv.writer(fp, quoting=csv.QUOTE_NONNUMERIC)

            if headers:
                writer.writerow([
                    'letter',
                    *include_parameters,
                    *[i + 1 for i in range(image_length)]
                ])

            for row in df:
                writer.writerow([
                    row['letter'],
                    *[getattr(row['glyph'], attr) for attr in include_parameters],
                    *row['image'].reshape(-1)
                ])

    def to_json(self):
        return {
            **self.font_processor.features,
            'glyphs': [
                {
                    'letter': letter,
                    'path': glyph.d
                }
                for glyph, letter
                in zip(self.glyphs, self.letters)
            ]
        }
