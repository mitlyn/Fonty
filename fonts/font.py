import os
import sys
import tempfile

import matplotlib.pyplot as plt

import cairosvg
import svgpathtools

import itertools


ALPHABETS = {
    'ua': [*'АаБбВвГгҐґДдЕеЄєЖжЗзИиІіЇїЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЬьЮюЯя'],
    'en': [*'AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz'],
    'puncts': [*',.;:\'"()/[]{}\\/!@#$%^&*?-+=*<>'],
    'digits': '0123456789'
}


def get_alphabet(name: str = None):
    """Returns alphabets from ALPHABETS. Returns all alphabets if not arguments
    prodived.
    """
    if name != 'all':
        return ALPHABETS[name]
    else:
        return list(itertools.chain.from_iterable(ALPHABETS.values()))


class Font:

    def __init__(self, glyphs: list, properties: dict):
        self._glyphs = [
            Glyph(glyph['letter'], glyph['path'], self)
            for glyph in glyphs
        ]
        self._properties = properties
        self._temp_dir = tempfile.TemporaryDirectory()

    @classmethod
    def fromProperties(
        cls, glyphs: list,
        family_name: str, font_name: str,
        panose: dict,
        image_w: int = 64, image_h: int = 64,
        glyph_proportion: int = 1,
        baseline_y: int = 0, cap_height: int = 1000
    ):
        return cls(
            glyphs=glyphs,
            properties={
                'image_w': image_w,
                'image_h': image_h,
                'glyph_proportion': glyph_proportion,
                'baseline_y': baseline_y,
                'cap_height': cap_height,
                'family_name': family_name,
                'font_name': font_name,
                'panose': panose
            }
        )

    @classmethod
    def fromMongoQuery(cls, mongo_collection, query: dict, *args, **kwargs):
        font = mongo_collection.find_one(query)

        try:
            cap_height = font['cap_height']
        except KeyError:
            cap_height = 1000
            print('This font does not contain cap_height property. Assuming cap_height=1000', file=sys.stderr)

        try:
            baseline_y = font['baseline_y']
        except KeyError:
            baseline_y = 0
            print('This font does not contain baseline_y property. Assuming baseline_y=0', file=sys.stderr)

        return cls.fromProperties(
            glyphs=font['glyphs'],
            family_name=font['family_name'],
            font_name=font['font_name'],
            panose=font['panose'],
            cap_height=cap_height,
            baseline_y=baseline_y,
            *args, **kwargs
        )

    def __getitem__(self, property_name):
        return self._properties[property_name]

    def __setitem__(self, property_name, value):
        self._properties[property_name] = value

    def copy_from_unicode_subset(self, unicode_subset: list):
        """Generates new glyphset on the given unicode letters.
        """
        unique_letters = list(set(unicode_subset))

        glyphs = [
            {
                'letter': glyph._letter,
                'path': glyph._path
            }
            for glyph in self._glyphs
            if glyph._letter in unique_letters
        ]

        return Font.fromProperties(glyphs=glyphs, **self._properties)

    @property
    def ua_subset(self):
        return self.copy_from_unicode_subset(ALPHABETS['ua'])

    @property
    def en_subset(self):
        return self.copy_from_unicode_subset(ALPHABETS['en'])


class Glyph:

    def __init__(self, letter, path, font):
        self._letter = letter
        self._path = path
        self._font = font
        self._flip_glyphs_v = True
        self._flip_glyphs_h = False

    def _calculate_transform(
        self,
        vector_width: int,
        image_h: int, image_w: int,
        glyph_size_proportion: int
    ):
        vector_ymin, vector_ymax = self._font['baseline_y'], self._font['cap_height']

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

    def _get_glyph_bbox(self, path: str, fname: str) -> tuple:

        def get_paths_bounding_box(paths):
            for i, path in enumerate(paths):
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

            return xmax, xmin, ymax, ymin

        with open(fname, mode='w+') as f_glyph:
            content = self._get_svg_boilerplate(path)
            f_glyph.write(content)

        paths, _ = svgpathtools.svg2paths(fname)

        return get_paths_bounding_box(paths)

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

    def to_png(self, fname: str):
        bounding_box = self._get_glyph_bbox(
            self._path, os.path.join(self._font._temp_dir.name, fname + '.svg')
        )
        svg_text = self._get_svg_boilerplate(
            path=self._path, bounding_box=bounding_box,
            image_w=self._font['image_w'], image_h=self._font['image_h'],
            glyph_size_proportion=self._font['glyph_proportion']
        )
        self._svg2png(self._path, svg_text, self._font['image_w'], self._font['image_h'], fname)

    @property
    def np_array(self):
        glyph_fname = (self._letter or '?').encode().hex() + '.png'
        glyph_path = os.path.join(self._font._temp_dir.name, glyph_fname)
        self.to_png(glyph_path)
        return plt.imread(glyph_path)
