import os
import sys
import tempfile

import matplotlib.pyplot as plt

import cairosvg
import svgpathtools


class Font:

    def __init__(
        self, mongo_collection, query: dict,
        render_image_w: int = 64, render_image_h: int = 64,
        render_glyph_proportion: int = 1
    ):
        self._font_object = mongo_collection.find_one(query)
        self._glyphs = [
            Glyph(glyph['letter'], glyph['path'], self)
            for glyph
            in self._font_object['glyphs']
        ]
        self._image_w = render_image_w
        self._image_h = render_image_h
        self._temp_dir = tempfile.TemporaryDirectory()
        self._glyph_proportion = render_glyph_proportion

        try:
            self.baseline_y = self._font_object['baseline_y']
        except AttributeError:
            self.baseline_y = 0

        try:
            self.cap_height = self._font_object['cap_height']
        except AttributeError:
            self.cap_height = 1000
            sys.stderr.write('This font does not contain cap_height property. Assuming cap_height=1000')

    @property
    def glyphs(self):
        return self._glyphs

    @property
    def family_name(self):
        return self._font_object['family_name']

    @property
    def font_name(self):
        return self._font_object['font_name']

    @property
    def panose(self):
        return self._font_object['panose']


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
        vector_ymin, vector_ymax = self._font.baseline_y, self._font.cap_height

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
            image_w=self._font._image_w, image_h=self._font._image_h,
            glyph_size_proportion=self._font._glyph_proportion
        )
        self._svg2png(self._path, svg_text, self._font._image_w, self._font._image_h, fname)

    @property
    def np_array(self):
        glyph_fname = (self._letter or '?').encode().hex() + '.png'
        glyph_path = os.path.join(self._font._temp_dir.name, glyph_fname)
        self.to_png(glyph_path)
        return plt.imread(glyph_path)