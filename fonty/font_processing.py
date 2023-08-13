import os
import tempfile

import requests

from collections import namedtuple

import fontforge
import cairosvg
import svgpathtools

import matplotlib.pyplot as plt

from bs4 import BeautifulSoup

import errors


Glyph = namedtuple('Glyph', ['d', 'glyph_name', 'unicode'])


class EmptyPathError(errors.FontyError):
    pass


class FontProcessor:

    def __init__(self, input_address: str, flip_v: bool = False, flip_h: bool = False):
        """Opens font and reads all the glyphs in it.

        Arguments:
            flip_v, flip_h: flips glyphs vertically or horizontally
        """
        self._flip_glyphs_v = flip_v
        self._flip_glyphs_h = flip_h

        with tempfile.TemporaryDirectory() as dir:
            font = fontforge.open(input_address)
            file_address = os.path.join(dir, input_address + '._generated.svg')
            font.generate(file_address)
            font.close()

            with open(file_address, mode='r') as fp:
                xml_data = fp.read()
                glyphs = BeautifulSoup(xml_data, 'xml').find_all('glyph')

        self.glyphs = [
            Glyph(item.get('d'), item.get('glyph-name'), item.get('unicode'))
            for item in glyphs
        ]

    @classmethod
    def fromUrl(cls, url: str, extension: str = None):
        """Opens font by URL and reads all the glyphs in it. Extension of the font
        will be identified automatically from the URL. You can type it manually otherwise.
        """
        response = requests.get(url)

        extension = extension or url.split('.')[-1]

        with tempfile.TemporaryDirectory() as dir:
            fname = os.path.join(dir, 'font.' + extension)
            with open(fname, 'wb') as fp:
                fp.write(response.content)

            instance = cls(fname)

        return instance

    def _fill_into_bb(self, object_bb: tuple, rect_sizes: tuple):
        """Calculates translation and scaling for that object with given
        xmax, xmin, ymax, ymin, so that it will be filled and centered inside
        a rectangle with given width and height.
        """

        src_obj_xmax, src_obj_xmin, src_obj_ymax, src_obj_ymin = object_bb
        im_w, im_h = rect_sizes

        src_obj_w = abs(src_obj_xmax - src_obj_xmin)
        src_obj_h = abs(src_obj_ymax - src_obj_ymin)

        if src_obj_w > src_obj_h:
            scale_x = im_w / src_obj_w
            scale_y = scale_x
            dst_obj_w = src_obj_w * scale_x
            dst_obj_h = src_obj_h * scale_y

        else:
            scale_y = im_h / src_obj_h
            scale_x = scale_y
            dst_obj_w = src_obj_w * scale_x
            dst_obj_h = src_obj_h * scale_y

        translate_x = -src_obj_xmin * scale_x + (im_w - dst_obj_w) / 2
        translate_y = -src_obj_ymin * scale_y + (im_h - dst_obj_h) / 2

        if self._flip_glyphs_h:
            scale_x *= -1
            translate_x += dst_obj_w

        if self._flip_glyphs_v:
            scale_y *= -1
            translate_y += dst_obj_h

        return (
            (round(translate_x, 5), round(translate_y, 5)),
            (round(scale_x, 5), round(scale_y, 5))
        )

    def _get_svg_boilerplate(
        self,
        path: str,
        bounding_box: tuple = None,
        image_w: str = 128, image_h: str = 128,
        background: str = None, fill: str = None
    ):
        """Generates SVG code for a given glyph.
        """
        image_size = f'width=\'{image_w}\' height=\'{image_h}\''

        if bounding_box:
            translation, scaling = self._fill_into_bb(bounding_box, (image_w, image_h))
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

    def _get_glyph_bb(self, path: str, fname: str) -> tuple:

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

    def glyph2png(self, glyph: Glyph, fname: str, image_w: int = 128, image_h: int = 128):
        """Renders a glyph to PNG image with the given size.
        """
        with tempfile.TemporaryDirectory() as dir:
            bounding_box = self._get_glyph_bb(glyph.d, os.path.join(dir, fname + '.svg'))
            svg_text = self._get_svg_boilerplate(glyph.d, bounding_box)
            print(svg_text)
            self._svg2png(glyph.d, svg_text, image_w, image_h, fname)

    def glyph2array(self, glyph: Glyph, image_w: int = 128, image_h: int = 128):
        """Converts a given glyph to a pyplot image array with given size.
        Arguments is the same as in Font.glyph2png.
        """
        with tempfile.TemporaryDirectory() as dir:
            glyph_fname = (glyph.unicode or '?').encode().hex() + '.png'
            glyph_path = os.path.join(dir, glyph_fname)
            self.glyph2png(glyph, glyph_path, image_w, image_h)
            return plt.imread(glyph_path)
