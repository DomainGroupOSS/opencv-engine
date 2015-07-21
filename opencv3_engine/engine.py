import cv2

from colour import Color

from thumbor.engines import BaseEngine
from pexif import JpegFile, ExifSegment
import numpy as np

try:
    from thumbor.ext.filters import _composite
    FILTERS_AVAILABLE = True
except ImportError:
    FILTERS_AVAILABLE = False

FORMATS = {
    '.jpg': 'JPEG',
    '.jpeg': 'JPEG',
    '.gif': 'GIF',
    '.png': 'PNG',
    '.webp': 'WEBP'
}

__author__ = 'konstantin.burov'

class Engine(BaseEngine):
    def read(self, extension, quality):
        if quality is None:
            quality = self.context.config.QUALITY

        options = None
        extension = extension or self.extension
        try:
            if FORMATS[extension] == 'JPEG':
                options = [cv2.IMWRITE_JPEG_QUALITY, quality]
        except KeyError:
            # default is JPEG so
            options = [cv2.IMWRITE_JPEG_QUALITY, quality]

        try:
            if FORMATS[extension] == 'WEBP':
                options = [cv2.IMWRITE_WEBP_QUALITY, quality]
        except KeyError:
            # default is JPEG so
            options = [cv2.IMWRITE_WEBP_QUALITY, quality]

        data = cv2.imencode(extension, self.image, options or [])[1].tostring()

        if FORMATS[extension] == 'JPEG' and self.context.config.PRESERVE_EXIF_INFO:
            if hasattr(self, 'exif'):
                img = JpegFile.fromString(data)
                img._segments.insert(0, ExifSegment(self.exif_marker, None, self.exif, 'rw'))
                data = img.writeString()
        return data

    def crop(self, crop_left, crop_top, crop_right, crop_bottom):
        self.image = self.image[crop_top:crop_bottom, crop_left:crop_right]

    def image_data_as_rgb(self, update_image=True):
        pass

    def create_image(self, buffer):
        # FIXME: opencv doesn't support gifs, even worse, the library
        # segfaults when trying to decoding a gif. An exception is a
        # less drastic measure.
        try:
            if FORMATS[self.extension] == 'GIF':
                raise ValueError("opencv doesn't support gifs")
        except KeyError:
            pass

        # imagefiledata = cv.CreateMatHeader(1, len(buffer), cv.CV_8UC1)
        # cv.SetData(imagefiledata, buffer, len(buffer))
        img0 = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_UNCHANGED)

        if FORMATS[self.extension] == 'JPEG':
            try:
                info = JpegFile.fromString(buffer).get_exif()
                if info:
                    self.exif = info.data
                    self.exif_marker = info.marker
            except Exception:
                pass

        return img0

    def paste(self):
        pass

    def gen_image(self):
        pass

    def flip_vertically(self):
        pass

    def flip_horizontally(self):
        pass

    def set_image_data(self, data):
        pass

    def enable_alpha(self):
        pass

    def resize(self, width, height):
        size = (int(round(width, 0)), int(round(height, 0)))
        self.image = cv2.resize(self.image, size, cv2.INTER_AREA)

    @property
    def size(self):
        height, width, channels = self.image.shape
        return width, height

    def normalize(self):
        pass
