import cv2

from thumbor.engines import BaseEngine
from pexif import JpegFile, ExifSegment
import numpy as np

F_WEBP = 'WEBP'
F_PNG = 'PNG'
F_GIF = 'GIF'
F_JPEG = 'JPEG'

try:
    from thumbor.ext.filters import _composite
    FILTERS_AVAILABLE = True
except ImportError:
    FILTERS_AVAILABLE = False

FORMATS = {
    '.jpg': F_JPEG,
    '.jpeg': F_JPEG,
    '.gif': F_GIF,
    '.png': F_PNG,
    '.webp': F_WEBP
}

__author__ = 'konstantin.burov'

class Engine(BaseEngine):
    def read(self, extension, quality):
        options = None
        image_format = self._get_format(extension)
        if image_format is F_JPEG or image_format is F_WEBP:
            if quality is None:
                quality = self.context.config.QUALITY
            quality_option = cv2.IMWRITE_JPEG_QUALITY if image_format is F_JPEG else cv2.IMWRITE_WEBP_QUALITY
            options = [quality_option, quality]
        data = cv2.imencode(extension, self.image, options or [])[1].tostring()

        if image_format is F_JPEG and self.context.config.PRESERVE_EXIF_INFO:
            if hasattr(self, 'exif'):
                img = JpegFile.fromString(data)
                img._segments.insert(0, ExifSegment(self.exif_marker, None, self.exif, 'rw'))
                data = img.writeString()
        return data

    def _get_format(self, extension=None):
        e = extension or self.extension
        image_format = FORMATS.get(e)
        return image_format

    def crop(self, crop_left, crop_top, crop_right, crop_bottom):
        self.image = self.image[crop_top:crop_bottom, crop_left:crop_right]

    def create_image(self, buffer):
        # FIXME: opencv doesn't support gifs, even worse, the library
        # segfaults when trying to decoding a gif. An exception is a
        # less drastic measure.
        image_format = self._get_format()
        if image_format is F_GIF:
            raise ValueError("opencv doesn't support gifs")

        # TODO: figure out what this was doing.
        # imagefiledata = cv.CreateMatHeader(1, len(buffer), cv.CV_8UC1)
        # cv.SetData(imagefiledata, buffer, len(buffer))
        img0 = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_UNCHANGED)

        if image_format is F_JPEG:
            # TODO: does webp support exif? Maybe preserve the data.
            try:
                info = JpegFile.fromString(buffer).get_exif()
                if info:
                    self.exif = info.data
                    self.exif_marker = info.marker
            except Exception:
                pass

        return img0

    def resize(self, width, height):
        size = (int(round(width, 0)), int(round(height, 0)))
        current_width, current_height = self.size
        if width < current_width and height < current_height:
            # When down sampling use INTER_AREA as per
            # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
            interpolation_method = cv2.INTER_AREA
        else:
            interpolation_method = cv2.INTER_LINEAR
        self.image = cv2.resize(src=self.image, dsize=size, interpolation=interpolation_method)

    @property
    def size(self):
        height, width, channels = self.image.shape
        return width, height

    def normalize(self):
        pass
