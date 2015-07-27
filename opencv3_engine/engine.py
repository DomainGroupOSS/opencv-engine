from io import BytesIO
from tempfile import NamedTemporaryFile
from get_image_size import get_image_size
import cv2

from thumbor.engines import BaseEngine
from pexif import JpegFile, ExifSegment
import numpy as np
from PIL import Image

C_NO_WEBP_OUTPUT = 'OPENCV3_ENGINE_NO_WEBP_OUTPUT'

C_ENGINE_TMP_DIR = 'OPENCV3_ENGINE_TMP_DIR'

C_SCALE_ON_LOAD = 'OPENCV3_ENGINE_SCALE_ON_LOAD'

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
        if image_format is F_GIF:
            # OpenCV doesn't have GIF encoder, fall back to PIL.
            self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
            img = Image.fromarray(self.image)
            b = BytesIO()
            img.save(b, F_GIF)
            data = b.getvalue()
            return data
        else:
            if self.context.config.get(C_NO_WEBP_OUTPUT, False):
                # webp encoding is slow, output jpeg
                image_format = F_JPEG
                extension = '.jpeg'
            if image_format is F_JPEG or image_format is F_WEBP:
                if quality is None:
                    quality = self.context.config.QUALITY
                quality_option = cv2.IMWRITE_JPEG_QUALITY if image_format is F_JPEG else cv2.IMWRITE_WEBP_QUALITY
                options = [quality_option, quality]
            data = cv2.imencode(extension, self.image, options or [])[1].tostring()

            if image_format is F_JPEG and self.context.config.PRESERVE_EXIF_INFO and hasattr(self, 'exif'):
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
        image_format = self._get_format()
        if image_format is F_GIF:
            # OpenCV doesn't support GIF, fallback to PIL when decoding GIFs.
            image_open = Image.open(BytesIO(buffer)).convert('RGB')
            img = np.array(image_open)
            # http://stackoverflow.com/a/11590526/272824 OpenCV uses BGR.
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif self.context.config.get(C_SCALE_ON_LOAD, False):
            self._f = NamedTemporaryFile(bufsize=len(buffer), dir=self.context.config.get(C_ENGINE_TMP_DIR, None))
            self._f.write(buffer)
            # just read image width and height
            width, height = get_image_size(self._f.name)
            factor = 1
            out_width = self.context.request.width
            out_height = self.context.request.height
            while width/factor > out_width*2 and height*2/factor > 2*out_height:
                factor *= 2
            img = cv2.imread_reduced(self._f.name, cv2.IMREAD_UNCHANGED, scale_denom=factor)
            self._f.close()
            # TODO: figure out what this was doing.
            # imagefiledata = cv.CreateMatHeader(1, len(buffer), cv.CV_8UC1)
            # cv.SetData(imagefiledata, buffer, len(buffer))
        else:
            img = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_UNCHANGED)
        if image_format is F_JPEG and self.context.config.PRESERVE_EXIF_INFO:
            try:
                info = JpegFile.fromString(buffer).get_exif()
                if info:
                    self.exif = info.data
                    self.exif_marker = info.marker
            except Exception:
                pass
        return img

    def resize(self, width, height):
        size = (int(round(width, 0)), int(round(height, 0)))
        current_width, current_height = self.size
        downscale = width < current_width and height < current_height
        if downscale:
            # When down sampling use INTER_AREA as per
            # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
            interpolation_method = cv2.INTER_AREA
        else:
            interpolation_method = cv2.INTER_LINEAR
        self.image = cv2.resize(src=self.image, dsize=size, interpolation=interpolation_method)

    @property
    def size(self):
        width = self.image.shape[1]
        height = self.image.shape[0]
        return width, height
