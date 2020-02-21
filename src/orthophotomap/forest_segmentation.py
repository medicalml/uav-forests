import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk
import cv2


class ForestSegmentation:
    def __init__(self,
                 brightness: int = 64,
                 contrast: int = 90,
                 entropy_window_size: int = 15,
                 opening_window_size: int = 7,
                 brightness_threshold: float = 0.6):

        assert brightness > 0, \
            "Brightness should be positive integer"
        assert contrast > 0, \
            "Contrast should be positive integer"
        assert entropy_window_size > 1, \
            "Entropy window size should be greater than 1"
        assert entropy_window_size % 2 == 1, \
            "Entropy window size should be odd number"
        assert opening_window_size > 1, \
            "Opening window size should be greater than 1"
        assert opening_window_size % 2 == 1, \
            "Opening window size should be odd"
        assert 0 < brightness_threshold <= 1, \
            "Brightness threshold should be a number from range (0,1]"

        self.brightness = brightness
        self.contrast = contrast
        self.entropy_window_size = entropy_window_size
        self.opening_window_size = opening_window_size
        self.brightness_threshold = brightness_threshold

    @staticmethod
    def _apply_brightness_contrast(input_img: np.ndarray, brightness: int = 0, contrast: int = 0):

        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow) / 255
            gamma_b = shadow

            buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
        else:
            buf = input_img.copy()

        if contrast != 0:
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)

            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        return buf

    def mask(self, rgb_image: np.ndarray, ndvi_image: np.ndvi_image):

        assert 3 == len(rgb_image.shape), \
            "RGB image array should be 3-dimensional"
        assert 2 == len(ndvi_image.shape), \
            "NDVI image array should be 2-dimensional"
        assert rgb_image.shape[:2] == ndvi_image.shape, \
            "NDVI image should have the same height and width as RGB"

        l_channel = self._apply_brightness_contrast(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY),
                                                    self.brightness, self.contrast)
        entr = entropy(l_channel, disk(self.entropy_window_size))
        entr = (entr - entr.min()) / (entr.max() - entr.min())

        kernel = np.ones((self.opening_window_size, self.opening_window_size), np.uint8)
        ret, mask_r = cv2.threshold(entr, self.brightness_threshold, 1, cv2.THRESH_BINARY)
        mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, kernel)
        mask_r = np.uint8(mask_r)

        return mask_r
