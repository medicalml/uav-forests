import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk
import cv2
from src.counting.classical_tree_counter import show


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

    # def mask(self, rgb_image: np.ndarray):
    #
    #     assert 3 == len(rgb_image.shape), \
    #         "RGB image array should be 3-dimensional"
    #     # assert 2 == len(ndvi_image.shape), \
    #     #     "NDVI image array should be 2-dimensional"
    #     # assert rgb_image.shape[:2] == ndvi_image.shape, \
    #     #     "NDVI image should have the same height and width as RGB"
    #
    #     l_channel = self._apply_brightness_contrast(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY),
    #                                                 self.brightness, self.contrast)
    #     entr = entropy(l_channel, disk(self.entropy_window_size))
    #     entr = (entr - entr.min()) / (entr.max() - entr.min())
    #
    #     kernel = np.ones((self.opening_window_size, self.opening_window_size), np.uint8)
    #     ret, mask_r = cv2.threshold(entr, self.brightness_threshold, 1, cv2.THRESH_BINARY)
    #     mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, kernel)
    #     mask_r = np.uint8(mask_r)
    #
    #     return mask_r

    def mask(self, rgb_image: np.ndarray):

        assert 3 == len(rgb_image.shape), \
            "RGB image array should be 3-dimensional"
        img_rgb = rgb_image
        img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        bilateral = cv2.bilateralFilter(img, 21, 160, 168)
        rgb_planes = cv2.split(bilateral)

        result_planes = []
        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_planes.append(diff_img)
            result_norm_planes.append(norm_img)

        result_norm = cv2.merge(result_norm_planes)
        r_gray = cv2.cvtColor(result_norm, cv2.COLOR_RGB2GRAY)
        r_gray = self._apply_brightness_contrast(r_gray, 64, 0)
        entr = entropy(r_gray, disk(30))
        entr = (entr - entr.min()) / (entr.max() - entr.min())
        entr = cv2.GaussianBlur(entr, (5, 5), cv2.BORDER_DEFAULT)
        kernel = np.ones((15, 15), np.uint8)
        ret, mask_r = cv2.threshold(entr, 0.65, 1, cv2.THRESH_BINARY)
        mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, kernel)
        mask_r = np.uint8(mask_r)

        img_new = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_r)
        l_channel = cv2.cvtColor(img_new, cv2.COLOR_RGB2GRAY)
        l_channel = self._apply_brightness_contrast(l_channel, 64, 30)
        entr_new = entropy(l_channel, disk(45))
        entr_new = (entr_new - entr_new.min()) / (entr_new.max() - entr_new.min())

        kernel = np.ones((15, 15), np.uint8)
        ret, mask_r_new = cv2.threshold(entr_new, 0.65, 1, cv2.THRESH_BINARY)
        mask_r_new = cv2.morphologyEx(mask_r_new, cv2.MORPH_OPEN, kernel)
        mask_r_new = np.uint8(mask_r_new)

        return mask_r_new
