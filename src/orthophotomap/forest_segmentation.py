import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk
import cv2


class ForestSegmentation:
    def __init__(self):
        pass

    @staticmethod
    def _apply_brightness_contrast(input_img: np.ndarray, brightness: int = 0, contrast: int = 0):
        '''
        Applies brightness and contrast correction to the image
        :param input_img: numpy array with image, channels last, valued 0-255
        :param brightness: Value of brightness correction, integer
        :param contrast: Value of contrast correction, integer
        :return: ndarray of the input shape with corrected image. Values 0-255
        '''
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

    def mask(self, rgb_image: np.ndarray):
        '''
        Mask the provided image, removing most of the non-forest areas
        :param rgb_image: Channels last rgb image of values 0-255
        :return: single channel mask of 0/1 values of shape same as provided image, but single channel
        '''

        assert 3 == len(rgb_image.shape), \
            "RGB image array should be 3-dimensional"

        # Histogram equalization rgb_image to img
        img_yuv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

        # Bilateral Filter and shadow detection
        bilateral = cv2.bilateralFilter(img, 21, 160, 168)
        rgb_planes = cv2.split(bilateral)
        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_norm_planes.append(norm_img)
        result_norm = cv2.merge(result_norm_planes)

        # Shadow removal, and initial entropy filtering
        r_gray = cv2.cvtColor(result_norm, cv2.COLOR_RGB2GRAY)
        r_gray = self._apply_brightness_contrast(r_gray, 64, 0)
        entr = entropy(r_gray, disk(30))
        entr = (entr - entr.min()) / (entr.max() - entr.min())
        entr = cv2.GaussianBlur(entr, (5, 5), cv2.BORDER_DEFAULT)
        kernel = np.ones((15, 15), np.uint8)
        ret, mask_r = cv2.threshold(entr, 0.65, 1, cv2.THRESH_BINARY)
        mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, kernel)
        mask_r = np.uint8(mask_r)

        # Masking the image
        img_new = cv2.bitwise_and(rgb_image, rgb_image, mask=mask_r)

        # Final Entropy filtering, removing excess material
        l_channel = cv2.cvtColor(img_new, cv2.COLOR_RGB2GRAY)
        l_channel = self._apply_brightness_contrast(l_channel, 64, 30)
        entr_new = entropy(l_channel, disk(45))
        entr_new = (entr_new - entr_new.min()) / (entr_new.max() - entr_new.min())
        kernel = np.ones((15, 15), np.uint8)
        ret, mask_r_new = cv2.threshold(entr_new, 0.65, 1, cv2.THRESH_BINARY)
        mask_r_new = cv2.morphologyEx(mask_r_new, cv2.MORPH_OPEN, kernel)
        mask_r_new = np.uint8(mask_r_new)

        return mask_r_new
