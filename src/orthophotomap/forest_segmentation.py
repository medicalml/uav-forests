import cv2
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk

from src.utils.image_processing import apply_brightness_contrast


class ForestSegmentation:
    def __init__(self, shadow_brightness_correction=64, shadow_contrast_correction=0, shadow_opening_size=15,
                 shadow_entropy_size=30, shadow_threshold=0.65, rgb_brightness_correction=64,
                 rgb_contrast_correction=30, rgb_opening_size=15, rgb_entropy_size=45, rgb_threshold=0.65):
        self.shadow_brightness_correction = shadow_brightness_correction
        self.shadow_contrast_correction = shadow_contrast_correction
        self.shadow_opening_size = shadow_opening_size
        self.shadow_entropy_size = shadow_entropy_size
        self.shadow_threshold = shadow_threshold

        self.rgb_brightness_correction = rgb_brightness_correction
        self.rgb_contrast_correction = rgb_contrast_correction
        self.rgb_opening_size = rgb_opening_size
        self.rgb_entropy_size = rgb_entropy_size
        self.rgb_threshold = rgb_threshold

    def equalize_histogram(self, rgb_image):
        '''
        Equalize histogram in YUV colorspace
        :param rgb_image: input image valued 0-255, 3 channels
        :return: rgb image with equalized histogram
        '''
        img_yuv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    def bilateral_shadow_detection(self, rgb_image):
        '''
        Filter bilaterally, then leave the shadows
        :param rgb_image: input image valued 0-255, 3 channels
        :return: Shadows only image
        '''
        bilateral = cv2.bilateralFilter(rgb_image, 21, 160, 168)
        rgb_planes = cv2.split(bilateral)
        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_norm_planes.append(norm_img)
        return cv2.merge(result_norm_planes)

    def create_shadow_mask(self, rgb_shadowed):
        '''
        Change shadows only image to mask. Shadows between trees left unchanged
        :param rgb_shadowed: Shadows only image, normalized
        :return: mask with the shadows cut out
        '''
        r_gray = cv2.cvtColor(rgb_shadowed, cv2.COLOR_RGB2GRAY)
        r_gray = apply_brightness_contrast(r_gray, self.shadow_brightness_correction, self.shadow_contrast_correction)
        entr = entropy(r_gray, disk(self.shadow_entropy_size))
        entr = (entr - entr.min()) / (entr.max() - entr.min())
        entr = cv2.GaussianBlur(entr, (5, 5), cv2.BORDER_DEFAULT)
        kernel = np.ones((self.shadow_opening_size, self.shadow_opening_size), np.uint8)
        ret, mask_r = cv2.threshold(entr, self.shadow_threshold, 1, cv2.THRESH_BINARY)
        mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, kernel)
        return np.uint8(mask_r)

    def filter_entropy_by_color(self, rgb_img):
        '''
        Entropy filter the image, to leave textures similiar to trees
        :param rgb_img: rgb image valued 0-255, 3 channels
        :return: mask of everything without tree texture
        '''
        l_channel = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        l_channel = apply_brightness_contrast(l_channel, self.rgb_brightness_correction, self.rgb_contrast_correction)
        entr_new = entropy(l_channel, disk(self.rgb_entropy_size))
        entr_new = (entr_new - entr_new.min()) / (entr_new.max() - entr_new.min())
        kernel = np.ones((self.rgb_opening_size, self.rgb_opening_size), np.uint8)
        ret, mask_r_new = cv2.threshold(entr_new, self.rgb_threshold, 1, cv2.THRESH_BINARY)
        mask_r_new = cv2.morphologyEx(mask_r_new, cv2.MORPH_OPEN, kernel)
        return np.uint8(mask_r_new)

    def mask(self, rgb_image: np.ndarray):
        '''
        Mask the provided image, removing most of the non-forest areas
        :param rgb_image: Channels last rgb image of values 0-255
        :return: single channel mask of 0/1 values of shape same as provided image, but single channel
        '''

        assert 3 == len(rgb_image.shape), \
            "RGB image array should be 3-dimensional"

        img = self.equalize_histogram(rgb_image)
        result = self.bilateral_shadow_detection(img)
        mask_r = self.create_shadow_mask(result)
        img_new = cv2.bitwise_and(rgb_image, rgb_image, mask=mask_r)
        mask_r_new = self.filter_entropy_by_color(img_new)

        return mask_r_new
