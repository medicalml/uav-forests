import cv2
import numpy as np
from scipy.ndimage import label, center_of_mass
import matplotlib.pyplot as plt
from skimage import filters
from src.utils.image_processing import apply_brightness_contrast


class TreeCounter:
    def __init__(self, goal=1, contrast=128, opening_size=2, threshold=36):
        self.goal = goal
        self.contrast = contrast
        self.opening = (opening_size, opening_size)
        self.threshold = threshold
        self.iterator = 0

    def _preprocess_forest_img(self, img, alpha):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_np = np.ma.masked_array(np.array(gray), 1-(alpha/255))
        try:
            val = filters.threshold_otsu(gray_np.compressed())
        except:
            val = 110
        rate = self.goal * (self.contrast - val)
        contrast = self.contrast
        l_channel = apply_brightness_contrast(gray, int(rate), int(contrast))

        ret, mask_r = cv2.threshold(l_channel, 140, 255, cv2.THRESH_BINARY)

        kernel = np.ones(self.opening, np.uint8)
        mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, kernel)

        mask_r = np.uint8(mask_r)
        return mask_r

    def count(self, rgb_image: np.ndarray, alpha: np.ndarray):
        assert 3 == len(rgb_image.shape), \
            "RGB image array should be 3-dimensional"

        masked_rgb = self._preprocess_forest_img(rgb_image, alpha)

        labels, count = label(masked_rgb)
        indices_unique, counts_indices = np.unique(labels, return_counts=True)
        centers = center_of_mass(np.ones(labels.shape), labels,
                                 [indices_unique[i] for i in range(len(indices_unique)) if
                                  counts_indices[i] > self.threshold])
        # print(centers)
        centers = [center for center in centers if not alpha[int(round(center[0])), int(round(center[1]))] == 0]
        count = len(centers)
        return {"trees": centers, "count": count, "mask": masked_rgb}
