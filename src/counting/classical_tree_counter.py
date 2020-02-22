import cv2
import numpy as np
from scipy.ndimage import label, center_of_mass

from src.utils.image_processing import apply_brightness_contrast


class TreeCounter:
    def __init__(self, mean_rate=0.75, contrast=128, opening_size=2, threshold=36):
        self.mean_rate = mean_rate
        self.contrast = contrast
        self.opening = (opening_size, opening_size)
        self.threshold = threshold

    def _preprocess_forest_img(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean = np.mean(gray)
        rate = mean / self.mean_rate
        contrast = self.contrast
        l_channel = apply_brightness_contrast(gray, int(rate), int(contrast))

        ret, mask_r = cv2.threshold(l_channel, 140, 255, cv2.THRESH_BINARY)

        kernel = np.ones(self.opening, np.uint8)
        mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, kernel)

        mask_r = np.uint8(mask_r)
        return mask_r

    def count(self, rgb_image: np.ndarray,
              forest_mask: np.ndarray):
        assert 3 == len(rgb_image.shape), \
            "RGB image array should be 3-dimensional"
        assert 2 == len(forest_mask.shape), \
            "Forest mask array should be 2-dimensional"
        assert rgb_image.shape[:2] == forest_mask.shape, \
            "Forest mask should have the same height and width as RGB"

        masked_rgb = cv2.bitwise_and(rgb_image, rgb_image, mask=forest_mask)
        masked_rgb = self._preprocess_forest_img(masked_rgb)

        labels, count = label(masked_rgb)
        indices_unique, counts_indices = np.unique(labels, return_counts=True)
        centers = center_of_mass(np.ones(labels.shape), labels,
                                 [indices_unique[i] for i in range(len(indices_unique)) if
                                  counts_indices[i] > self.threshold])
        return {"trees": centers, "count": count, "mask": masked_rgb}
