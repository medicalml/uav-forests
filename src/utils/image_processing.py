import numpy as np
import cv2

CONTRAST_FACTOR_RANGE = 131
CONTRAST_FACTOR_HALF = 127


def sliding_window_iterator(image: np.ndarray, window_size: int,
                            step_size: int = None):

    step_size = step_size or window_size

    for row in range(0, image.shape[0], step_size):
        for col in range(0, image.shape[1], step_size):
            window = image[row: row + window_size,
                           col: col + window_size]

            yield (row, col, window)


def apply_brightness_contrast(input_img, brightness=0, contrast=0):
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
        f = CONTRAST_FACTOR_RANGE * (contrast + CONTRAST_FACTOR_HALF) / (
                CONTRAST_FACTOR_HALF * (CONTRAST_FACTOR_RANGE - contrast))
        alpha_c = f
        gamma_c = CONTRAST_FACTOR_HALF * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf
