import numpy as np


def nir_to_ndvi(nir_img, red_channel_img):
    red_channel_img_norm = red_channel_img / np.max(red_channel_img)
    ndvi = (nir_img - red_img_norm) / (red_img_norm + nir_img)
    ndvi = (ndvi + 1) / 2
    ndvi = np.float32(ndvi)
    return ndvi
