import numpy as np


def nir_to_ndvi(nir_img, red_channel_img):

    if np.issubdtype(red_channel_img.dtype, np.integer):
        norm_factor = 255
    elif np.issubdtype(red_channel_img.dtype, np.float):
        norm_factor = 1.0
    else:
        norm_factor = red_channel_img.max()

    red_img_norm = red_channel_img / norm_factor
    
        
    with np.errstate(divide='raise'):
        try:
            ndvi = (nir_img - red_img_norm) / (red_img_norm + nir_img)   # this gets caught and handled as an exception
        except RuntimeWarning:
            print('oh no!')
            return None
    ndvi = (ndvi + 1) / 2
    ndvi = np.float32(ndvi)
    return ndvi