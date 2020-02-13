import numpy as np
import sys

def nir_to_ndvi(nir_img, red_channel_img):

    if np.issubdtype(red_channel_img.dtype, np.integer):
        norm_factor = 255
    elif np.issubdtype(red_channel_img.dtype, np.float):
        norm_factor = 1.0
    else:
        norm_factor = red_channel_img.max()
    red_img_norm=red_channel_img/norm_factor
    #red_img_norm = np.where(norm_factor != 0.0, red_channel_img/norm_factor, 0)
    #np.divide(red_channel_img, norm_factor, out=np.zeros(red_channel_img.shape, dtype=np.float), where=norm_factor>0)
    #print(nir_img.dtype, red_img_norm.dtype)
    
    #red_img_norm = np.where(red_img_norm == np.NaN, 0.0 , red_img_norm)
    #red_img_norm = np.where(red_img_norm == np.Inf, 0.0 , red_img_norm)
    
    up = nir_img - red_img_norm
    
   
    down = red_img_norm + nir_img
    ndvi = up/down
    #ndvi = np.where(down!=0, up/down, 0)
    ndvi = (ndvi + 1) / 2
    ndvi = np.float32(ndvi)
    return ndvi