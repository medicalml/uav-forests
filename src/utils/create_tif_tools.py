import numpy as np

def pixel_setter(cutted_img_from_map, img):
    #function to only set pixels - usable because iterator return windows which can overlap
    new_img = np.where(cutted_img_from_map==0, img, cutted_img_from_map)
    return new_img