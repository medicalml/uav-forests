import numpy as np


class ForestSegmentation:

    def __init__(self, *args, **kwargs):
        '''
        Any required arguments for the algorithm 
        that stay unchanged for every run 
        on every forest part, and any required
        initialisation.
        '''
    pass

    def mask(self, rgb_image: np.ndarray, ndvi_image: np.ndvi_image):
        assert 3 == len(rgb_image.shape), \
            "RGB image array should be 3-dimensional"
        assert 2 == len(ndvi_image.shape), \
            "NDVI image array should be 2-dimensional"
        assert rgb_image.shape[:2] == ndvi_image.shape, \
            "NDVI image should have the same height and width as RGB"

        # ...
        # ...
        # ...
        return np.ones(ndvi_image.shape)
