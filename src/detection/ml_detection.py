import numpy as np


class SickTreesDetectron2Detector:

     def __init__(self, *args, **kwargs):
        '''
        Any required arguments for the algorithm
        that stay unchanged for every run
        on every forest part, and any required
        initialisation.
        '''
        pass

    def detect(self, rgb_image: np.ndarray, ndvi_image: np.ndarray,
              forest_mask: np.ndarray):
        assert 3 == len(rgb_image.shape), \
            "RGB image array should be 3-dimensional"
        assert 2 == len(ndvi_image.shape), \
            "NDVI image array should be 2-dimensional"
        assert 2 == len(forest_mask.shape), \
            "Forest mask array should be 2-dimensional"
        assert rgb_image.shape[:2] == ndvi_image.shape, \
            "NDVI image should have the same height and width as RGB"
        assert rgb_image.shape[:2] == forest_mask.shape, \
            "Forest mask should have the same height and width as RGB"

        # ...
        # ...
        # ...
        
        bboxes = [{"box": (25, 100, 60, 120), "score": 0.72}, 
                  {"box": (66, 20, 100, 32), "score": 0.96]
        # box : (row_min, col_min, row_max, col_max)
        
        return bboxes

