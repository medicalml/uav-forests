import numpy as np


class TreeCounter:

    def __init__(self, *args,
                 return_locations: bool = False,
                 **kwargs, ):
        '''
        Any required arguments for the algorithm 
        that stay unchanged for every run 
        on every forest part, and any required
        initialisation.
        '''
        self.return_locations = return_locations

    def count(self, rgb_image: np.ndarray, ndvi_image: np.ndarray,
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
        trees_points = [(14, 20), (444, 94)]  # (row, column)
        count = 17
        if self.return_locations:
            return {"trees": trees_points, "count": count}
        else:
            return count
