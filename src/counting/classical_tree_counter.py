"""
Example usage of Tree Counter class
"""

from src.orthophotomap.forest_iterator import ForestIterator
import os
import fiona
import rasterio as rio
from shapely.geometry import Point, mapping

import numpy as np
import cv2


def show(img):
    cv2.imshow("img", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


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
        self.params = self._get_blob_params()
        self.return_locations = return_locations

    def _get_blob_params(self):
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 0
        params.maxThreshold = 2000

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 1
        params.maxArea = 40

        # Filter by Circularity
        # params.filterByCircularity = True
        # params.minCircularity = 0.0

        # Filter by Convexity
        # params.filterByConvexity = True
        # params.minConvexity = 0.0

        # Filter by Inertia
        # params.filterByInertia = True
        # params.minInertiaRatio = 0.01

        return params

    def _detect_blobs(self, img, params=None):
        if cv2.__version__.startswith('2.'):
            detector = cv2.SimpleBlobDetector(params)
        else:
            detector = cv2.SimpleBlobDetector_create(params)

        keypoints = detector.detect(img)

        return keypoints

    def _preprocess_forest_img(self, img):
        l_channel = self._apply_brightness_contrast(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 64, 90)
        kernel = np.ones((1, 1), np.uint8)
        ret, mask_r = cv2.threshold(l_channel, 140, 255, cv2.THRESH_BINARY)
        mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, kernel)
        mask_r = np.uint8(mask_r)

        return mask_r

    def _apply_brightness_contrast(self, input_img, brightness=0, contrast=0):
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
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)

            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        return buf

    def count(self, rgb_image: np.ndarray,
              forest_mask: np.ndarray):
        assert 3 == len(rgb_image.shape), \
            "RGB image array should be 3-dimensional"
        assert 2 == len(forest_mask.shape), \
            "Forest mask array should be 2-dimensional"
        assert rgb_image.shape[:2] == forest_mask.shape, \
            "Forest mask should have the same height and width as RGB"

        count = 0
        trees_points = []
        all_key_points = []

        masked_rgb = cv2.bitwise_and(rgb_image, rgb_image, mask=forest_mask)


        masked_rgb = self._preprocess_forest_img(masked_rgb)

        # show(masked_rgb)

        masked_rgb = cv2.bitwise_not(masked_rgb)


        keypoints = self._detect_blobs(img=masked_rgb, params=self.params)

        all_key_points += keypoints
        trees_points += [k.pt for k in keypoints]
        count += len(trees_points)


        return {"trees": trees_points, "count": count, "keypoints": all_key_points, "mask": masked_rgb}


if __name__ == '__main__':
    WINDOW_SIZE = 500
    X = 1000
    Y = 1000

    name = 'Swiebodzin'
    path = "/media/piotr/824F-8A2A/Swiebodzin/"

    shape_path = os.path.join(path, 'obszar_' + name.lower() + '.shp')
    shapes = fiona.open(shape_path)
    rgb_path = os.path.join(path, 'RGB_' + name + '.tif')
    nir_path = os.path.join(path, 'NIR_' + name + '.tif')
    it = ForestIterator(rgb_path, shape_path, nir_path)

    patch = it[53]
    rgb = patch['rgb']

    rgb = np.moveaxis(rgb, 0, -1)

    forest_img = rgb

    tree_couter = TreeCounter()

    # we assume all image is a forest, it is not a case always but for now it will be suficient
    mask = np.ones_like(forest_img)[:, :, 2]

    counting_dict = tree_couter.count(forest_img, mask)

    schema = {
    'geometry': 'Point',
    }


    # Write a new Shapefile
    output_shapefile = fiona.open('trees.shp', 'w', 'ESRI Shapefile', schema)

    trees = counting_dict["trees"]

    for x, y in trees:
        x = x+path["x_min"]
        y = y + path["y_min"]

        point = Point(rio.transform.xy(it.rgb_tif_handler.transformm, y, x))
        output_shapefile.write(mapping(point))


