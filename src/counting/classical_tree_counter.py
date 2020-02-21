import os

import cv2
import fiona
import numpy as np
import rasterio as rio
from shapely.geometry import Point, mapping
from scipy.ndimage import label, center_of_mass

from src.orthophotomap.forest_iterator import ForestIterator
from src.orthophotomap.forest_segmentation import ForestSegmentation


class TreeCounter:
    def __init__(self, mean_rate=0.75, contrast=128, opening_size=2, threshold=36):
        self.mean_rate = mean_rate
        self.contrast = contrast
        self.opening = (opening_size, opening_size)
        self.threshold = threshold

    def _preprocess_forest_img(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean = np.mean(gray)
        rate = mean/self.mean_rate
        contrast = self.contrast
        # print(rate)
        l_channel = self._apply_brightness_contrast(gray, int(rate), int(contrast))
        # 48, 128
        ret, mask_r = cv2.threshold(l_channel, 140, 255, cv2.THRESH_BINARY)

        kernel = np.ones(self.opening, np.uint8)
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

        masked_rgb = cv2.bitwise_and(rgb_image, rgb_image, mask=forest_mask)
        masked_rgb = self._preprocess_forest_img(masked_rgb)

        labels, count = label(masked_rgb)
        indices_unique, counts_indices = np.unique(labels, return_counts=True)
        centers = center_of_mass(np.ones(labels.shape), labels,
                                 [indices_unique[i] for i in range(len(indices_unique)) if counts_indices[i] > self.threshold])
        return {"trees": centers, "count": count, "mask": masked_rgb}


if __name__ == '__main__':
    name = 'Swiebodzin'
    path = "/home/piotr/Downloads/Swiebodzin 04 1800m las jednolity"

    shape_path = os.path.join(path, 'obszar_' + name.lower() + '.shp')
    shapes = fiona.open(shape_path)
    rgb_path = os.path.join(path, 'RGB_' + name + '.tif')
    nir_path = os.path.join(path, 'NIR_' + name + '.tif')
    schema = {
        'geometry': 'Point',
        'properties': {"id": "int"}
    }
    # Write a new Shapefile
    output_shapefile = fiona.open(os.path.join(path, 'trees.shp'), 'w', 'ESRI Shapefile', schema)
    tree_couter = TreeCounter()
    it = ForestIterator(rgb_path, shape_path, nir_path)
    masking_tool = ForestSegmentation()
    edit_initial_shape = []

    # for patch in tqdm(it):
    patch = it[53]
    rgb = patch['rgb']
    rgb = np.moveaxis(rgb, 0, -1)
    forest_img = rgb

    # we assume all image is a forest, it is not a case always but for now it will be suficient
    mask = masking_tool.mask(forest_img)
    counting_dict = tree_couter.count(forest_img, mask)
    trees = counting_dict["trees"]
    number_of_trees = len(trees)
    edit_initial_shape.append((patch["description"]["id_ob"], number_of_trees))
    for idx, (y, x) in enumerate(trees):
        y_max, x_min = rio.transform.rowcol(it.rgb_tif_handler.transform, patch["x_min"], patch["y_max"])
        y += y_max
        x += x_min
        point = Point(rio.transform.xy(it.rgb_tif_handler.transform, y, x))
        output_shapefile.write({
            'geometry': mapping(point),
            'properties': {'id': idx},
        })

    it.update_shapefile(edit_initial_shape, ["drzewa"])