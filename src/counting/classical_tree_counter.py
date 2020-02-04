"""
Example usage of Tree Counter class
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) ) #import parent of parent directory of current module directory
from src.orthophotomap.forest_iterator import ForestIterator

import fiona

import numpy as np
import cv2
import tqdm
from shapely.geometry import Point, mapping, Polygon
import rasterio as rio 
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
        params.maxThreshold = 100

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 1
        params.maxArea = 10

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.0

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.0

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.01

        return params

    def _detect_blobs(self, img, params=None):
        if cv2.__version__.startswith('2.'):
            detector = cv2.SimpleBlobDetector(params)
        else:
            detector = cv2.SimpleBlobDetector_create(params)

        keypoints = detector.detect(img)

        return keypoints

    def _preprocess_forest_img(self, img):
        #print(img.shape)
        l_channel = self._apply_brightness_contrast(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 64, 90)
        kernel = np.ones((3, 3), np.uint8)
        ret, mask_r = cv2.threshold(l_channel, 170, 255, cv2.THRESH_BINARY)
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

        masked_rgb = cv2.bitwise_not(masked_rgb)
        if masked_rgb is None:
            return {"trees": None, "count": 0, "keypoints": None}
        masked_rgb = self._preprocess_forest_img(masked_rgb)

        keypoints = self._detect_blobs(img=masked_rgb, params=self.params)

        all_key_points += keypoints
        trees_points += [k.pt for k in keypoints]
        count += len(trees_points)


        return {"trees": trees_points, "count": count, "keypoints": all_key_points}


if __name__ == '__main__':
    WINDOW_SIZE = 500
    X = 1000
    Y = 1000

    name = 'Swiebodzin'
    path = "/home/h/ML dane dla kola/Swiebodzin/"

    shape_path = os.path.join(path, 'obszar_' + name.lower() + '.shp')
    shapes = fiona.open(shape_path)
    rgb_path = os.path.join(path, 'RGB_' + name + '.tif')
    schema = {
    'geometry': 'Point',
    'properties': {'id': 'int'},
    }
    schema_polygon = {
    'geometry': 'Polygon',
    'properties': {'id': 'int'},
    }

    # Write a new Shapefile
    c = fiona.open('trees.shp', 'w', 'ESRI Shapefile', schema)
    tools = fiona.open('tools.shp', 'w', 'ESRI Shapefile', schema_polygon)
    
    geotiff = rio.open(rgb_path)
    nir_path = os.path.join(path, 'NIR_' + name + '.tif')
    it = ForestIterator(rgb_path, shape_path, nir_path)
    for i, patch in enumerate(tqdm.tqdm(it)):        
        rgb = patch['rgb']
        x_min, y_min = patch['left_upper_corner_coordinates']

        x_max, y_max = patch['right_lower_corner_coordinates']
        rgb = np.moveaxis(rgb, 0, -1)

        forest_img = rgb#rgb[: WINDOW_SIZE, :WINDOW_SIZE, :]

        tree_couter = TreeCounter()

        # we assume all image is a forest, it is not a case always but for now it will be suficient
        mask = np.ones_like(forest_img)[:, :, 2]
        if forest_img is None or mask is None:
            print("problem")
            continue
        counting_dict = tree_couter.count(forest_img, mask)
        

        keypoints = counting_dict["keypoints"]

        
    
        if keypoints is not None:
            for nr, keypoint in enumerate(keypoints):
                x,y = keypoint.pt
                y_min_pixels, x_min_pixels = rio.transform.rowcol(geotiff.transform, x_min, y_min)
                point = Point(rio.transform.xy(geotiff.transform, y_min_pixels+y, x_min_pixels+x))
                coors_to_write = []
                coors_to_write.append([x_min, y_min])
                coors_to_write.append([x_max, y_min ])
                coors_to_write.append([x_max, y_max])
                coors_to_write.append([x_min, y_max ])
                
                poly_tool = Polygon([coors_to_write[0], coors_to_write[1], coors_to_write[2], coors_to_write[3], coors_to_write[0]])
                c.write({
                    'geometry': mapping(point),
                    'properties': {'id': i*1000+nr},
                })
                tools.write({
                    'geometry': mapping(poly_tool),
                    'properties': {'id': i*1000+nr},
                })
                
            imgKeyPoints = cv2.drawKeypoints(forest_img, keypoints, np.array([]), (0, 0, 255),
                                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            cv2.imwrite("keypoints/"+str(i*1000)+".png", imgKeyPoints)
    c.close()
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

'''
python src/counting/classical_tree_counter.py      
'''