'''
File for scoring the tree counter. NOT FOR USE, ONLY TESTS.
'''
import os
import random

import fiona
import numpy as np
import cv2
import rasterio
from scipy.spatial import KDTree
from pykdtree.kdtree import KDTree

from src.counting.classical_tree_counter import TreeCounter
from src.orthophotomap.forest_segmentation import ForestSegmentation


# hyperparameters
DOT_SIZE = 3
DOT_COLOR = [0, 0, 255]


def get_trees_positions_and_rgb_img(rgb_path, shape_path):
    shp_handler = fiona.open(shape_path)

    lon = []
    lan = []

    for shape in shp_handler:
        x, y = shape['geometry']['coordinates']
        lon.append(x)
        lan.append(y)

    lon = np.array(lon)
    lan = np.array(lan)

    with rasterio.open(rgb_path) as tif_handler:
        row_start, col_start = tif_handler.index(lon.min(), lan.min())
        row_stop, col_stop = tif_handler.index(lon.max(), lan.max())

        win = rasterio.windows.Window.from_slices((row_stop, row_start), (col_start, col_stop))
        print(win)

        img = np.moveaxis(np.stack([tif_handler.read(i + 1, window=win) for i in range(3)]), 0, -1)

        pixel_positions = [tif_handler.index(x, y) for x, y in zip(lon, lan)]

    x_min = min([x for x, y in pixel_positions])
    y_min = min([y for x, y in pixel_positions])

    # we move each pixel
    pixel_positions = [(x - x_min, y - y_min) for x, y in pixel_positions]

    return pixel_positions, img


def get_corresponding_points_and_count_total_detection_error(original_points, detections):
    tree = KDTree(detected_trees_positons)
    neighbor_dists, neighbor_indices = tree.query(filtered_original_positions)

    corresponding_points = []

    for idx, point in enumerate(original_points):
        x1, y1 = point
        x2, y2 = detections[neighbor_indices[idx]]

        corresponding_points.append([(x1, y1), (x2, y2)])

    return corresponding_points, sum(neighbor_dists)


def add_points_to_img(img, points):
    img = img.copy()

    for x, y in points:
        color = list(np.random.choice(range(256), size=3))
        img[x:x + DOT_SIZE, y: y + DOT_SIZE] = color

    return img


if __name__ == '__main__':
    path = "/home/piotr/Downloads/Swiebodzin 04 1800m las jednolity/"

    rgb_path = os.path.join(path, "Swiebodzin_04_wycinek.tif")

    shape_path = os.path.join(path, "tree07.shp")

    X_MIN = 2200
    Y_MIN = 2200

    X_MAX = 2800
    Y_MAX = 2800

    pixel_positions, img = get_trees_positions_and_rgb_img(rgb_path, shape_path)

    filtered_original_positions = list(
        filter(lambda p: X_MIN <= p[0] <= X_MAX and Y_MIN <= p[1] <= Y_MAX, pixel_positions))

    filtered_original_positions = [(x - X_MIN, y - Y_MIN) for x, y in filtered_original_positions]

    filtered_original_positions = np.array(filtered_original_positions)

    tree_couter = TreeCounter()

    small_img = img[X_MIN: X_MAX, Y_MIN: Y_MAX, :]

    forest_segentation = ForestSegmentation()

    mask = forest_segentation.mask(small_img)

    counting_dict = tree_couter.count(small_img, mask)
    detected_trees_positons = counting_dict["trees"]

    detected_trees_positons = np.array(detected_trees_positons, dtype=np.int)

    corresponding_points, sum_of_errors = get_corresponding_points_and_count_total_detection_error(
        filtered_original_positions, detected_trees_positons)

    print(f"Detected points: {len(detected_trees_positons)} Original points: {len(filtered_original_positions)}")

    # for original, detected in corresponding_points:
    #     color = list(np.random.choice(range(256), size=3))
    #     x1, y1 = original
    #     x2, y2 = detected
    #     small_img[x1:x1 + DOT_SIZE, y1: y1 + DOT_SIZE] = color
    #     small_img[x2:x2 + DOT_SIZE, y2: y2 + DOT_SIZE] = color

    small_img = add_points_to_img(small_img, detected_trees_positons)

    small_img = cv2.resize(small_img, (1000, 1000))

    cv2.imshow(f"Avg error: {int(sum_of_errors / len(corresponding_points))}", small_img)
    # cv2.imshow("Masked", cv2.bitwise_and(small_img, small_img, mask=mask))
    cv2.waitKey()
    cv2.destroyAllWindows()
