import argparse
import os

import fiona
import rasterio as rio
import numpy as np
from shapely.geometry import Point, mapping
from tqdm import tqdm
import cv2

from src.counting.classical_tree_counter import TreeCounter
from src.orthophotomap.forest_iterator import ForestIterator
from src.orthophotomap.forest_segmentation import ForestSegmentation
from src.utils.image_processing import sliding_window_iterator
from src.utils.shapefile_modifications import update_shapefile


def perform_tree_counting(args):
    WINDOW_SIZE = int(args.window_size)

    if not os.path.exists(args.target_dir):
        os.mkdir(args.target_dir)

    with rio.open(args.geotiff) as geotiff:
        shape_path = args.shapefile
        schema = {
            'geometry': 'Point',
            'properties': {"id": "int"}
        }
        # Write a new Shapefile
        with fiona.open(os.path.join(args.target_dir, 'trees.shp'), 'w', 'ESRI Shapefile', schema) as output_shapefile:
            # output_shapefile = fiona.open(os.path.join(args.target_dir, 'trees.shp'), 'w', 'ESRI Shapefile', schema)
            tree_couter = TreeCounter(goal=float(args.brightness), threshold=int(args.minimal_size))
            it = ForestIterator(args.geotiff, shape_path, channels_first=False)

            if int(args.end_id) == -1 or int(args.end_id) > len(it):
                end_id = len(it)
            else:
                end_id = args.end_id

            masking_tool = ForestSegmentation()
            edit_initial_shape = []
            iterator = 0
            for i in tqdm(range(int(args.start_id), int(end_id))):
                patch = it[i]
                if patch is not None:
                    rgb = patch['rgb']
                    alpha = patch['alpha_channel']
                    # rgb = np.moveaxis(rgb, 0, -1)

                    forest_img = rgb
                    if not args.no_masking:
                        mask = masking_tool.mask(forest_img)
                    else:
                        mask = np.ones(forest_img.shape[0:2], dtype=np.uint8)

                    number_of_trees = 0
                    for forest_iterator_output, mask_iterator_output, alpha_iterator in \
                            zip(sliding_window_iterator(forest_img, WINDOW_SIZE),
                                sliding_window_iterator(mask, WINDOW_SIZE),
                                sliding_window_iterator(alpha, WINDOW_SIZE)):

                        row_add, col_add, local_forest_img = forest_iterator_output
                        _, _, local_mask = mask_iterator_output
                        _, _, alpha_iterator = alpha_iterator
                        masked = cv2.bitwise_and(local_forest_img, local_forest_img, mask=local_mask)
                        masked_alpha = cv2.bitwise_and(alpha_iterator, alpha_iterator, mask=local_mask)
                        if np.sum(masked_alpha) > 0:
                            counting_dict = tree_couter.count(masked, masked_alpha)
                            trees = counting_dict["trees"]
                            number_of_trees += counting_dict["count"]

                            for idx, (row, col) in enumerate(trees):
                                row_max, col_min = rio.transform.rowcol(it.rgb_tif_handler.transform
                                                                        , patch["x_min"], patch["y_max"])
                                row += row_max + row_add
                                col += col_min + col_add
                                point = Point(rio.transform.xy(it.rgb_tif_handler.transform, row, col))
                                output_shapefile.write({
                                    'geometry': mapping(point),
                                    'properties': {'id': idx},
                                })
                    edit_initial_shape.append((patch["description"][args.index], number_of_trees))

            path, filename = os.path.split(shape_path)
            filename, extenstion = os.path.splitext(filename)
            save_path = os.path.join(args.target_dir, filename + "_updated" + extenstion)
            update_shapefile(shape_path, save_path, edit_initial_shape, ["drzewa"], {"drzewa": "int32"}, args.index)

def main():
    parser = argparse.ArgumentParser(prog="generate_shapefile_with_trees_positions.py",
                                     description=("Count tree on geotiff.\nExample command: \n"
                                                  + " " * 4
                                                  + "python3 generate_shapefile_with_trees_positions.py \\\n"
                                                  + " " * 9
                                                  + "  --geotiff file.tiff --shapefile file.shp \\\n"
                                                  + " " * 9
                                                  + "  --target_dir folder_were_I_want_to_store_trees_positons/ \n"
                                                  + " " * 9
                                                  + "  --window_size 512\n"),
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--geotiff", required=True, help="path to geotiff file")
    parser.add_argument("--shapefile", required=True, help="path to shapefile annotation file")
    parser.add_argument("--target_dir", required=True, help="directory to store output shape with tree positions")
    parser.add_argument("--window_size", default=128, help="size of a window on which program should count trees")
    parser.add_argument("--suspend_mask", dest="no_masking", action="store_true", default=False,
                        help="whether to use the masking capability")
    parser.add_argument("--minimal_size", default=36, help="Minimal tree area in px on the rgb image(default: 36)")
    parser.add_argument("--brightness", default=1, help="Adjust brightness parameter of counting")
    parser.add_argument("--start_id", default=0, help="First Area id to count trees in")
    parser.add_argument("--end_id", default=-1, help="Last Area id to count trees in")
    parser.add_argument("--index", default="id_ob", help="Parameter table index name")

    args = parser.parse_args()
    perform_tree_counting(args)

if __name__ == '__main__':
    main()


