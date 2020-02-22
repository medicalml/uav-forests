import argparse
import os

import fiona
import numpy as np
import rasterio as rio
from shapely.geometry import Point, mapping
from tqdm import tqdm

from src.counting.classical_tree_counter import TreeCounter
from src.orthophotomap.forest_iterator import ForestIterator
from src.orthophotomap.forest_segmentation import ForestSegmentation
from src.utils.shapefile_modifications import update_shapefile

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="generate_shapefile_with_trees_positions.py",
                                     description=("Count tree on geotiff.\nExample command: \n"
                                                  + " " * 4
                                                  + "python3 generate_shapefile_with_trees_positions.py \\\n"
                                                  + " " * 9
                                                  + "  --geotiff file.tiff --shapefile file.shp \\\n"
                                                  + " " * 9
                                                  + "  --target_dir folder_were_I_want_to_save_trees_positons/ \n"
                                                  + " " * 9
                                                  + "  --update_shapefile True\n"),
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--geotiff", required=True, help="path to geotiff file")
    parser.add_argument("--shapefile", required=True, help="path to shapefile annotation file")
    parser.add_argument("--target_dir", required=True, help="directory to store output shape with tree positions")

    args = parser.parse_args()
    # iterator = 0
    with rio.open(args.geotiff) as geotiff:
        shape_path = args.shapefile
        schema = {
            'geometry': 'Point',
            'properties': {"id": "int"}
        }
        # Write a new Shapefile
        with fiona.open(os.path.join(args.target_dir, 'trees.shp'), 'w', 'ESRI Shapefile', schema) as output_shapefile:
            # output_shapefile = fiona.open(os.path.join(args.target_dir, 'trees.shp'), 'w', 'ESRI Shapefile', schema)
            tree_couter = TreeCounter()
            it = ForestIterator(args.geotiff, shape_path)
            masking_tool = ForestSegmentation()
            edit_initial_shape = []

            for patch in tqdm(it):
                rgb = patch['rgb']
                rgb = np.moveaxis(rgb, 0, -1)
                forest_img = rgb

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

            path, filename = os.path.split(shape_path)
            filename, extenstion = os.path.splitext(filename)
            save_path = os.path.join(args.target_dir, filename+"_updated"+extenstion)
            update_shapefile(shape_path, save_path, edit_initial_shape, ["drzewa"])
                # if iterator > 6:
                #     break
                # iterator += 1
