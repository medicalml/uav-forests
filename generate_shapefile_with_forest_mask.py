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
from src.orthophotomap.forest_segmentation import ForestSegmentation, SegMaskToGeometryConverter
from src.utils.image_processing import sliding_window_iterator
from src.utils.shapefile_modifications import update_shapefile
from src.utils.coordinates_converters import convert_geoometry_from_pixel_to_coords

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="generate_shapefile_with_forest_mask.py",
                                     description=("Generate forest mask from geotiff.\nExample command: \n"
                                                  + " " * 4
                                                  + "python3 generate_shapefile_with_forest_mask.py \\\n"
                                                  + " " * 9
                                                  + "  --geotiff file.tiff --shapefile file.shp \\\n"
                                                  + " " * 9
                                                  + "  --target_dir folder_were_I_want_to_store_forest_mask/ \n"),
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--geotiff", required=True,
                        help="path to geotiff file")
    parser.add_argument("--shapefile", required=True,
                        help="path to shapefile annotation file")
    parser.add_argument("--target_dir", required=True,
                        help="directory to store output shape with forest mask")
    parser.add_argument("--start_id", default=0, help="First Area id to mask")
    parser.add_argument("--end_id", default=-1, help="Last Area id to mask")
    parser.add_argument("--index", default="id_ob",
                        help="Parameter table index name")

    args = parser.parse_args()

    if not os.path.exists(args.target_dir):
        os.mkdir(args.target_dir)

    with rio.open(args.geotiff) as geotiff:

        shape_path = args.shapefile

        forest_iterator = ForestIterator(args.geotiff, shape_path,
                                         channels_first=False)
        segmenter = ForestSegmentation()
        converter = SegMaskToGeometryConverter()

        # Write a new Shapefile
        schema = {'geometry': 'Polygon',
                  'properties': {"id": "int"}}

        with fiona.open(os.path.join(args.target_dir, 'forest_mask.shp'), 'w',
                        driver='ESRI Shapefile', schema=schema,
                        crs=forest_iterator.shapes_handler.crs) as output_shapefile:

            if 0 <= int(args.end_id) < len(forest_iterator):
                end_id = args.end_id
            else:
                end_id = len(forest_iterator)

            edit_initial_shape = []
            for i in tqdm(range(int(args.start_id), int(end_id))):
                patch = forest_iterator[i]

                if patch is None:
                    continue

                forest_img = patch['rgb']
                alpha = patch['alpha_channel']

                mask = segmenter.mask(forest_img)
                mask_polygon = convert_geoometry_from_pixel_to_coords(geotiff, converter.convert(mask),
                                                                      patch["row_min"], patch["col_min"])

                output_shapefile.write({'geometry': mapping(mask_polygon),
                                        'properties': patch["properties"]})
