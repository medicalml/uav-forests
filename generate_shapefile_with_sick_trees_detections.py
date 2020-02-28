import argparse
import os

import fiona
import rasterio as rio
from shapely.geometry import Polygon, mapping
from tqdm import tqdm

from src.detection.ml_detection import SickTreesDetectron2Detector
from src.orthophotomap.forest_iterator import ForestIterator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="generate_shapefile_with_sick_trees_detections.py",
                                     description=("Detect sick trees on a given shapefile.\nExample command: \n"
                                                  + " " * 4
                                                  + "python3 generate_shapefile_with_sick_trees_detections.py \\\n"
                                                  + " " * 9
                                                  + "  --geotiff file.tiff --shapefile file.shp \\\n"
                                                  + " " * 9
                                                  + "  --target_dir folder_were_I_want_to_store_trees_positons/ \n"
                                                  + " " * 9
                                                  + " --config_file config.yml"
                                                  + " " * 9
                                                  + " --weights_file model_weights.pth"),
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--geotiff", required=True, help="path to geotiff file")
    parser.add_argument("--shapefile", required=True, help="path to shapefile annotation file")
    parser.add_argument("--target_dir", required=True, help ="directory to store output shape with tree positions")
    parser.add_argument("--config_file", required=True, help="Neural Netowork configuration")
    parser.add_argument("--weights", required=True, help="Neural Netowork weighs file")
    parser.add_argument("--cpu", dest="device", action="store_true", default=False,
                        help="whether to use the masking capability")
    # parser.add_argument("--threshold", required=False, nargs='?', const=0.3, type=float, help="thresold for sick trees detctions")

    args = parser.parse_args()

    if args.device:
        device = "cuda"
    else:
        device = "cpu"

    detector = SickTreesDetectron2Detector(args.config_file,
                                           args.weights, device=device)


    if not os.path.exists(args.target_dir):
        os.mkdir(args.target_dir)

    with rio.open(args.geotiff) as geotiff:
        shape_path = args.shapefile
        schema = {
            'geometry': 'Polygon',
            'properties': {"id": "int"}
        }

        with fiona.open(os.path.join(args.target_dir, 'trees.shp'), 'w', 'ESRI Shapefile', schema) as output_shapefile:
            it = ForestIterator(args.geotiff, shape_path, channels_first=False)

            for patch in tqdm(it):
                rgb = patch['rgb']

                row_max, col_min = rio.transform.rowcol(it.rgb_tif_handler.transform, patch["x_min"], patch["y_max"])

                res = detector.detect(rgb)

                for idx, detection in enumerate(res):
                    x_min, y_min = detection["row_min"], detection["col_min"]
                    x_max, y_max = detection["row_max"], detection["col_max"]

                    x0, y0 = x_min, y_min
                    x1, y1 = x_min, y_max
                    x2, y2 = x_max, y_min
                    x3, y3 = x_max, y_max

                    row_0, col_0 = x0 + row_max, y0 + col_min
                    row_1, col_1 = x1 + row_max, y1 + col_min
                    row_2, col_2 = x2 + row_max, y2 + col_min
                    row_3, col_3 = x3 + row_max, y3 + col_min

                    lon_0, lan_0 = rio.transform.xy(it.rgb_tif_handler.transform, row_0, col_0)
                    lon_1, lan_1 = rio.transform.xy(it.rgb_tif_handler.transform, row_1, col_1)
                    lon_2, lan_2 = rio.transform.xy(it.rgb_tif_handler.transform, row_2, col_2)
                    lon_3, lan_3 = rio.transform.xy(it.rgb_tif_handler.transform, row_3, col_3)

                    polygon = Polygon([(lon_0, lan_0), (lon_1, lan_1), (lon_2, lan_2), (lon_3, lan_3), (lon_0, lan_0)])
                    # print(polygon)

                    output_shapefile.write({
                        'geometry': mapping(polygon),
                        'properties': {'id': idx},
                    })