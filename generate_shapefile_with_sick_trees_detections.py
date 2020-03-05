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
                                     description=("Detect sick trees on a given tiff file.\nExample command: \n"
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
    parser.add_argument("--target_dir", required=True, help ="directory to store output shape with trees positions")
    parser.add_argument("--config_file", required=True, help="Neural Network configuration")
    parser.add_argument("--weights", required=True, help="Neural Network weights file")
    parser.add_argument("--cpu", dest="device", action="store_true", default=False,
                        help="whether to use the masking capability")
    parser.add_argument("--threshold", nargs='?', required=False, default=0.4, type=float, help="threshold for sick trees detctions")
    parser.add_argument("--no-overlap", dest="overlap", action="store_false", default=True, 
                        help="whether to detect trees on overlapping tiles. Dafault: overlapping enabled. "
                             "Disable to speed up computing by roughly ~20\%. Detection quality may drop.")
    parser.add_argument("--suspend_mask", dest="no_masking", action="store_true", default=False,
                        help="whether to use the masking capability")
    parser.add_argument("--start_id", default=0, help="First Area id to count trees in")
    parser.add_argument("--end_id", default=-1, help="Last Area id to count trees in")


    args = parser.parse_args()




    if args.device:
        device = "cuda"
    else:
        device = "cpu"

    detector = SickTreesDetectron2Detector(args.config_file, args.weights, 
                                           device=device, threshold=args.threshold,
                                           overlap_windows=args.overlap, 
                                           postprocess=True)


    if not os.path.exists(args.target_dir):
        os.mkdir(args.target_dir)

    with rio.open(args.geotiff) as geotiff:
        shape_path = args.shapefile
        schema = {
            'geometry': 'Polygon',
            'properties': {"id": "int",
                           "geometry_area": "float",
                           "top_detection_area": "float",
                           "score": "float",
                           "score_mean": "float",
                           "score_min": "float",
                           "score_max": "float",
                           "score_weighted": "float",
                           "nb_detections": "int"}
        }

        with fiona.open(os.path.join(args.target_dir, 'trees.shp'), 'w', 'ESRI Shapefile', schema) as output_shapefile:
            it = ForestIterator(args.geotiff, shape_path, channels_first=False)

            if int(args.end_id) == -1 or int(args.end_id) > len(it):
                end_id = len(it)

            idx = 0

            for i in tqdm(range(int(args.start_id), int(end_id))):
                patch = it[i]
                if patch is not None:
                    rgb = patch['rgb']


                patch_row_max, patch_col_min = rio.transform.rowcol(it.rgb_tif_handler.transform, patch["x_min"], patch["y_max"])

                res = detector.detect(rgb)

                for detection in res:
                    row_min, col_min = detection["row_min"], detection["col_min"]
                    row_max, col_max = detection["row_max"], detection["col_max"]

                    row_0, col_0 = row_min, col_max
                    row_1, col_1 = row_min, col_min
                    row_2, col_2 = row_max, col_min
                    row_3, col_3 = row_max, col_max

                    row_0, col_0 = row_0 + patch_row_max, col_0 + patch_col_min
                    row_1, col_1 = row_1 + patch_row_max, col_1 + patch_col_min
                    row_2, col_2 = row_2 + patch_row_max, col_2 + patch_col_min
                    row_3, col_3 = row_3 + patch_row_max, col_3 + patch_col_min

                    x_0, y_0 = rio.transform.xy(it.rgb_tif_handler.transform, row_0, col_0)
                    x_1, y_1 = rio.transform.xy(it.rgb_tif_handler.transform, row_1, col_1)
                    x_2, y_2 = rio.transform.xy(it.rgb_tif_handler.transform, row_2, col_2)
                    x_3, y_3 = rio.transform.xy(it.rgb_tif_handler.transform, row_3, col_3)

                    polygon = Polygon([(x_0, y_0), (x_1, y_1), (x_2, y_2), (x_3, y_3), (x_0, y_0)])
                    # print(polygon)
                    # print()

                    output_shapefile.write({
                        'geometry': mapping(polygon),
                        'properties': {'id': idx, **{detection.get(key) 
                                                     for key in schema["properties"] 
                                                     if key != "id"}},
                    })

                    idx += 1
