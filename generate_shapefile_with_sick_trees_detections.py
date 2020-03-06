import argparse
import os

import fiona
import rasterio as rio
from shapely.geometry import Polygon, mapping, shape
from tqdm import tqdm

from src.detection.ml_detection import SickTreesDetectron2Detector, DetectionsPostProcessor
from src.orthophotomap.forest_iterator import ForestIterator
from src.utils.coordinates_converters import convert_geoometry_from_pixel_to_coords


def perform_sick_tree_detection(args):
    if args.device:
        device = "cuda"
    else:
        device = "cpu"

    detector = SickTreesDetectron2Detector(args.config_file,
                                           args.weights, device=device, threshold=args.threshold)

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

        forest_iterator = ForestIterator(args.geotiff, shape_path,
                                         channels_first=False)
        detector = SickTreesDetectron2Detector(args.config_file, args.weights,
                                               device=device, threshold=args.threshold,
                                               overlap_windows=args.overlap)
        postprocessor = DetectionsPostProcessor()

        if int(args.end_id) == -1 or int(args.end_id) > len(forest_iterator):
            end_id = len(forest_iterator)

        with fiona.open(os.path.join(args.target_dir, 'trees.shp'), 'w',
                        driver='ESRI Shapefile', schema=schema,
                        crs=forest_iterator.shapes_handler.crs) as output_shapefile:

            idx = 0

            for i in tqdm(range(int(args.start_id), int(end_id))):
                patch = forest_iterator[i]
                if patch is not None:
                    rgb = patch['rgb']

                patch_row_max, patch_col_min = rio.transform.rowcol(it.rgb_tif_handler.transform, patch["x_min"],
                                                                    patch["y_max"])

                res = detector.detect(rgb)

                    polygon = convert_geoometry_from_pixel_to_coords(
                        forest_iterator.rgb_tif_handler, detection["geometry"],
                        row_offset=patch["row_min"], col_offset=patch["col_min"])

                    final_polygon = (polygon & forest_shape)

                    if (final_polygon.area / polygon.area) < 0.5:
                        continue

                    output_shapefile.write(
                        {'geometry': mapping(final_polygon),
                         'properties': {'id': idx,
                                        **{key: detection.get(key)
                                           for key in schema["properties"]
                                           if key != "id"}}
                         })
                    idx += 1

def main():
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
    parser.add_argument("--target_dir", required=True, help="directory to store output shape with tree positions")
    parser.add_argument("--config_file", required=True, help="Neural Netowork configuration")
    parser.add_argument("--weights", required=True, help="Neural Netowork weighs file")
    parser.add_argument("--cpu", dest="device", action="store_true", default=False,
                        help="whether to use the masking capability")
    parser.add_argument("--threshold", nargs='?', required=False, default=0.4, type=float,
                        help="thresold for sick trees detctions")
    parser.add_argument("--suspend_mask", dest="no_masking", action="store_true", default=False,
                        help="whether to use the masking capability")
    parser.add_argument("--start_id", default=0, help="First Area id to count trees in")
    parser.add_argument("--end_id", default=-1, help="Last Area id to count trees in")

    args = parser.parse_args()
    perform_sick_tree_detection(args)




if __name__ == '__main__':
    main()
