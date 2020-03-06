import io

from gooey import GooeyParser
from gooey import Gooey

import os

import fiona
import rasterio as rio
from shapely.geometry import Polygon, mapping, shape
from tqdm import tqdm

from src.detection.ml_detection import SickTreesDetectron2Detector, DetectionsPostProcessor
from src.orthophotomap.forest_iterator import ForestIterator
from src.utils.coordinates_converters import convert_geoometry_from_pixel_to_coords



@Gooey
def main():

    default_weights_path = os.path.join(os.getcwd(), "model", "model_weights.pth")
    default_config_path = os.path.join(os.getcwd(), "model", "config.yml")

    parser = GooeyParser(description="Sick tree detection app")

    f = io.StringIO()


    parser.add_argument('geotiff', help="Path to RGB geotiff", widget='FileChooser', )
    parser.add_argument('shapefile', help="Patch to shapefile", widget='FileChooser', )
    parser.add_argument('target_dir', help="Where should result files be saved", widget='DirChooser', )
    parser.add_argument("--config_file", widget='FileChooser', help="Neural Netowork configuration", default=default_config_path)
    parser.add_argument("--weights", widget='FileChooser', help="Neural Netowork weighs file", default=default_weights_path)
    parser.add_argument("--CPU", dest="device", action="store_true", default=False,
                        help="Should programme use CPU instead of GPU")
    parser.add_argument("--threshold", default=0.4, type=float,
                        help="thresold for sick trees detctions")
    parser.add_argument(
        "--no_masking",
        default=False,
        action="store_true",
        help="Should Counting mask be suspended"
    )
    parser.add_argument("--overlap", action="store_false", default=True,
                        help="Should overlap be turned on")
                        # help="whether to detect trees on overlapping tiles")

    parser.add_argument('--start_id', help="First Area id to count trees in", default="0")
    parser.add_argument('--end_id', help="Last Area id to count trees in", default=-1)
    # parser.add_argument('index', help="Parameter table index name", default="id_ob")

    args = parser.parse_args()

    args.start_id = int(args.start_id)
    if not args.device:
        device = "cuda"
    else:
        device = "cpu"

    print(args.device)
    print(device)

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

        else:
            end_id = args.end_id

        with fiona.open(os.path.join(args.target_dir, 'trees.shp'), 'w',
                        driver='ESRI Shapefile', schema=schema,
                        crs=forest_iterator.shapes_handler.crs) as output_shapefile:

            idx = 0

            for i in tqdm(range(int(args.start_id), int(end_id)), ncols=50):
                prog = f.getvalue().split('\r ')[-1].strip()
                print(prog)

                patch = forest_iterator[i]
                if patch is not None:
                    rgb = patch['rgb']

                detections = detector.detect(rgb)
                refined_detections = postprocessor(detections)

                forest_shape = shape(
                    forest_iterator.shapes_handler[i]['geometry'])

                for detection in refined_detections:

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


if __name__ == '__main__':
    main()
