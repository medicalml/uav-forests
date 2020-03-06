from gooey import GooeyParser
import os
from gooey import Gooey

from generate_shapefile_with_sick_trees_detections import perform_sick_tree_detection


@Gooey
def main():

    default_weights_path = os.path.join(os.getcwd(), "model", "model_weights.pth")
    default_config_path = os.path.join(os.getcwd(), "model", "config.yml")

    parser = GooeyParser(description = "Sick tree detection app")
    parser.add_argument('geotiff', help="Path to RGG geotiff", widget='FileChooser', )
    parser.add_argument('shapefile', help="Patch to shapefile", widget='FileChooser', )
    parser.add_argument('target_dir', help="Where should result files be saved", widget='DirChooser', )
    parser.add_argument("--config_file", widget='FileChooser', help="Neural Netowork configuration", default=default_config_path)
    parser.add_argument("--weights", widget='FileChooser', help="Neural Netowork weighs file", default=default_weights_path)
    parser.add_argument("--device", dest="CPU", action="store_true", default=False,
                        help="Should programme use CPU instead of GPU")
    parser.add_argument("--threshold", default=0.4, type=float,
                        help="thresold for sick trees detctions")
    parser.add_argument(
        "--no_masking",
        default=False,
        action="store_true",
        help="Should Counting mask be suspended"
    )
    parser.add_argument("--overlap", action="store_true", default=True,
                        help="Should overlap be turned on")
                        # help="whether to detect trees on overlapping tiles")

    parser.add_argument('--start_id', help="First Area id to count trees in", default="0")
    parser.add_argument('--end_id', help="Last Area id to count trees in", default=-1)
    # parser.add_argument('index', help="Parameter table index name", default="id_ob")

    args = parser.parse_args()
    args.start_id = int(args.start_id)
    perform_sick_tree_detection(args)

if __name__ == '__main__':
    main()
