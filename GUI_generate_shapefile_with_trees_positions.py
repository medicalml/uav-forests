from gooey import GooeyParser

from gooey import Gooey

from generate_shapefile_with_trees_positions import perform_tree_counting

@Gooey
def main():
    parser = GooeyParser(description = "Tree counting app")
    parser.add_argument('geotiff', help="Path to RGG geotiff", widget='FileChooser', )
    parser.add_argument('shapefile', help="Patch to shapefile", widget='FileChooser', )
    parser.add_argument('target_dir', help="Where should result files be saved", widget='DirChooser', )
    parser.add_argument('window_size', help="size of a window on which program should count trees", default=128)
    parser.add_argument(
        "--no_masking",
        default=False,
        action="store_true",
        help="Should Counting mask be suspended"
    )
    parser.add_argument('minimal_size', help="Minimal tree area in px on the rgb image", default=36)
    parser.add_argument('brightness', help="Adjust brightness parameter of counting", default=1)
    parser.add_argument('start_id', help="First Area id to count trees in", default="0")
    parser.add_argument('end_id', help="Last Area id to count trees in", default=-1)
    parser.add_argument('index', help="Parameter table index name", default="id_ob")


    args = parser.parse_args()
    args.start_id = int(args.start_id)

    perform_tree_counting(args)


if __name__ == '__main__':
    main()