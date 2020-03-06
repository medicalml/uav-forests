from gooey import GooeyParser, Gooey

from generate_shapefile_with_forest_mask import perform_mask_generation


@Gooey
def main():
    parser = GooeyParser(description = "Mask detection app")


    parser.add_argument('geotiff', help="Path to RGG geotiff", widget='FileChooser', )
    parser.add_argument('shapefile', help="Patch to shapefile", widget='FileChooser', )
    parser.add_argument('target_dir', help="Where should result files be saved", widget='DirChooser', )
    parser.add_argument('--start_id', help="First Area id to count trees in", default="0")
    parser.add_argument('--end_id', help="Last Area id to count trees in", default=-1)
    parser.add_argument('index', help="Parameter table index name", default="id_ob")

    args = parser.parse_args()
    args.start_id = int(args.start_id)

    perform_mask_generation(args)

if __name__ == '__main__':
    main()