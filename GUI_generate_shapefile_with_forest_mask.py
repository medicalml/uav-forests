import io

from gooey import GooeyParser, Gooey

import os

import fiona
import rasterio as rio
from shapely.geometry import Point, mapping
from tqdm import tqdm

from src.orthophotomap.forest_iterator import ForestIterator
from src.orthophotomap.forest_segmentation import ForestSegmentation, SegMaskToGeometryConverter
from src.utils.coordinates_converters import convert_geoometry_from_pixel_to_coords


@Gooey
def main():
    parser = GooeyParser(description = "Mask detection app")

    f = io.StringIO()

    parser.add_argument('geotiff', help="Path to RGB geotiff", widget='FileChooser', )
    parser.add_argument('shapefile', help="Patch to shapefile", widget='FileChooser', )
    parser.add_argument('target_dir', help="Where should result files be saved", widget='DirChooser', )
    parser.add_argument('--start_id', help="First Area id to count trees in", default="0")
    parser.add_argument('--end_id', help="Last Area id to count trees in", default=-1)
    parser.add_argument('index', help="Parameter table index name", default="id_ob")

    args = parser.parse_args()
    args.start_id = int(args.start_id)

    if not os.path.exists(args.target_dir):
        os.mkdir(args.target_dir)

    with rio.open(args.geotiff) as geotiff:

        shape_path = args.shapefile

        forest_iterator = ForestIterator(args.geotiff, shape_path,
                                         channels_first=False)
        segmenter = ForestSegmentation()
        converter = SegMaskToGeometryConverter()

        # Write a new Shapefile
        schema = forest_iterator.shapes_handler.schema

        with fiona.open(os.path.join(args.target_dir, 'forest_mask.shp'), 'w',
                        driver='ESRI Shapefile', schema=schema,
                        crs=forest_iterator.shapes_handler.crs) as output_shapefile:

            if 0 <= int(args.end_id) < len(forest_iterator):
                end_id = args.end_id
            else:
                end_id = len(forest_iterator)

            edit_initial_shape = []
            for i in tqdm(range(int(args.start_id), int(end_id)), ncols=50):
                prog = f.getvalue().split('\r ')[-1].strip()
                print(prog)

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

if __name__ == '__main__':
    main()