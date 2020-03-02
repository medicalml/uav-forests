import numpy as np
import cv2
import rasterio as rio
import rasterio.windows

import shapely as shp
from shapely.geometry import Polygon, MultiPolygon
from typing import Union


def coordinates_to_window(rio_handler, x_min, y_min, x_max, y_max,
                          lr_margin=0, tb_margin=0):
    row_1, col_1 = rio_handler.index(x_min, y_min)
    row_2, col_2 = rio_handler.index(x_max, y_max)
    return rio.windows.Window(col_off=min(col_1, col_2) - lr_margin,
                              row_off=min(row_1, row_2) - tb_margin,
                              width=np.abs(col_1 - col_2) + lr_margin,
                              height=np.abs(row_1 - row_2) + tb_margin)


def convert_geoometry_from_pixel_to_coords(rio_handler: rio.DatasetReader,
                                           geometry: Union[Polygon, MultiPolygon],
                                           row_offset=0, col_offset=0):
    """
    Convert shapely geometry represented in pixel coordinates to geographic
    coordinates in a reference sysem used in a given rasterio file handler.
    :param rio_handler: Rasterion geotiff handler
    :param geometry: Shapely geometry - Polygon or MultiPolygon
    :return: Shapely geometry - Polygon or MultiPolygon
    """
    def _convert_pixel(x, y):
        return rio.transform.xy(rio_handler.transform,
                                row_offset + y, col_offset + x)

    return shp.ops.transform(_convert_pixel, geometry)
