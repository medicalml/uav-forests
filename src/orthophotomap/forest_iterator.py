import fiona
import rasterio as rio
import rasterio.mask
import rasterio.plot
import rasterio.features
import numpy as np
import cv2
from shapely.geometry import Point
import geopandas as gpd
import os

from src.utils import infrared
from src.utils.coordinates_converters import coordinates_to_window


class ForestIterator:

    def __init__(self, rgb_tif_path: str, forest_shp_path: str, nir_tif_path: str = None,
                 alpha_channel: bool = False, channels_first: bool = True,
                 lr_margin: int = 0, tb_margin: int = 0,
                 apply_mask: bool = True):
        '''
        :param rgb_tif_path: Path to the RGB.tif
        :param forest_shp_path: Path to the shapefile with forests .shp. Need to have "id_ob" column in the properties
        :param nir_tif_path:  Path to the NIR.tif
        :param alpha_channel: If alpha channel is in the rgb
        :param channels_first:
        '''
        self.rgb_path = rgb_tif_path
        self.nir_path = nir_tif_path
        self.shape_path = forest_shp_path
        self.alpha_channel = alpha_channel
        self.channels_first = channels_first
        self.rgb_tif_handler = rio.open(rgb_tif_path)
        if self.nir_path is not None:
            self.nir_tif_handler = rio.open(nir_tif_path)
        self.shapes_handler = fiona.open(forest_shp_path)
        self.length = len(self.shapes_handler)
        self.lr_margin = lr_margin
        self.tb_margin = tb_margin
        self.apply_mask = apply_mask

    def initiate_geoms(self, shp_geometry: dict):
        '''
        Unpack the basic geometry sequence
        :param shp_geometry: Fiona shape geometry dictionary
        :return:
        '''
        if shp_geometry['type'] == 'Polygon':
            return shp_geometry['coordinates']
        else:
            return [poly[0] for poly in shp_geometry['coordinates']]

    def create_ndvi(self, x_min, y_min, x_max, y_max):
        '''
        Create the ndvi sequence for window of coorinates
        :param x_min:
        :param y_min:
        :param x_max:
        :param y_max:
        :return: NDVI numpy array of shape matching the rgb image
        '''
        rgb_win = coordinates_to_window(self.rgb_tif_handler,
                                        x_min - self.lr_margin, y_min - self.tb_margin,
                                        x_max + self.lr_margin, y_max + self.tb_margin)

        nir_win = coordinates_to_window(self.nir_tif_handler,
                                        x_min - self.lr_margin, y_min - self.tb_margin,
                                        x_max + self.lr_margin, y_max + self.tb_margin)

        red_channel_img = self.rgb_tif_handler.read(1, window=rgb_win)

        nir_img = self.nir_tif_handler.read(1, window=nir_win,
                                            out_shape=red_channel_img.shape)

        ndvi = infrared.nir_to_ndvi(nir_img, red_channel_img)
        return ndvi

    def __getitem__(self, item):
        single_shape = self.shapes_handler[item]

        if single_shape is None:
            return None

        shp = self.initiate_geoms(single_shape['geometry'])
        x = np.asarray([point[0] for poly in shp for point in poly])
        y = np.asarray([point[1] for poly in shp for point in poly])

        win = coordinates_to_window(self.rgb_tif_handler,
                                    x.min() - self.lr_margin, y.min() - self.tb_margin,
                                    x.max() + self.lr_margin, y.max() + self.tb_margin)

        bands = [1, 2, 3]

        # if self.alpha_channel:
        #     bands.append(4)

        img = rio.plot.reshape_as_image(
            self.rgb_tif_handler.read(bands, window=win))

        alpha_channel = rio.plot.reshape_as_image(
            self.rgb_tif_handler.read([4], window=win))

        mask = self.build_mask(img, single_shape['geometry'], win,
                               col_offset=win.col_off,
                               row_offset=win.row_off)

        masked = cv2.bitwise_and(img, img, mask=mask)

        masked_alpha_channel = cv2.bitwise_and(
            alpha_channel, alpha_channel, mask=mask)

        if self.channels_first:
            masked = rio.plot.reshape_as_raster(masked)

        row_min, col_min = rio.transform.rowcol(
            self.rgb_tif_handler.transform, x.min(), y.max())

        row_max, col_max = rio.transform.rowcol(
            self.rgb_tif_handler.transform, x.max(), y.min())

        result = {"rgb": masked,
                  "description": single_shape['properties'],
                  "alpha_channel": masked_alpha_channel,
                  "mask": mask,
                  "x_min": x.min(),
                  "x_max": x.max(),
                  "y_min": y.min(),
                  "y_max": y.max(),
                  "row_min": row_min,
                  "row_max": row_max,
                  "col_min": col_min,
                  "col_max": col_max,
                  "properties": single_shape["properties"]}

        if self.nir_path is not None:
            ndvi = self.create_ndvi(x.min(), y.min(), x.max(), y.max())
            masked_ndvi = cv2.bitwise_and(ndvi, ndvi, mask=mask)
            result["ndvi"] = masked_ndvi

        return result

    def build_mask(self, img, shape, win, col_offset, row_offset):
        '''
        Build mask from the polygons from geometry
        :param img: numpy image to mask
        :param shapes: shapes and geometries to mask by
        :param col_offset:
        :param row_offset:
        :return: mask for shape of img
        '''
        if not self.apply_mask:
            return np.ones(img.shape[:2], dtype=np.uint8)
        # mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask = rio.features.rasterize([shape], img.shape[:2], default_value=255, fill=0, dtype=np.uint8,
                                      transform=rio.windows.transform(win, self.rgb_tif_handler.transform))
        return mask

    def __len__(self):
        return self.length
