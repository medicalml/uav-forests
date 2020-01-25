import fiona
import rasterio as rio
import rasterio.mask
import numpy as np
import cv2
from src.utils import infrared


class ForestIterator:

    def __init__(self, rgb_tif_path, forest_shp_path, nir_tif_path=None,
                 alpha_channel=False, channels_first=True):
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

    def initiate_geoms(self, shp_geometry: dict):  # fiona shape geometry dict
        if shp_geometry['type'] == 'Polygon':
            return shp_geometry['coordinates']
        else:
            return [poly[0] for poly in shp_geometry['coordinates']]

    def create_ndvi(self, rows, cols):
        row_start, col_start = self.rgb_tif_handler.index(rows[0], rows[1])
        row_stop, col_stop = self.rgb_tif_handler.index(cols[0], cols[1])
        rgb_win = rio.windows.Window.from_slices(
            (row_stop, row_start), (col_start, col_stop))

        row_start, col_start = self.nir_tif_handler.index(rows[0], rows[1])
        row_stop, col_stop = self.nir_tif_handler.index(cols[0], cols[1])
        nir_win = rio.windows.Window.from_slices((row_stop, row_start),
                                                 (col_start, col_stop))

        red_channel_img = self.rgb_tif_handler.read(
            1, window=rgb_win)[..., np.newaxis]

        nir_img = self.nir_tif_handler.read(
            1, out_shape=(1, *red_channel_img.shape[:2]),
            window=nir_win)[..., np.newaxis]

        ndvi = infrared.nir_to_ndvi(nir_img, red_channel_img)
        return ndvi

    def __getitem__(self, item):
        single_shape = self.shapes_handler[item]
        shp = self.initiate_geoms(single_shape['geometry'])
        x = np.array([a[0] for poly in shp for a in poly])
        y = np.array([a[1] for poly in shp for a in poly])
        row_start, col_start = self.rgb_tif_handler.index(x.min(), y.min())
        row_stop, col_stop = self.rgb_tif_handler.index(x.max(), y.max())

        win = rio.windows.Window.from_slices(
            (row_stop, row_start), (col_start, col_stop))
        img = np.stack([self.rgb_tif_handler.read(i + 1, window=win)
                        for i in range(3 + self.alpha_channel)],
                       axis=-1)
        mask = self.build_mask(img, shp,
                               col_offset=col_start,
                               row_offset=row_stop)

        masked = cv2.bitwise_and(img, img, mask=mask)

        if self.channels_first:
            masked = masked.transpose(2, 0, 1)

        result = {'rgb': masked,
                  'description': single_shape['properties']}

        if self.nir_path is not None:
            ndvi = self.create_ndvi((x.min(), y.min()), (x.max(), y.max()))
            masked_ndvi = cv2.bitwise_and(ndvi, ndvi, mask=mask)
            result["ndvi"] = masked_ndvi

        return result

    def build_mask(self, img, shapes, col_offset, row_offset):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        for poly in shapes:
            joint = [self.rgb_tif_handler.index(l[0], l[1]) for l in poly]
            joint = np.array([[[l[1] - col_offset, l[0] - row_offset]
                               for l in joint]], dtype=np.int32)
            cv2.fillPoly(mask, pts=[joint], color=255)
        return mask

    def __len__(self):
        return self.length
