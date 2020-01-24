import os
import fiona
import rasterio
import rasterio.mask
import numpy as np
import cv2


def initiate_geoms(shp):
    if shp['type'] == 'Polygon':
        return shp['coordinates']
    else:
        return [poly[0] for poly in shp['coordinates']]

class ForestIterator:
    def __init__(self, rgb_tif_path, nir_tif_path, forest_shp_path):
        self.rgb_path = rgb_tif_path
        self.nir_path = nir_tif_path
        self.shape_path = forest_shp_path
        self.rgb_tif_handler = rasterio.open(rgb_tif_path)
        self.nir_tif_handler = rasterio.open(nir_tif_path)
        self.shapes_handler = fiona.open(forest_shp_path)
        self.length = len(self.shapes_handler)

    def create_ndvi(self, window):
        red_channel_img = np.moveaxis(np.stack([self.rgb_tif_handler.read(1,window=window)]), 0, -1)
        #red_channel_img = np.expand_dims(rgb_img[:, :, 0], axis=-1)
        normalized_red_channel_img = red_channel_img / np.max(red_channel_img)
        print(normalized_red_channel_img.shape)
        nir_img = np.moveaxis(np.stack([self.nir_tif_handler.read(1, window=window)]), 0, -1)
        print(nir_img.shape)
        ndvi = (nir_img - normalized_red_channel_img) / (normalized_red_channel_img + nir_img)
        return ndvi

    def __getitem__(self, item):
        single_shape = self.shapes_handler[item]
        shp = initiate_geoms(single_shape['geometry'])
        x = np.array([a[0] for poly in shp for a in poly])
        y = np.array([a[1] for poly in shp for a in poly])
        row_start, col_start = self.rgb_tif_handler.index(x.min(), y.min())
        row_stop, col_stop = self.rgb_tif_handler.index(x.max(), y.max())
        win = rasterio.windows.Window.from_slices((row_stop, row_start), (col_start, col_stop))
        img = np.stack([self.rgb_tif_handler.read(i + 1, window=win) for i in range(3)])
        mask = np.zeros(img.shape[1:], dtype=np.uint8)
        for poly in shp:
            joint = [self.rgb_tif_handler.index(l[0], l[1]) for l in poly]
            joint = np.array([[[l[1] - col_start, l[0] - row_stop] for l in joint]], dtype=np.int32)
            cv2.fillPoly(mask, pts=[joint], color=255)
        masked = cv2.bitwise_and(img, img, mask=np.stack([mask, mask, mask]))
        ndvi = self.create_ndvi(win)
        print(ndvi.shape, mask.shape)
        masked_ndvi = cv2.bitwise_and(ndvi, ndvi, mask=mask)
        return {'rgb':masked,
                'nir':masked_ndvi,
                'descr':single_shape['properties']}

    def __len__(self):
        return self.length






if __name__ == '__main__':
    name = 'Swiebodzin'
    path = os.path.join('D:', '_drzewaBZBUAS')
    shapes = fiona.open(os.path.join(path, 'obszar_' + name.lower() + '.shp'))