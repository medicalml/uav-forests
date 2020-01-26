import fiona
import rasterio
import rasterio.mask
import numpy as np
import cv2
import os


def initiate_geoms(shp):
    if shp['type'] == 'Polygon':
        return shp['coordinates']
    else:
        return [poly[0] for poly in shp['coordinates']]


class ForestIterator:
    def __init__(self, rgb_tif_path, forest_shp_path, nir_tif_path = None, ill_trees_shp_path=None):

        self.rgb_path = rgb_tif_path
        self.nir_path = nir_tif_path
        self.forest_shp_path = forest_shp_path
        self.tree_shp_path = ill_trees_shp_path
        self.rgb_tif_handler = rasterio.open(rgb_tif_path)

        if self.nir_path is not None:
            self.nir_tif_handler = rasterio.open(nir_tif_path)


        self.forest_shp_handler = fiona.open(forest_shp_path)

        self.ill_trees_shp_handler = fiona.open(ill_trees_shp_path)


        self.length = len(self.forest_shp_handler)

    def create_ndvi(self, rows, cols):
        row_start, col_start = self.rgb_tif_handler.index(rows[0], rows[1])
        row_stop, col_stop = self.rgb_tif_handler.index(cols[0], cols[1])
        rgb_win = rasterio.windows.Window.from_slices((row_stop, row_start), (col_start, col_stop))

        row_start, col_start = self.nir_tif_handler.index(rows[0], rows[1])
        row_stop, col_stop = self.nir_tif_handler.index(cols[0], cols[1])
        nir_win = rasterio.windows.Window.from_slices((row_stop, row_start), (col_start, col_stop))

        red_channel_img = np.moveaxis(np.stack([self.rgb_tif_handler.read(1, window=rgb_win)]), 0, -1)

        normalized_red_channel_img = red_channel_img / np.max(red_channel_img)
        nir_img = np.moveaxis(np.stack([self.nir_tif_handler.read(1, window=nir_win,
                                                                  out_shape=(1,
                                                                             int(normalized_red_channel_img.shape[0]),
                                                                             int(normalized_red_channel_img.shape[1]))
                                                                  )]), 0, -1)
        ndvi = (nir_img - normalized_red_channel_img) / (normalized_red_channel_img + nir_img)
        ndvi = (ndvi + 1)/2
        ndvi = np.float32(ndvi)
        return ndvi

    def __getitem__(self, item):
        single_shape = self.forest_shp_handler[item]
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

        if self.nir_path is not None:
            ndvi = self.create_ndvi((x.min(), y.min()), (x.max(), y.max()))
            masked_ndvi = cv2.bitwise_and(ndvi, ndvi, mask=np.stack(mask))
            return {'rgb': masked,
                    'ndvi': masked_ndvi,
                    'description': single_shape['properties']}
        else:
            return {
                'rgb': masked,
                'description': single_shape['properties']
            }

    def __len__(self):
        return self.length


if __name__ == '__main__':
    name = 'Swiebodzin'
    path = "/media/piotr/824F-8A2A/Swiebodzin/"

    forest_shp_file = os.path.join(path, 'obszar_' + name.lower() + '.shp')

    ill_trees_shp_file = os.path.join(path, 'drzewa_' + name.lower() + '.shp')

    rgb_path = os.path.join(path, 'RGB_' + name + '.tif')
    nir_path = os.path.join(path, 'NIR_' + name + '.tif')

    it = ForestIterator(rgb_path, forest_shp_file, nir_path, ill_trees_shp_file)

    data = it[163]

    print(data)