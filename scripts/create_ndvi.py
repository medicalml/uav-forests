import rasterio
import numpy as np
from PIL import Image

INFILE_NIR = r"0_nir.tif"

INFILE_RGB  = r"0_rgb.tif"


def create_ndvi_tif(rgb_tif_name, nir_tif_name, output_file_name):

    with rasterio.open(rgb_tif_name) as rgb_tif_handler:
        rgb_img = np.moveaxis(np.stack([rgb_tif_handler.read(i + 1) for i in range(3)]), 0, -1)

        print(f"Rgb img shape: {rgb_img.shape}")

        red_channel_img = np.expand_dims(rgb_img[:, :, 0], axis=-1)

        print(f"Rgb img shape: {red_channel_img.shape}")

        normalized_red_channel_img = red_channel_img / np.max(red_channel_img)



    with rasterio.open(nir_tif_name) as nir_tif_handler:
        nir_img = np.moveaxis(np.stack([nir_tif_handler.read(1)]), 0, -1)

        print(f"Rgb img shape: {nir_img.shape}")

    ndvi = (nir_img - normalized_red_channel_img )/(normalized_red_channel_img + nir_img)

    np.save('ndvi.npy', ndvi)  # save




if __name__ == '__main__':
    create_ndvi_tif(INFILE_RGB, INFILE_NIR, "zjj.jpg")