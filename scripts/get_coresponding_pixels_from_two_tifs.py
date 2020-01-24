import rasterio as rio

INFILE_RGB = r"/media/piotr/824F-8A2A/Swiebodzin/RGB_Swiebodzin.tif"
INFILE_NIR = r"/media/piotr/824F-8A2A/Swiebodzin/NIR_Swiebodzin.tif"

# INFILE_RGB = r"/media/piotr/824F-8A2A/Lubsko/RGB_lubsko.tif"
# INFILE_NIR = r"helper.tif"

# INFILE_RGB = r"/media/piotr/824F-8A2A/Zagan/RGB_Zagan.tif"
# INFILE_NIR = r"/media/piotr/824F-8A2A/Zagan/NIR_Zagan.tif"




coordinates = [
    (238598.6793652528, 415716.2769632211), (238602.8562761794, 415715.4415810358),
    (238602.0208939941, 415712.3785130229),
    (238598.4009045243, 415710.7077486523), (238598.6793652528, 415716.2769632211)
]


# Your NxN window
N = 500



def get_pixels_by_coordinates(tif_infile, outfile, top_left_coordinates, bottom_right_coordinates):
    with rio.open(tif_infile) as tif_handler:

        lon, lat = top_left_coordinates
        top_py, right_px = tif_handler.index(lon, lat)

        lon, lat = bottom_right_coordinates
        bottom_py, left_px = tif_handler.index(lon, lat)

        print(f"Top left: y={top_py}, x={left_px}")
        print(f"Bottom right: y={bottom_py}, x={right_px}")

        window = rio.windows.Window.from_slices((bottom_py, top_py), (right_px, left_px))

        print(window)
        if window.height == 0 or window.width == 0:
            raise ValueError("Window width or height equals 0")


        # Read the data in the window
        # clip is a nbands * N * N numpy array
        clip = tif_handler.read(window=window)

        # You can then write out a new file
        meta = tif_handler.meta
        meta['width'], meta['height'] = N, N
        meta['transform'] = rio.windows.transform(window, tif_handler.transform)

        with rio.open(outfile, 'w', **meta) as dst:
            dst.write(clip)





def get_coresponding_pixels(rgb_tif, nir_tif, coordinates):
    outfile_rgb = r'0_rgb.tif'
    outfile_nir = r'0_nir.tif'

    get_pixels_by_coordinates(tif_infile=rgb_tif, outfile=outfile_rgb, top_left_coordinates=coordinates[0],
                              bottom_right_coordinates=coordinates[1])

    get_pixels_by_coordinates(tif_infile=nir_tif, outfile=outfile_nir,
                              top_left_coordinates=coordinates[0], bottom_right_coordinates=coordinates[1])




if __name__ == '__main__':
    coordinates = [(220492.4102842412, 438505.3498667013)]

    coordinates = [(246346.639, 493755.314), (247102.011, 494537.954)]

    get_coresponding_pixels(INFILE_RGB, INFILE_NIR, coordinates)