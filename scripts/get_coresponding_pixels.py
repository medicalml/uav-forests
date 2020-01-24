import rasterio as rio

infile_rgb = r"/media/piotr/KINGSTON/Zagan/RGB_Zagan.tif"
infile_nir = r"/media/piotr/KINGSTON/Zagan/NIR_Zagan.tif"




coordinates = [
    (238598.6793652528, 415716.2769632211), (238602.8562761794, 415715.4415810358),
    (238602.0208939941, 415712.3785130229),
    (238598.4009045243, 415710.7077486523), (238598.6793652528, 415716.2769632211)
]


# Your NxN window
N = 500

def get_pixels_by_coordinates(tif_infile, outfile, coordinates, N):

    # Open the raster
    with rio.open(tif_infile) as tif_handler:

        # Loop through your list of coords
        for i, (lon, lat) in enumerate(coordinates):

            # Get pixel coordinates from map coordinates
            py, px = tif_handler.index(lon, lat)
            print('Pixel Y, X coords: {}, {}'.format(py, px))

            # Build an NxN window
            window = rio.windows.Window(px - N // 2, py - N // 2, N, N)
            print(window)

            # Read the data in the window
            # clip is a nbands * N * N numpy array
            clip = tif_handler.read(window=window)

            # You can then write out a new file
            meta = tif_handler.meta
            meta['width'], meta['height'] = N, N
            meta['transform'] = rio.windows.transform(window, tif_handler.transform)

            with rio.open(outfile.format(i), 'w', **meta) as dst:
                dst.write(clip)


def get_coresponding_pixels(rgb_tif, nir_tif, coordinates, N):
    outfile_rgb = r'{}_rgb.tif'
    outfile_nir = r'{}_nir.tif'

    get_pixels_by_coordinates(tif_infile=rgb_tif, outfile=outfile_rgb, coordinates=coordinates, N=N)
    get_pixels_by_coordinates(tif_infile=nir_tif, outfile=outfile_nir, coordinates=coordinates, N=N)



if __name__ == '__main__':
    get_coresponding_pixels(infile_rgb, infile_nir, [(238598.6793652528, 415716.2769632211)], N=N)