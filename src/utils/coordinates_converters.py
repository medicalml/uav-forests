import rasterio as rio

def coordinates_to_window(tif_handler,x_min, y_min, x_max, y_max):

    with tif_handler:

        window = rio.windows.Window.from_slices((y_min, y_max), (x_max, x_min))


        if window.height == 0 or window.width == 0:
            raise ValueError("Window width or height equals 0")

        return window