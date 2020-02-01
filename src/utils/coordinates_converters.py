import numpy as np
import rasterio as rio
import rasterio.windows


def coordinates_to_window(rio_handler, x_min, y_min, x_max, y_max):
    row_1, col_1 = rio_handler.index(x_min, y_min)
    row_2, col_2 = rio_handler.index(x_max, y_max)
    return rio.windows.Window(col_off=min(col_1, col_2),
                              row_off=min(row_1, row_2),
                              width=np.abs(col_1 - col_2),
                              height=np.abs(row_1 - row_2))
