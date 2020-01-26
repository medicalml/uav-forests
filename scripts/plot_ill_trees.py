import fiona
import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio




def initiate_geoms(shp):
    if shp['type'] == 'Polygon':
        return shp['coordinates']
    else:
        return [poly[0] for poly in shp['coordinates']]


def get_ill_trees_coords_only_from_a_given_forest(ill_trees_sha_handler,
                                                  forest_x_min, forest_x_max, forest_y_min, forest_y_max):

    ill_trees_in_forest = []

    for tree in ill_trees_sha_handler:
        local_ill_tree_coords = []

        for x, y in tree["geometry"]["coordinates"][0]:
            if forest_x_min <= x <= forest_x_max and forest_y_min <= y <= forest_y_max:
                local_ill_tree_coords.append((x, y))

        if local_ill_tree_coords:
            ill_trees_in_forest.append(local_ill_tree_coords)

    return ill_trees_in_forest

def plot_ill_trees(specyfic_forest, ill_trees_shp_handler):

    forest_coords = initiate_geoms(specyfic_forest['geometry'])

    x_forest_coords = np.array([a[0] for poly in forest_coords for a in poly])
    y_forest_coords = np.array([a[1] for poly in forest_coords for a in poly])

    x_max = x_forest_coords.max()
    y_max = y_forest_coords.max()

    x_min = x_forest_coords.min()
    y_min = y_forest_coords.min()


    ill_trees_in_forest = get_ill_trees_coords_only_from_a_given_forest(ill_trees_shp_handler, x_min, x_max, y_min, y_max)

    with rasterio.open(os.path.join(path, 'RGB_' + name + '.tif')) as tif_handler:

        row_start, col_start = tif_handler.index(x_min, y_min)
        row_stop, col_stop = tif_handler.index(x_max, y_max)

        win = rasterio.windows.Window.from_slices((row_stop, row_start), (col_start, col_stop))
        img = np.stack([tif_handler.read(i + 1, window=win) for i in range(3)])
        print(win)
        img = np.moveaxis(np.stack([tif_handler.read(i + 1, window=win) for i in range(3)]), 0, -1)

        plt.imshow(img)

        for local_ill_tree in ill_trees_in_forest:

            positions_y = []
            positions_x = []

            for x, y in local_ill_tree:
                py, px = tif_handler.index(x, y)
                positions_y.append(py)
                positions_x.append(px)

            positions_y = np.array(positions_y)
            positions_x = np.array(positions_x)

            positions_y = positions_y - row_stop
            positions_x = positions_x - col_start

            plt.plot(positions_x, positions_y, color="red")

        plt.show()



if __name__ == '__main__':
    name = 'Swiebodzin'
    path = "/media/piotr/824F-8A2A/Swiebodzin/"

    I = 163

    forest_shp_handler = fiona.open(os.path.join(path,'obszar_'+name.lower()+'.shp'))

    ill_trees_shp_handler = fiona.open(os.path.join(path,'drzewa_'+name.lower()+'.shp'))
    specyfic_forest = forest_shp_handler[I]

    plot_ill_trees(specyfic_forest, ill_trees_shp_handler)
