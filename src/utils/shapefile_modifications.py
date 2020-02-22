import os

import geopandas as gpd


def update_shapefile(shape_path, save_path, update_list, new_cols):
    '''
    Update shapefile with new information
    :param shape_path: path to the modified shapefile
    :param save_path: path to save the shapefile
    :param update_list: List of structure [(id_ob, col1, col2,...)]
    :param new_cols: List of added columns names
    :return: Nothing
    '''
    assert len(update_list[0]) == len(new_cols) + 1, \
        "New column names lenght does not match the update list shape"

    gdf = gpd.read_file(shape_path)
    for col in new_cols:
        gdf[col] = 0
    gdf = gdf.set_index("id_ob")
    new_df = gpd.GeoDataFrame(update_list, columns=["id_ob"] + new_cols)
    new_df = new_df.set_index("id_ob")
    gdf.update(new_df)
    gdf.to_file(save_path)
