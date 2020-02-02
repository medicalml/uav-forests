import rasterio as rio
from shapely.geometry import mapping, Polygon
import fiona
import json
import pandas as pd
import re
geotiff = rio.open('/home/h/_drzewaBZBUAS/RGB_szprotawa_transparent_mosaic_group1.tif')

print(geotiff._crs)
#

# Here's an example Shapely geometry


# Define a polygon feature geometry with one attribute
schema = {
    'geometry': 'Polygon',
    'properties': {'id': 'int'},
}

# Write a new Shapefile
with fiona.open('my_shp2.shp', 'w', 'ESRI Shapefile', schema) as c:
    ## If there are multiple geometries, put the "for" loop here
    annotations = pd.read_csv('data_out/annotation.csv')
    preds = open('/home/h/efficientdet/predictions/predictions.json', 'r')
    preds = json.load(preds)
    image_list = annotations['patch_number'].tolist()
    for i, k in enumerate(preds.keys()):
        indexik = image_list.index(int(k[:-4]))
        a = annotations['global_pixel_shape'][indexik]
        x0, y0 = [int(float(s)) for s in re.findall(r'-?\d+\.?\d*', a)][:2]
        final_coors = []
        for x, coor_set in enumerate(preds[k]):
            x_min, y_min, x_max, y_max = coor_set
            coors_to_write = []
            coors_to_write.append(rio.transform.xy(geotiff.transform, x_min+x0, y_min+y0) )
            coors_to_write.append(rio.transform.xy(geotiff.transform, x_max+x0, y_min+y0) )
            coors_to_write.append(rio.transform.xy(geotiff.transform, x_max+x0, y_max+y0) )
            coors_to_write.append(rio.transform.xy(geotiff.transform, x_min+x0, y_max+y0) )
           
            poly = Polygon([coors_to_write[0], coors_to_write[1], coors_to_write[2], coors_to_write[3], coors_to_write[0]])
            c.write({
                'geometry': mapping(poly),
                'properties': {'id': i*10+x},
            })