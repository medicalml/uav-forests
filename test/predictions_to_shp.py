import pickle
import rasterio as rio
from shapely.geometry import mapping, Polygon
import fiona
import json
import pandas as pd
import re
import argparse

predictions = pickle.load(open( "temp/save.p", "rb" ))
print(predictions)
score_threshold = 0.5

parser = argparse.ArgumentParser(prog="test_ml_detection.py",
                                     description=("Get simple prediction from pickle and have result in shp file Example command \n"
                                                  + " "*4
                                                  + "python test_ml_detection.py \\\n"
                                                  + " "*9
                                                  + "--rgb_tif_path=/home/h/_drzewaBZBUAS/RGB_szprotawa_transparent_mosaic_group1.tif"),
                                     formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--rgb_tif_path", required=True, help="path to tiff with rgb map")

args = parser.parse_args()

geotiff = rio.open(args.rgb_tif_path)

print(geotiff._crs)
#

# Here's an example Shapely geometry


# Define a polygon feature geometry with one attribute
schema = {
    'geometry': 'Polygon',
    'properties': {'id': 'int'},
}

# Write a new Shapefile
with fiona.open('temp/predictions.shp', 'w', 'ESRI Shapefile', schema) as c:
    for i, pred in enumerate(predictions):
        
        if pred['score'] > score_threshold:
            coors_to_write = []
            coors_to_write.append(rio.transform.xy(geotiff.transform, pred['row_min'], pred['col_min']) )
            coors_to_write.append(rio.transform.xy(geotiff.transform, pred['row_min'], pred['col_max']) )
            coors_to_write.append(rio.transform.xy(geotiff.transform, pred['row_max'], pred['col_max']) )
            coors_to_write.append(rio.transform.xy(geotiff.transform, pred['row_max'], pred['col_min']) )
            
            poly = Polygon([coors_to_write[0], coors_to_write[1], coors_to_write[2], coors_to_write[3], coors_to_write[0]])
            c.write({
                'geometry': mapping(poly),
                'properties': {'id': i},
            })
