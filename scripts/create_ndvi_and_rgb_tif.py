import sys
import os
import argparse
import shutil
import pickle
import logging as l

import rasterio as rio 
import numpy as np
import tqdm
import fiona
from shapely.geometry import Point, mapping, Polygon
import torch

l.basicConfig(level=l.INFO, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) #import parent directory of current module directory
from src.orthophotomap.forest_iterator import ForestIterator
from src.utils.create_tif_tools import pixel_setter


parser = argparse.ArgumentParser(prog="test_ndvi.py",
                                     description=("This script create ndvi and save results as tif \n"
                                                  + " "*4
                                                  + "python test_ndvi.py \\\n"
                                                  + " "*9
                                                  + "--rgb_tif_path=/home/h/_drzewaBZBUAS/RGB_szprotawa_transparent_mosaic_group1.tif"
                                                  + " "*4
                                                  + "--nir_tif_path="
                                                  + " "*4
                                                  + "--forest_shp_path="),
                                    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--rgb_tif_path", required=True, help="path to tiff with rgb map")
parser.add_argument("--nir_tif_path", required=True, help="path to tiff with nir")
parser.add_argument("--forest_shp_path", required=True, help="path to tiff with shp showing where trees are ")

args = parser.parse_args()
for path in ['output']:
    if not os.path.exists(path):
        os.makedirs(path)
geotiff = rio.open(args.rgb_tif_path)
transform = geotiff.transform
profile = geotiff.profile
ndvi_output = np.zeros(geotiff.shape[:2], dtype=np.uint8)
rgb_output = np.zeros((geotiff.shape[0], geotiff.shape[1], 3), dtype=np.uint8)
profile.update(dtype=rio.uint8, count=1, compress='lzw')

l.info("zakończono otwieranie pliku tiff")

iterator = ForestIterator(args.rgb_tif_path, args.forest_shp_path, args.nir_tif_path)


for patch in tqdm.tqdm(iterator, total=len(iterator)):
    if patch is None:
        break
    ndvi = patch['ndvi']*255.0
    ndvi = np.clip(ndvi, 0, 255)
    rgb = patch['rgb']
    rgb = np.moveaxis(rgb, 0, -1)
    #print("SZEJPY", rgb.shape, ndvi.shape)
    rgb_output[min(patch['rows']):max(patch['rows']), min(patch['columns']):max(patch['columns']), :] = pixel_setter(rgb_output[min(patch['rows']):max(patch['rows']), min(patch['columns']):max(patch['columns']), :], rgb.astype(np.uint8) )
    ndvi_output[min(patch['rows']):max(patch['rows']), min(patch['columns']):max(patch['columns'])] = pixel_setter(ndvi_output[min(patch['rows']):max(patch['rows']), min(patch['columns']):max(patch['columns'])], ndvi.astype(np.uint8) )

print("saving ndvi")
with rio.open('output/ndvi.tif', 'w', **profile) as dst:
    dst.transform = transform
    dst.write(ndvi_output.astype(rio.uint8), 1)

profile.update(dtype=rio.uint8, count=3, compress='lzw', bigtiff='YES')
with rio.open('output/rgb.tif', 'w', **profile) as dst:
    dst.transform = transform
    print("saving first rgb channel")
    dst.write(rgb_output[:,:,0].astype(rio.uint8), 1)
    print("saving second rgb channel")
    dst.write(rgb_output[:,:,1].astype(rio.uint8), 2)
    print("saving last rgb channel")
    dst.write(rgb_output[:,:,2].astype(rio.uint8), 3)
print("done")
#python scripts/create_ndvi_and_rgb_tif.py --rgb_tif_path=/home/h/ML\ dane\ dla\ kola/Swiebodzin/RGB_Swiebodzin.tif --nir_tif_path=/home/h/ML\ dane\ dla\ kola/Swiebodzin/NIR_Swiebodzin.tif --forest_shp_path=/home/h/ML\ dane\ dla\ kola/Swiebodzin/obszar_swiebodzin.shp