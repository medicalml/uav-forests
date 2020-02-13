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
from src.orthophotomap.forest_segmentation import ForestSegmentation
from src.utils.create_tif_tools import pixel_setter

parser = argparse.ArgumentParser(prog="test_mask.py",
                                     description=("This script create predictions and save results in pickle Example command \n"
                                                  + " "*4
                                                  + "python test_mask.py \\\n"
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
geotiff = rio.open(args.rgb_tif_path)
transform = geotiff.transform
profile = geotiff.profile
mask_out = np.zeros(geotiff.shape[:2])
profile.update(dtype=rio.uint8, count=1, compress='lzw')

l.info("zakończono otwieranie pliku tiff")

forest_segmentator = ForestSegmentation()
iterator = ForestIterator(args.rgb_tif_path, args.forest_shp_path, args.nir_tif_path)

for path in ['output']:
    if not os.path.exists(path):
        os.makedirs(path)

for patch in tqdm.tqdm(iterator, total=len(iterator)):
    if patch is None:
        break
    rgb = patch['rgb']
    rgb = np.moveaxis(rgb, 0, -1)
    mask = forest_segmentator.mask(rgb,  patch['ndvi'])
    mask_out[min(patch['rows']):max(patch['rows']), min(patch['columns']):max(patch['columns'])] = pixel_setter(mask_out[min(patch['rows']):max(patch['rows']), min(patch['columns']):max(patch['columns'])], mask)
    
with rio.open('output/mask.tif', 'w', **profile) as dst:
    dst.transform = transform
    dst.write(mask_out.astype(rio.uint8), 1)
#python test/test_mask.py --rgb_tif_path=/home/h/ML\ dane\ dla\ kola/Swiebodzin/RGB_Swiebodzin.tif --nir_tif_path=/home/h/ML\ dane\ dla\ kola/Swiebodzin/NIR_Swiebodzin.tif --forest_shp_path=/home/h/ML\ dane\ dla\ kola/Swiebodzin/obszar_swiebodzin.shp