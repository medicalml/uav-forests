import subprocess
import geopandas
import pickle
import numpy as np
import os
import shutil

multi_folder = 'data_out'

tile_size = 256
step = 256
common_arg = ['python3', 'scripts/orthophotomap_ir_to_patches_dataset.py' ,'--geotiff', '/home/h/_drzewaBZBUAS/RGB_szprotawa_transparent_mosaic_group1.tif' ,'--shapefile', '/home/h/_drzewaBZBUAS/szprotawa_shp/drzewa_szprotawa.shp', '--tile-size='+str(tile_size), '--step='+str(step), '--verbose']

multi_proc = ['--target-dir='+multi_folder]

subprocess.call(common_arg + multi_proc)