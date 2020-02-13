import subprocess
import geopandas
import pickle
import numpy as np
import os
import shutil

multi_folder = 'data_out'

tile_size = 256
step = 256
common_arg = ['python3', 'scripts/orthophotomap_ir_to_patches_dataset.py' ,'--geotiff', '/home/h/ML\ dane\ dla\ kola/RGB_Swiebodzin.tif', '--nirtiff', '/home/h/ML\ dane\ dla\ kola/NIR_Swiebodzin.tif' ,'--shapefile', '/home/h/ML\ dane\ dla\ kola/obszar_swiebodzin.shp', '--tile-size='+str(tile_size), '--step='+str(step), '--verbose', '--single-process']

multi_proc = ['--target-dir='+multi_folder]

subprocess.call(common_arg + multi_proc)