import subprocess
import geopandas
import pickle
import numpy as np
import os
import shutil

delete_dirs_after_end = True
multi_folder = 'multi_folder'
single_folder = 'single_folder'
tile_size = 256
step = 256
common_arg = ['python3', 'scripts/ortophotomap_to_patches_dataset_multiprocess.py' ,'--geotiff', '/home/h/_drzewaBZBUAS/RGB_szprotawa_transparent_mosaic_group1.tif' ,'--shapefile', '/home/h/_drzewaBZBUAS/szprotawa_shp/drzewa_szprotawa.shp', '--tile-size='+str(tile_size), '--step='+str(step), '--verbose']

single_proc = ['--target-dir='+single_folder, '--single-process']
multi_proc = ['--target-dir='+multi_folder]

subprocess.call(common_arg + multi_proc)
subprocess.call(common_arg + single_proc)


single_proc_out = open(single_folder+'/annotation.pkl', 'rb')
multi_proc_out = open(multi_folder+'/annotation.pkl', 'rb')

single_geoframe = pickle.load(single_proc_out)
multi_geoframe = pickle.load(multi_proc_out)

if 'patch_number' in single_geoframe and 'patch_number' in multi_geoframe:
    single_array = np.asarray(single_geoframe['patch_number'])
    multi_array = np.asarray(multi_geoframe['patch_number'])


    if False in np.equal(single_array, multi_array):
        print("Annotation Test NOT PASSED")
    else:
        print("Annotation Test PASSED")
    
    if os.listdir(single_folder) == os.listdir(multi_folder):
        print("Output image test PASSED") 
    else:
        print("Output image test NOT PASSED")

else:
    print("TEST IMPOSSIBLE TO RUN - NO SHAPES DETECTED")



if delete_dirs_after_end: 
    if os.path.exists(single_folder):
        shutil.rmtree(single_folder)
    if os.path.exists(multi_folder):
        shutil.rmtree(multi_folder)
