import subprocess

multi_folder = 'data_out'

tile_size = 256
step = 256
common_arg = ['python3', 'scripts/orthophotomap_to_rgb_ndvi_patches_dataset.py' ,'--geotiff', 
'/home/h/ML dane dla kola/Szprotawa/RGB_Szprotawa.tif', '--nirtiff',
"/home/h/ML dane dla kola/Szprotawa/NIR_Szprotawa.tif" ,'--shapefile', 
'/home/h/ML dane dla kola/Szprotawa/drzewa_szprotawa.shp', 
'--tile-size='+str(tile_size), '--step='+str(step), '--verbose']

multi_proc = ['--target-dir='+multi_folder]

subprocess.call(common_arg + multi_proc)