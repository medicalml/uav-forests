import sys
import os
import argparse
import rasterio as rio 
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) #import parent directory of current module directory
from src.detection.ml_detection import SickTreesDetectron2Detector
from src.orthophotomap.forest_iterator import ForestIterator

parser = argparse.ArgumentParser(prog="test_ml_detection.py",
                                     description=("Get simple prediction and have result in shp file Example command \n"
                                                  + " "*4
                                                  + "python test_ml_detection.py \\\n"
                                                  + " "*9
                                                  + "--config_yml_path='/home/username/Pobrane/config.yml"
                                                  + " "*4
                                                  + "--weights_snapshot_path=/home/h/username/Pobrane/model_final.pth"
                                                  + "--rgb_tif_path=/home/h/_drzewaBZBUAS/RGB_szprotawa_transparent_mosaic_group1.tif"
                                                  + "--forest_shp_path=/home/h/_drzewaBZBUAS/szprotawa_shp/drzewa_szprotawa.shp"),
                                     formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--config_yml_path", required=True, help="path to detectron2 configuration file")
parser.add_argument("--weights_snapshot_path", required=True, help="path to detectron2 model")
parser.add_argument("--rgb_tif_path", required=True, help="path to tiff with rgb map")
parser.add_argument("--forest_shp_path", required=True, help="path to ill tree annotations")

args = parser.parse_args()
detector = SickTreesDetectron2Detector(args.config_yml_path, args.weights_snapshot_path)
with rio.open(args.rgb_tif_path) as rgb_tiff:
    rgb_img = rgb_tiff.read()
rgb_img = rgb_img[(0,1,2),:,:]
rgb_img = rgb_img[:::-1]
print(rgb_img.shape)
ndvi_non_existing = np.zeros(rgb_img.shape[:2])
mask_non_existing = np.zeros(rgb_img.shape[:2])

detections = detector.detect(rgb_img, ndvi_non_existing, mask_non_existing)
print(detections)
'''
forest = ForestIterator(args.rgb_tif_path, args.forest_shp_path)
for obj in forest:
    #print(obj.keys())
    #print(obj['description'])
    detector.predictor()
     python test/test_ml_detection.py --config_yml_path=/home/h/Pobrane/config.yml --weights_snapshot_path=/home/h/Pobrane/model_final.pth --rgb_tif_path=/home/h/_drzewaBZBUAS/RGB_szprotawa_transparent_mosaic_group1.tif --forest_shp_path=/home/h/_drzewaBZBUAS/szprotawa_shp/drzewa_szprotawa.shp
'''