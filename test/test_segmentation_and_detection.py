import sys
import os
import argparse
import rasterio as rio 
import numpy as np
import pickle 
import logging as l
l.basicConfig(level=l.INFO, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) #import parent directory of current module directory
from src.detection.ml_detection import SickTreesDetectron2Detector
from src.orthophotomap.forest_iterator import ForestIterator
from src.orthophotomap.forest_segmentation import ForestSegmentation

parser = argparse.ArgumentParser(prog="test_ml_detection.py",
                                     description=("This script create predictions and save results in pickle Example command \n"
                                                  + " "*4
                                                  + "python test_ml_detection.py \\\n"
                                                  + " "*9
                                                  + "--config_yml_path='/home/username/Pobrane/config.yml"
                                                  + " "*4
                                                  + "--weights_snapshot_path=/home/username/Pobrane/model_final.pth"
                                                  + " "*4
                                                  + "--rgb_tif_path=/home/h/_drzewaBZBUAS/RGB_szprotawa_transparent_mosaic_group1.tif"),
                                    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--config_yml_path", required=True, help="path to detectron2 configuration file")
parser.add_argument("--weights_snapshot_path", required=True, help="path to detectron2 model")
parser.add_argument("--rgb_tif_path", required=True, help="path to tiff with rgb map")

args = parser.parse_args()
detector = SickTreesDetectron2Detector(args.config_yml_path, args.weights_snapshot_path)
l.info("rozpoczęto otwieranie pliku tiff, upewnij się że masz wystarczająco ramu albo duży swap - łącznie 56 GB")
with rio.open(args.rgb_tif_path) as rgb_tiff:
    rgb_img = rgb_tiff.read()
l.info("zakończono otwieranie pliku tiff")
print(rgb_img.shape)
rgb_img = rgb_img[(0,1,2),:,:]
rgb_img =  np.moveaxis(rgb_img, 0, 2)
ndvi_non_existing = np.zeros(rgb_img.shape[:2])

forest_segmentator = ForestSegmentation()
l.info("zaczynam segmentować las od innych rzeczy - potrwa to co najmniej 31 minut")
mask = forest_segmentator.mask(rgb_img,  ndvi_non_existing)
print(mask, mask.shape)
l.info("segmentacja lasu zakończona")

l.info("zaczynam wykrywać chore drzewa - zajmie to co najmniej 20min, jeśli nie masz GPU to dłużeeej")
detections = detector.detect(rgb_img, ndvi_non_existing, mask) #takes 20minutes on GTX1080Ti

if not os.path.exists("temp"):
    os.mkdir("temp")

pickle.dump(detections, open("temp/save.p", "wb" ) )
subprocess.call(["python", "test/predictions_to_shp.py", "--rgb_tif_path=/home/h/_drzewaBZBUAS/RGB_szprotawa_transparent_mosaic_group1.tif"])
'''
python test/test_segmentation_and_detection.py --rgb_tif_path=/home/h/_drzewaBZBUAS/RGB_szprotawa_transparent_mosaic_group1.tif --config_yml_path=/home/h/Pobrane/config.yml --weights_snapshot_path=/home/h/Pobrane/model_final.pth
'''