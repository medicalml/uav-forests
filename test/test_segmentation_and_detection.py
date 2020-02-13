import sys
import os
import argparse
import rasterio as rio 
import numpy as np
import pickle 
import tqdm
import fiona
from shapely.geometry import Point, mapping, Polygon
import logging as l
import torch

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
parser.add_argument("--forest_shp_path", required=True, help="path to tiff with shp showing where trees are ")

args = parser.parse_args()
l.info("II ")
if torch.cuda.device_count():
    device = 'cuda'
else:
    device = 'cpu'
detector = SickTreesDetectron2Detector(args.config_yml_path, args.weights_snapshot_path, device=device)
l.info("rozpoczęto otwieranie pliku tiff, upewnij się że masz wystarczająco ramu albo duży swap")
geotiff = rio.open(args.rgb_tif_path)
l.info("zakończono otwieranie pliku tiff")



forest_segmentator = ForestSegmentation()
l.info("zaczynam segmentować las od innych rzeczy - potrwa to co najmniej 31 minut")


l.info("segmentacja lasu zakończona")
iterator = ForestIterator(args.rgb_tif_path, args.forest_shp_path)
l.info("zaczynam wykrywać chore drzewa - zajmie to co najmniej 20min, jeśli nie masz GPU to dłużeeej")
#detections = detector.detect(rgb_img, ndvi_non_existing, mask) #takes 20minutes on GTX1080Ti

schema = {
    'geometry': 'Polygon',
    'properties': {'id': 'int'},
}
c = fiona.open('temp_avg/predictions.shp', 'w', 'ESRI Shapefile', schema)
score_threshold = 0.5
for patch in tqdm.tqdm(iterator, total=len(iterator)):
    rgb = patch['rgb']
    rgb = np.moveaxis(rgb, 0, -1)
    #print(rgb.shape)
    ndvi = patch['ndvi']
    
    #mask = forest_segmentator.mask(rgb,  ndvi_non_existing)
    mask = np.ones(rgb.shape[:2])
    detections = detector.detect(rgb, ndvi_non_existing, mask)
    x_min, y_min = patch['left_upper_corner_coordinates']
    y_min_pixels, x_min_pixels = rio.transform.rowcol(geotiff.transform, x_min, y_min)
    
    for i, pred in enumerate(detections):
        print(pred['col_min'], pred['row_min'], pred['col_max'], pred['row_max'])
        if pred['score'] > score_threshold:
            coors_to_write = []
            coors_to_write.append(rio.transform.xy(geotiff.transform, y_min_pixels+pred['row_min'], x_min_pixels+pred['col_min']) )
            coors_to_write.append(rio.transform.xy(geotiff.transform, y_min_pixels+pred['row_min'], x_min_pixels+pred['col_max']) )
            coors_to_write.append(rio.transform.xy(geotiff.transform, y_min_pixels+pred['row_max'], x_min_pixels+pred['col_max']) )
            coors_to_write.append(rio.transform.xy(geotiff.transform, y_min_pixels+pred['row_max'], x_min_pixels+pred['col_min']) )
            
            poly = Polygon([coors_to_write[0], coors_to_write[1], coors_to_write[2], coors_to_write[3], coors_to_write[0]])
            c.write({
                'geometry': mapping(poly),
                'properties': {'id': i},
            })
'''
python test/test_segmentation_and_detection.py --rgb_tif_path=/home/m/ML\ dane\ dla\ kola/Swiebodzin/RGB_Swiebodzin.tif --config_yml_path=/home/h/uav-forests/tboard_logs/retinanet_test_2020-01-21T23:40/model_0012249.pth --forest_shp_path=/home/m/ML\ dane\ dla\ kola/Swiebodzin/config.yml --weights_snapshot_path=/home/h/uav-forests/tboard_logs/retinanet_test_2020-01-21T23:40/model_0012249.pth --forest_shp_path=/home/m/ML\ dane\ dla\ kola/Swiebodzin/obszar_swiebodzin.shp
'''