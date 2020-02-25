from detectron2.data import detection_utils as utils
import os
import cv2
import pickle
import pandas as pd
import geopandas as gpd
import numpy as np
import copy
import torch

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.heatmaps import HeatmapsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog, DatasetMapper, detection_utils as det_utils
from detectron2.data import build_detection_test_loader, build_detection_train_loader

from detectron2.evaluation import COCOEvaluator
from detectron2.engine import DefaultTrainer
from detectron2.config import CfgNode

from src.utils.dataset import split_train_val_test
from src.utils.augmenter import Augmenter
from src.utils.custom_coco_evaluator import CustomCOCOEvaluator


from typing import List, Dict


class SickTreesDatasetMapper:

    def __init__(self, cfg: CfgNode, is_train: bool,
                 nb_channels: int = 4,
                 augmenter: Augmenter = None):
        self.cfg = cfg
        self.is_train = is_train
        self.nb_channels = nb_channels
        self.augmenter = augmenter

    def __call__(self, data_dict):
        data_dict = copy.deepcopy(data_dict)

        image = cv2.imread(data_dict["file_name"], cv2.IMREAD_UNCHANGED)
        image = image[:, :, :self.nb_channels]

        # if image.shape[2] == 4:
        #     conversion = cv2.COLOR_BGRA2RGBA
        # else:
        #     conversion = cv2.COLOR_BGR2RGB
        # image = cv2.cvtColor(image, conversion)

        # data_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1)
        #                                      .astype("float32"))

        if self.augmenter:
            rgb_image = image[:, :, :3]
            ndvi_image = None
            if image.shape[2] == 4:
                ndvi_image = image[:, :, 3]

            data_dict["original_annotations"] = data_dict["annotations"]
            data_dict["original_image"] = image

            augmented = self.augmenter(rgb_image,
                                       data_dict["annotations"],
                                       ndvi_image)

            data_dict["annotations"] = augmented["annotations"]
            if augmented['ndvi'] is not None:
                image = np.dstack([augmented["rgb_image"], augmented["ndvi"]])
            else:
                image = augmented["rgb_image"]

        if self.cfg.INPUT.FORMAT == "BGR":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif self.cfg.INPUT.FORMAT == "BGRN":
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)

        data_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1)
                                             .astype("float32"))

        instances = det_utils.annotations_to_instances(data_dict["annotations"],
                                                       image.shape[:2])

        data_dict["instances"] = det_utils.filter_empty_instances(instances)

        return data_dict


class SickTreesCFGTrainer(DefaultTrainer):

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name,
                                           mapper=SickTreesDatasetMapper(cfg, is_train=False,
                                                                         nb_channels=3,
                                                                         augmenter=None))

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.AUGMENTATION == "ON":
            augmenter = Augmenter()
        else:
            augmenter = None
        return build_detection_train_loader(cfg, mapper=SickTreesDatasetMapper(cfg, is_train=True,
                                                                               nb_channels=3,
                                                                               augmenter=augmenter))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs(cfg.OUTPUT_DIR+f"/eval/{dataset_name}", exist_ok=True)

        return CustomCOCOEvaluator(dataset_name, cfg, False, output_dir=cfg.OUTPUT_DIR+f"/eval/{dataset_name}")


class SickTreesAugmentedTrainer(DefaultTrainer):

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name,
                                           mapper=SickTreesDatasetMapper(cfg, is_train=False,
                                                                         nb_channels=3,
                                                                         augmenter=None))

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=SickTreesDatasetMapper(cfg, is_train=True,
                                                                               nb_channels=3,
                                                                               augmenter=Augmenter()))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs(cfg.OUTPUT_DIR+f"/eval/{dataset_name}", exist_ok=True)

        return CustomCOCOEvaluator(dataset_name, cfg, False, output_dir=cfg.OUTPUT_DIR+f"/eval/{dataset_name}")


class SickTreesNDVIAugmentedTrainer(DefaultTrainer):

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name,
                                           mapper=SickTreesDatasetMapper(cfg, is_train=False,
                                                                         nb_channels=4,
                                                                         augmenter=None))

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=SickTreesDatasetMapper(cfg, is_train=True,
                                                                               nb_channels=4,
                                                                               augmenter=Augmenter()))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs(cfg.OUTPUT_DIR+f"/eval/{dataset_name}", exist_ok=True)

        return CustomCOCOEvaluator(dataset_name, cfg, False, output_dir=cfg.OUTPUT_DIR+f"/eval/{dataset_name}")
