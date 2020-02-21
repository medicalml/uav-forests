from detectron2.data import detection_utils as utils
import os
import cv2
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


from sklearn.model_selection import train_test_split

from typing import List, Dict


class Augmenter:

    def __init__(self, flip_probability=0.5,
                 contrast_probability=0.2,
                 defocus_probability=0.1,
                 motion_blur_probability=0.2,
                 cutout_probability=0.2,
                 scale_probability=0.1,
                 shift_probability=0.2,
                 fog_probability=0.1):
        rot = iaa.Rot90(ia.ALL)
        shift = iaa.Affine(translate_percent=(-0.2, 0.2))
        flip = iaa.OneOf([iaa.Fliplr(1), iaa.Flipud(1)])
        scale = iaa.Affine(scale=(0.75, 1.25))
        motion_blur = iaa.MotionBlur()
        cutout = iaa.Cutout(
            nb_iterations=(1, 3),
            size=(0.01, 0.1),
            squared=False)
        fog = iaa.imgcorruptlike.Fog(severity=1)
        defocus = iaa.imgcorruptlike.DefocusBlur(severity=1)
        contrast = iaa.imgcorruptlike.Contrast(severity=1)

        aug_weather = iaa.Sequential([
            iaa.Sometimes(fog_probability, fog),
        ], random_order=True)

        aug_initial = iaa.Sequential([
            rot,
            iaa.Sometimes(flip_probability, flip),
            iaa.Sometimes(contrast_probability, contrast)
        ], random_order=True)

        aug_camera = iaa.Sequential([
            iaa.Sometimes(defocus_probability, defocus),
            iaa.Sometimes(motion_blur_probability, motion_blur),
        ], random_order=False)

        aug_obstacles = iaa.Sequential([
            iaa.Sometimes(cutout_probability, cutout),
            iaa.Sometimes(scale_probability, scale),
            iaa.Sometimes(shift_probability, shift)
        ], random_order=False)

        self.augmentation_pipeline = iaa.Sequential(
            [aug_weather, aug_initial, aug_camera, aug_obstacles],
            random_order=False)

    def augment(self, rgb_image: np.ndarray,
                bounding_boxes: List[Dict],
                ndvi_image: np.ndarray = None,
                forest_mask: np.ndarray = None):
        assert all(bb['bbox_mode'] == BoxMode.XYXY_ABS
                   for bb in bounding_boxes), \
            "Unsupported bbox_mode"

        bbox_on_image = BoundingBoxesOnImage([
            BoundingBox(x1=annotation["bbox"][0],
                        y1=annotation["bbox"][1],
                        x2=annotation["bbox"][2],
                        y2=annotation["bbox"][3],
                        label=annotation["category_id"])
            for annotation in bounding_boxes
        ], rgb_image.shape)

        hmap_on_image = None
        if ndvi_image:
            hmap_on_image = HeatmapsOnImage(ndvi_image, rgb_image.shape,
                                            min_value=ndvi_image.min(),
                                            max_value=ndvi_image.max())
        smap_on_image = None
        if forest_mask:
            smap_on_image = SegmentationMapsOnImage(forest_mask,
                                                    rgb_image.shape)

        (aug_img, aug_smap,
         aug_hmap, aug_bbox) = self.augmentation_pipeline(image=rgb_image,
                                                          segmentation_maps=smap_on_image,
                                                          heatmaps=hmap_on_image,
                                                          bounding_boxes=bbox_on_image)

        return {"rgb_image": aug_img,
                "mask": aug_smap and aug_smap.arr.squeeze(),
                "ndvi": aug_hmap and aug_hmap.to_uint8().squeeze(),
                "annotations": [{"bbox": [bb.x1_int, bb.y1_int, bb.x2_int, bb.y2_int],
                                 "bbox_mode": BoxMode.XYXY_ABS,
                                 "category_id": bb.label}
                                for bb in aug_bbox.clip_out_of_image()]}

    def __call__(self, rgb_image: np.ndarray,
                 bounding_boxes: List[Dict],
                 ndvi_image: np.ndarray = None,
                 forest_mask: np.ndarray = None):
        return self.augment(rgb_image, bounding_boxes, ndvi_image, forest_mask)


class SickTreesDatasetMapper:

    def __init__(self, cfg: CfgNode, is_train: bool, augmenter: Augmenter = None):
        self.cfg = cfg
        self.is_train = is_train
        self.augmenter = augmenter

    def __call__(self, data_dict):
        data_dict = copy.deepcopy(data_dict)

        image = cv2.imread(data_dict["file_name"])
        if image.shape[2] == 4:
            conversion = cv2.COLOR_BGRA2RGBA
        else:
            conversion = cv2.COLOR_BGR2RGB
        image = cv2.cvtColor(image, conversion)

        data_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1)
                                             .astype("float32"))

        if self.augmenter:
            rgb_image = image[:, :, :3]
            ndvi_image = None
            if image.shape[2] == 4:
                ndvi_image = image[:, :, 3]

            data_dict["original_annotations"] = data_dict["annotations"]
            data_dict["original_image"] = data_dict["image"]

            augmented = self.augmenter(rgb_image,
                                       data_dict["annotations"],
                                       ndvi_image)

            data_dict["annotations"] = augmented["annotations"]
            if augmented['ndvi']:
                image = np.dstack([augmented["rgb_image"], augmented["ndvi"]])
            else:
                image = augmented["rgb_image"]

        data_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1)
                                             .astype("float32"))

        instances = det_utils.annotations_to_instances(data_dict["annotations"],
                                                       image.shape[:2])

        data_dict["instances"] = det_utils.filter_empty_instances(instances)

        return data_dict


class SickTreesAugmentedTrainer(DefaultTrainer):

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name,
                                           mapper=SickTreesDatasetMapper(cfg, is_train=False,
                                                                         augmenter=None))

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=SickTreesDatasetMapper(cfg, is_train=True,
                                                                               augmenter=Augmenter()))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluator(dataset_name, cfg, False, output_dir=cfg.OUTPUT_DIR+f"/eval/{dataset_name}")
