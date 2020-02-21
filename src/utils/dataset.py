import os
import cv2
import pandas as pd
import geopandas as gpd
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.heatmaps import HeatmapsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

from sklearn.model_selection import train_test_split

from typing import List, Dict, Set, Union, Optional, Tuple


def split_train_val_test(samples, train_ratio, val_ratio, test_ratio=None):
    if test_ratio is None:
        test_ratio = 0
    elif train_ratio + val_ratio == 1:
        test_ratio = 0
    else:
        test_ratio = (1.0 - train_ratio - val_ratio)
    assert 0 <= train_ratio <= 1
    assert 0 <= val_ratio <= 1
    assert 0 <= test_ratio <= 1, f"{test_ratio}"
    assert 0 < train_ratio + val_ratio + test_ratio <= 1

    train, valtest = train_test_split(samples, train_size=train_ratio)
    if test_ratio > 0:
        val, test = train_test_split(
            valtest, train_size=val_ratio / (val_ratio + test_ratio))
    else:
        val = valtest
        test = []

    return {"train": train,
            "val": val,
            "test": test}


class DatasetsDictsGenerator:

    def __init__(self, images_dir: str, patches_subset: Union[List, Set],
                 min_bbox_area: Union[int, float] = 200,
                 limit_samples: int = -1):
        self.images_dir = images_dir
        self.patches_subset = patches_subset
        self.min_bbox_area = min_bbox_area
        self.limit_samples = limit_samples

    @staticmethod
    def _convert_single_bbox(bbox: Dict):
        py, px = bbox.exterior.xy  # py, px inverted on purpose!
        return {"bbox": [int(min(px)), int(min(py)),
                         int(max(px)), int(max(py))],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 0}

    @staticmethod
    def _convert_single_patch(patch_df: pd.DataFrame, images_dir: str,
                              image_shape: Tuple[int, int] = (256, 256)):
        patch_number = patch_df["patch_number"].iloc[0]
        file_path = os.path.join(images_dir, f"patch_{patch_number}.png")
        image_shape = image_shape or cv2.imread(file_path).shape[:2]
        return {"file_name": file_path,
                "image_id": int(patch_number),
                "height": image_shape[0],
                "width": image_shape[1],
                "annotations": (patch_df["bbox"]
                                .apply(DatasetsDictsGenerator._convert_single_bbox)
                                .tolist())}

    def __call__(self):
        annotations = pd.read_pickle(
            f"{self.images_dir}/annotation.pkl").set_geometry("bbox")
        annotations = annotations[annotations["bbox"].area >=
                                  self.min_bbox_area]
        annotations = annotations[(annotations["patch_number"]
                                   .isin(self.patches_subset[:self.limit_samples]))]

        if len(annotations) == 0:
            return []
        return (annotations.groupby("patch_number")
                .apply(DatasetsDictsGenerator._convert_single_patch,
                       images_dir=self.images_dir,
                       image_shape=(256, 256))
                .tolist())


def register_detectron2_datasets(name: str, images_dir: str,
                                 splits: Dict[str, Union[List, Set]],
                                 min_bbox_area: Union[int, float] = 200,
                                 limits: Optional[Dict[str, int]] = None):
    if limits is None:
        limits = {"train": -1, "val": -1, "test": -1}

    for d in splits.keys():
        DatasetCatalog.register(f"{name}_{d}",
                                DatasetsDictsGenerator(images_dir,
                                                       splits[d],
                                                       min_bbox_area,
                                                       limits[d]))

        MetadataCatalog.get(f"{name}_{d}").set(thing_classes=["SickTrees"])


def register_detectron2_multipart_datasets(name: str,
                                           parts_images_dirs: Dict[str, str],
                                           splits: Dict[str, Dict[str, Union[List, Set]]],
                                           min_bbox_area: Union[int, float] = 200):
    for d in splits.keys():
        parts_loaders = []
        for part_name, part_dir in parts_images_dirs.items():
            part_loader = DatasetsDictsGenerator(part_dir,
                                                 splits[d][part_name],
                                                 min_bbox_area)
            parts_loaders.append(part_loader)

        def total_loader(parts_loaders=parts_loaders):
            return [single_dict
                    for part_loader in parts_loaders
                    for single_dict in part_loader()]

        DatasetCatalog.register(f"{name}_{d}", total_loader)
        MetadataCatalog.get(f"{name}_{d}").set(thing_classes=["SickTrees"])
