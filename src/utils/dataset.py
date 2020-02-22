import os

import cv2
import pandas as pd
import geopandas as gpd
import numpy as np

from sklearn.model_selection import train_test_split
from fvcore.common.file_io import PathManager

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog


def split_train_val_test(samples, train_ratio, val_ratio, test_ratio=None):
    test_ratio = test_ratio or (1.0 - train_ratio - val_ratio)
    assert 0 < train_ratio < 1
    assert 0 < val_ratio < 1
    assert 0 < test_ratio < 1
    assert 0 < train_ratio + val_ratio + test_ratio <= 1

    train, valtest = train_test_split(samples, train_size=train_ratio)
    val, test = train_test_split(
        valtest, train_size=val_ratio / (val_ratio + test_ratio))

    return {"train": train,
            "val": val,
            "test": test}


def _convert_single_bbox(bbox):
    py, px = bbox.exterior.xy  # py, px inverted on purpose!
    return {"bbox": [int(min(px)), int(min(py)),
                     int(max(px)), int(max(py))],
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": 0}


def _convert_single_patch(patch_df, images_dir, image_shape=(256, 256)):
    patch_number = patch_df["patch_number"].iloc[0]
    file_path = os.path.join(images_dir, f"patch_{patch_number}.png")
    image_shape = image_shape or cv2.imread(file_path).shape[:2]
    return {"file_name": file_path,
            "image_id": int(patch_number),
            "height": image_shape[0],
            "width": image_shape[1],
            "annotations": patch_df["bbox"].apply(_convert_single_bbox).tolist()}


def get_detectron2_dataset_dicts(images_dir, patches_subset, min_bbox_area=200,
                                 limit_samples=-1):

    annotations = pd.read_pickle(
        f"{images_dir}/annotation.pkl").set_geometry("bbox")
    annotations = annotations[annotations["bbox"].area >= min_bbox_area]
    annotations = annotations[(annotations["patch_number"]
                               .isin(patches_subset[:limit_samples]))]

    return (annotations.groupby("patch_number")
            .apply(_convert_single_patch,
                   images_dir=images_dir,
                   image_shape=(256, 256))
            .tolist())


def register_detectron2_datasets(name, images_dir, splits, min_bbox_area=200, limits=None):
    if limits is None:
        limits = {"train": -1, "val": -1, "test": -1}

    for d in splits.keys():
        DatasetCatalog.register(f"{name}_{d}",
                                lambda subset=splits[d], lim=limits[d]:
                                    get_detectron2_dataset_dicts(images_dir,
                                                                 subset,
                                                                 min_bbox_area,
                                                                 lim))

        MetadataCatalog.get(f"{name}_{d}").set(thing_classes=["SickTrees"])

def image_reader(file_name, format=None):
    with PathManager.open(file_name, "rb") as f:
        image = cv2.imread(f.name, cv2.IMREAD_UNCHANGED)
        return image