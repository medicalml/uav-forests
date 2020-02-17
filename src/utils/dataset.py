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

from typing import List, Dict


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


def augment_image(rgb_image: np.ndarray, bounding_boxes: List[Dict],
                  ndvi_image: np.ndarray = None,
                  forest_mask: np.ndarray = None):

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
        smap_on_image = SegmentationMapsOnImage(forest_mask, rgb_image.shape)

    rot = iaa.Rot90(ia.ALL)
    shift = iaa.Affine(translate_percent=(-0.2, 0.2))
    flip = iaa.OneOf([iaa.Fliplr(0.5), iaa.Flipud(0.5)])
    scale = iaa.Affine(scale=(0.75, 1.25))
    motion_blur = iaa.MotionBlur()
    cutout = iaa.Cutout(nb_iterations=(1, 3), size=(0.01, 0.1), squared=False)
    fog = iaa.imgcorruptlike.Fog(severity=1)
    defocus = iaa.imgcorruptlike.DefocusBlur(severity=1)
    contrast = iaa.imgcorruptlike.Contrast(severity=1)

    seq_weather = iaa.Sequential([
        iaa.Sometimes(0.1, fog),
    ], random_order=True)

    seq_initial = iaa.Sequential([
        rot,
        iaa.Sometimes(0.5, flip),
        iaa.Sometimes(0.2, contrast)
    ], random_order=True)

    seq_camera = iaa.Sequential([
        iaa.Sometimes(0.1, defocus),
        iaa.Sometimes(0.2, motion_blur),
    ], random_order=False)

    seq_obstacles = iaa.Sequential([
        iaa.Sometimes(0.2, cutout),
        iaa.Sometimes(0.1, scale),
        iaa.Sometimes(0.2, shift)
    ], random_order=False)

    seq = iaa.Sequential([seq_weather, seq_initial,
                          seq_camera, seq_obstacles],
                         random_order=False)

    aug_img, aug_smap, aug_hmap, aug_bbox = seq(image=rgb_image, 
                                                segmentation_maps=smap_on_image, 
                                                heatmaps=hmap_on_image, 
                                                bounding_boxes=bbox_on_image)
    
    return {"rgb": aug_img,
            "mask": aug_smap.arr.squeeze(),
            "ndvi": aug_hmap.to_uint8().squeeze(),
            "annotations": [{"bbox": [bb.x1_int, bb.y1_int, bb.x2_int, bb.y2_int],
                             "bbox_mode": BoxMode.XYXY_ABS,
                             "category_id": bb.label}
                            for bb in aug_bbox.bounding_boxes.clip_out_of_image()]}
    
