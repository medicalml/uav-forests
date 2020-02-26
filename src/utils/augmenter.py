import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.heatmaps import HeatmapsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

from typing import List, Dict

from detectron2.structures import BoxMode


class Augmenter:

    def __init__(self, augmentation_ratio=1.0,
                 flip_probability=0.5,
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
            iaa.Sometimes(augmentation_ratio * fog_probability, fog),
        ], random_order=True)

        aug_initial = iaa.Sequential([
            rot,
            iaa.Sometimes(flip_probability, flip),
            iaa.Sometimes(augmentation_ratio * contrast_probability, contrast)
        ], random_order=True)

        aug_camera = iaa.Sequential([
            iaa.Sometimes(augmentation_ratio * defocus_probability, defocus),
            iaa.Sometimes(augmentation_ratio *
                          motion_blur_probability, motion_blur),
        ], random_order=False)

        aug_obstacles = iaa.Sequential([
            iaa.Sometimes(augmentation_ratio * cutout_probability, cutout),
            iaa.Sometimes(augmentation_ratio * scale_probability, scale),
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
        if ndvi_image is not None:
            hmap_on_image = HeatmapsOnImage(ndvi_image.astype(np.float32), rgb_image.shape,
                                            min_value=ndvi_image.min(),
                                            max_value=ndvi_image.max() + 1e-9)
        smap_on_image = None
        if forest_mask is not None:
            smap_on_image = SegmentationMapsOnImage(forest_mask,
                                                    rgb_image.shape)

        (aug_img, aug_smap,
         aug_hmap, aug_bbox) = self.augmentation_pipeline(image=rgb_image,
                                                          segmentation_maps=smap_on_image,
                                                          heatmaps=hmap_on_image,
                                                          bounding_boxes=bbox_on_image)

        return {"rgb_image": aug_img,
                "mask": aug_smap and aug_smap.arr.squeeze(),
                "ndvi": aug_hmap and aug_hmap.get_arr().squeeze(),
                "annotations": [{"bbox": [bb.x1_int, bb.y1_int, bb.x2_int, bb.y2_int],
                                 "bbox_mode": BoxMode.XYXY_ABS,
                                 "category_id": bb.label}
                                for bb in aug_bbox.clip_out_of_image()]}

    def __call__(self, rgb_image: np.ndarray,
                 bounding_boxes: List[Dict],
                 ndvi_image: np.ndarray = None,
                 forest_mask: np.ndarray = None):
        return self.augment(rgb_image, bounding_boxes, ndvi_image, forest_mask)
