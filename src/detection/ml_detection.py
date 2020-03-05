import numpy as np
import pandas as pd
import geopandas as gpd
import shapely as shp
import torch
import detectron2 as dt2
import detectron2.config
import detectron2.checkpoint
import detectron2.engine
from src.utils.image_processing import sliding_window_iterator
from typing import Optional, List, Dict


class BatchedPredictor(dt2.engine.DefaultPredictor):
    def __call__(self, original_images_bgr):
        """
        Args:
            original_images (List[np.ndarray]): a list of images of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        if type(original_images_bgr) != list:
            original_images_bgr = [original_images_bgr]
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            inputs = []
            for original_image in original_images_bgr:
                if self.input_format == "RGB":
                    # whether the model expects BGR inputs or RGB
                    original_image = original_image[:, :, ::-1]
                height, width = original_image.shape[:2]
                image = self.transform_gen.get_transform(
                    original_image).apply_image(original_image)
                image = torch.as_tensor(
                    image.astype("float32").transpose(2, 0, 1))

                inputs.append(
                    {"image": image, "height": height, "width": width})

            predictions = self.model(inputs)
            return predictions


class SickTreesDetectron2Detector:

    DEFAULT_OVERLAP_PIXELS = 16

    def __init__(self, config_yml_path: str, weights_snapshot_path: str,
                 patch_size: int = 256, bgr_input: bool = True, device='cuda',
                 threshold: float = 0.3, batch_size: int = 32,
                 overlap_windows=True, overlap_pixels=None,
                 postprocess=True):
        """
        Class for sick trees detection using basic detectron2 based model.
        """
        self.cfg = dt2.config.CfgNode(
            dt2.config.CfgNode.load_yaml_with_base(config_yml_path))
        self.cfg.MODEL.WEIGHTS = weights_snapshot_path
        self.cfg.MODEL.DEVICE = device

        self.cfg.MODEL.RETINANET.SCORE_THRESH_TEST = threshold

        self.predictor = BatchedPredictor(self.cfg)
        self.patch_size = patch_size
        self.bgr_input = bgr_input
        self.batch_size = batch_size
        self.overlap_windows = overlap_windows
        self.overlap_pixels = overlap_pixels if overlap_pixels is not None else self.DEFAULT_OVERLAP_PIXELS
        self.postprocess = postprocess
        if self.postprocess:
            self.postprocessor = DetectionsPostProcessor()
        else:
            self.postprocessor = lambda x: x

    def detect(self, rgb_image: np.ndarray, ndvi_image: Optional[np.ndarray] = None):
        assert 3 == len(rgb_image.shape), \
            "RGB image array should be 3-dimensional"
        assert ndvi_image is None or 2 == len(ndvi_image.shape), \
            "NDVI image array should be 2-dimensional"
        # assert 2 == len(forest_mask.shape), \
        #     "Forest mask array should be 2-dimensional"
        assert ndvi_image is None or rgb_image.shape[:2] == ndvi_image.shape, \
            "NDVI image should have the same height and width as RGB"
        # assert rgb_image.shape[:2] == forest_mask.shape, \
        #     "Forest mask should have the same height and width as RGB"

        if self.bgr_input:
            rgb_image = rgb_image[:, :, ::-1]

        if ndvi_image is not None:
            image = np.dstack([rgb_image, ndvi_image])
        else:
            image = rgb_image

        # image = rgb_image * forest_mask[..., np.newaxis]
        predictions = []
        buffer_patches = []
        buffer_offsets = []

        step = self.patch_size
        if self.overlap_windows and (self.overlap_pixels < self.patch_size):
            step = self.patch_size - self.overlap_pixels

        for row, col, window in sliding_window_iterator(image, self.patch_size, step):

            patch = self._prepare_patch(window)

            if patch is not None:
                buffer_patches.append(patch)
                buffer_offsets.append({"row_offset": row, "col_offset": col})

            if len(buffer_patches) >= self.batch_size:
                predictions += self._detect_on_batch(buffer_patches,
                                                     buffer_offsets)
                buffer_offsets = []
                buffer_patches = []

        if len(buffer_patches) >= self.batch_size:
            predictions += self._detect_on_batch(buffer_patches,
                                                 buffer_offsets)

        if self.postprocess:
            predictions = self.postprocessor(predictions)

        return predictions

    def _prepare_patch(self, patch: np.ndarray):

        if (patch == 0).all():
            return None

        patch_shape = (self.patch_size, self.patch_size, patch.shape[2])
        background = np.zeros(patch_shape, dtype=np.uint8)
        background[:patch.shape[0], :patch.shape[1]] = patch
        patch = background
        return patch

    def _detect_on_batch(self, batch_patches: List[np.ndarray], batch_offsets: List[Dict]):

        batch_predictions = self.predictor(batch_patches)

        predictions = []
        for prediction, offsets in zip(batch_predictions, batch_offsets):
            instances = prediction['instances']
            for score, box in zip(instances.scores, instances.pred_boxes):
                c0, r0, c1, r1 = box.round().to(int)
                pred = {"row_min": offsets["row_offset"] + min(r0, r1).item(),
                        "row_max": offsets["row_offset"] + max(r0, r1).item(),
                        "col_min": offsets["col_offset"] + min(c0, c1).item(),
                        "col_max": offsets["col_offset"] + max(c0, c1).item(),
                        "score": score.item()}
                pred["box"] = (pred["row_min"], pred["col_min"],
                               pred["row_max"], pred["col_max"])
                predictions.append(pred)
        return predictions


class DetectionsPostProcessor:

    def __init__(self, buffer_size=2):
        self.buffer_size = buffer_size

    def __call__(self, predictions):
        return self.process(predictions)

    def process(self, predictions):
        predictions_df = self.convert_predictions_to_df(predictions)
        groups_df = self.find_grouped_detections(predictions_df)

        matched = gpd.sjoin(groups_df, predictions_df)
        matched = matched.merge(predictions_df[['prediction_id', 'detection_geometry']],
                                on='prediction_id')
        refined_predictions = self.compute_refined_predictions(matched)
        return refined_predictions

    def convert_predictions_to_df(self, predictions):
        predictions_df = gpd.GeoDataFrame(
            [{"prediction_id": i, "score": p["score"],
              "detection_geometry": shp.geometry.box(p["col_min"], p["row_min"],
                                                     p["col_max"], p["row_max"])}
                for i, p in enumerate(predictions)],
            geometry="detection_geometry")
        predictions_df["detection_area"] = predictions_df["detection_geometry"].area
        return predictions_df

    def find_grouped_detections(self, detections_df):

        detection_groups = (detections_df["detection_geometry"]
                            .buffer(self.buffer_size).unary_union.geoms)

        groups_df = gpd.GeoDataFrame(
            [{"group_id": i, "group_geometry": detection_groups[i]}
             for i in range(len(detection_groups))],
            geometry="group_geometry")
        groups_df['group_area'] = groups_df["group_geometry"].area

        return groups_df

    def compute_refined_predictions(self, grouped_df):
        refined_predictions = grouped_df.groupby("group_id").agg(
            group_id=("group_id", "first"),
            geometry=("group_geometry", "first"),
            geometry_area=("group_area", "first"),
            score_mean=("score", "mean"),
            score_min=("score", "min"),
            score_max=("score", "max"),
            nb_detections=("prediction_id", "count"))

        top_detections = grouped_df.groupby("group_id").apply(
            lambda df: (df[df["score"] == df["score"].max()]
                        [["detection_geometry",
                          "detection_area"]]
                        .iloc[:1]
                        .assign(score_weighted=self.compute_contribution_weighted_score(df)))
        )
        top_detections = top_detections.rename(columns={"detection_area": "top_detection_area",
                                                        "detection_geometry": "top_detection_geometry"})

        refined_predictions = refined_predictions.merge(
            top_detections, on="group_id")
        refined_predictions["score"] = refined_predictions["score_max"]

        refined_predictions = refined_predictions[["geometry", "geometry_area",
                                                   "score", "score_mean",
                                                   "score_min", "score_max",
                                                   "score_weighted",
                                                   "nb_detections",
                                                   "top_detection_area"]
                                                  ].to_dict(orient="rows")
        return refined_predictions

    def compute_contribution_weighted_score(self, df):
        df = df.sort_values("score", ascending=False)
        current_shape = shp.geometry.MultiPolygon([])
        weighted_score = 0
        for i, row in df.iterrows():
            new_shape = current_shape.union(row['detection_geometry'])
            increment = new_shape.area - current_shape.area
            weighted_score += increment * row['score']
            current_shape = new_shape
        overall_score = weighted_score / current_shape.area
        return overall_score
