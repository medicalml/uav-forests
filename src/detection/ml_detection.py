import numpy as np
import detectron2 as dt2
import detectron2.config
import detectron2.checkpoint
import detectron2.engine
from src.utils.image_processing import sliding_window_iterator
from typing import Optional


class SickTreesDetectron2Detector:

    def __init__(self, config_yml_path: str, weights_snapshot_path: str,
                 patch_size: int = 256, bgr_input: bool = True, device='cuda'):
        """
        Class for sick trees detection using basic detectron2 based model.
        """
        self.cfg = dt2.config.CfgNode(
            dt2.config.CfgNode.load_yaml_with_base(config_yml_path))
        self.cfg.MODEL.WEIGHTS = weights_snapshot_path
        self.cfg.MODEL.DEVICE = device

        self.predictor = dt2.engine.DefaultPredictor(self.cfg)
        self.patch_size = patch_size
        self.bgr_input = bgr_input

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
        for row, col, window in sliding_window_iterator(image, self.patch_size):
            pred = self._detect_on_patch(window, row, col)
            predictions += pred
        return predictions
        # box : (row_min, col_min, row_max, col_max)

    def _detect_on_patch(self, patch: np.ndarray,
                         row_offset: int, col_offset: int):
        patch_shape = (self.patch_size, self.patch_size, patch.shape[2])
        background = np.zeros(patch_shape, dtype=np.uint8)
        background[:patch.shape[0], :patch.shape[1]] = patch
        patch = background

        instances = self.predictor(patch)['instances']
        predictions = []
        for score, box in zip(instances.scores, instances.pred_boxes):
            c0, r0, c1, r1 = box.round().to(int)
            pred = {"row_min": row_offset + min(r0, r1).item(),
                    "row_max": row_offset + max(r0, r1).item(),
                    "col_min": col_offset + min(c0, c1).item(),
                    "col_max": col_offset + max(c0, c1).item(),
                    "score": score.item()}
            pred["box"] = (pred["row_min"], pred["col_min"],
                           pred["row_max"], pred["col_max"])
            predictions.append(pred)
        return predictions
