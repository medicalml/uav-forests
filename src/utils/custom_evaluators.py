import itertools
import json
import numpy as np
import shapely as shp
from collections import OrderedDict
from tabulate import tabulate

from detectron2.utils.logger import create_small_table

from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.evaluation import COCOEvaluator
from detectron2.data.detection_utils import BoxMode


class SickTreesEvaluator(DatasetEvaluator):

    def __init__(self, thresholds=(0, 25, 50, 75), log_inputs_and_outputs=False):
        self.thresholds = thresholds
        self.log_inputs_and_outputs = log_inputs_and_outputs

    def reset(self):
        self.inputs = []
        self.outputs = []
        self.ground_truths_count = 0
        self.detections_count = 0
        self.detected_ground_truths_counts = {t: 0 for t in self.thresholds}
        self.correct_detections_counts = {t: 0 for t in self.thresholds}

    def process(self, inputs, outputs):
        assert inputs[0]["annotations"][0]["bbox_mode"] == BoxMode.XYXY_ABS, \
            "Only BoxMode.XYXY_ABS supported"

        if self.log_inputs_and_outputs:
            self.inputs += inputs
            self.outputs += outputs

        for input, output in zip(inputs, outputs):

            self.ground_truths_count += len(input["annotations"])
            self.detections_count += len(output["instances"])

            boxes = [shp.geometry.box(*pred_box.tolist())
                     for pred_box in output['instances'].pred_boxes]
            gtruths = [shp.geometry.box(*ann['bbox'])
                       for ann in input['annotations']]

            for gt in gtruths:
                cover, _ = self._get_intersections_and_matching_geom(gt, boxes)
                cover_percentage = cover.area / gt.area * 100
                for t in self.thresholds:
                    if cover_percentage > t:
                        self.detected_ground_truths_counts[t] += 1

            for box in boxes:
                cover, _ = self._get_intersections_and_matching_geom(
                    box, gtruths)
                cover_percentage = cover.area / box.area * 100
                for t in self.thresholds:
                    if cover_percentage > t:
                        self.correct_detections_counts[t] += 1

    def _get_intersections_and_matching_geom(self, base, geoms):
        matching_geom = []
        intersections = []
        for geom in geoms:
            overlap = base.intersection(geom)
            if not overlap.is_empty:
                matching_geom.append(geom)
                intersections.append(overlap)

        intersections = shp.geometry.MultiPolygon(intersections).buffer(0)
        matching_geom = shp.geometry.MultiPolygon(matching_geom).buffer(0)
        return intersections, matching_geom

    def evaluate(self):
        recalls = {f"Recall_{t}": self.detected_ground_truths_counts[t] / self.ground_truths_count
                   for t in self.thresholds}
        precisions = {f"Precision_{t}": self.correct_detections_counts[t] / self.detections_count
                      for t in self.thresholds}
        return {"sick_trees_bbox": {"ground_truth_count": self.ground_truths_count,
                                    "detections_count": self.detections_count,
                                    **recalls,
                                    **precisions}}


class COCOEvaluatorWithRecall(COCOEvaluator):

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"] + ["AR1", "AR10", "AR100", "ARs", "ARm", "ARl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(
                coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(
                iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Note that some metrics cannot be computed.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(
            *[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        results = {f"{self._metadata.name}_{k}": results[k] for k in results}
        return results
