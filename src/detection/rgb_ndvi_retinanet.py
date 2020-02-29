import torch

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from detectron2.modeling.meta_arch import RetinaNet
from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.meta_arch.retinanet import RetinaNetHead
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.backbone import build_backbone
from detectron2.layers import ShapeSpec, batched_nms, cat
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.logger import log_first_n


@META_ARCH_REGISTRY.register()
class RGB_NDVI_RetinaNet(RetinaNet):
    def __init__(self, cfg):
        super(RetinaNet, self).__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        # fmt: off
        self.num_classes = cfg.MODEL.RETINANET.NUM_CLASSES
        self.in_features = cfg.MODEL.RETINANET.IN_FEATURES
        # Loss parameters:
        self.focal_loss_alpha = cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA
        self.smooth_l1_loss_beta = cfg.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA
        # Inference parameters:
        self.score_threshold = cfg.MODEL.RETINANET.SCORE_THRESH_TEST
        self.topk_candidates = cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST
        self.nms_threshold = cfg.MODEL.RETINANET.NMS_THRESH_TEST
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        # fmt: on

        self.backbone = build_backbone(cfg)

        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        self.head = RetinaNetHead(cfg, feature_shapes)
        self.anchor_generator = build_anchor_generator(cfg, feature_shapes)
        # Matching and loss
        self.box2box_transform = Box2BoxTransform(
            weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.matcher = Matcher(
            cfg.MODEL.RETINANET.IOU_THRESHOLDS,
            cfg.MODEL.RETINANET.IOU_LABELS,
            allow_low_quality_matches=True,
        )
        image_channel_nr = 4  # 3 channels + NDVI
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(
            self.device).view(image_channel_nr, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(
            self.device).view(image_channel_nr, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)
