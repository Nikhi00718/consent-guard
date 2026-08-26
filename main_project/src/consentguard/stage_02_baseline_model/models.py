"""Instance-segmentation model registry."""

from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn

from consentguard.shared.paths import project_path


# Keep large pretrained artifacts inside the project data cache by default.
# An explicit user/system TORCH_HOME still takes precedence.
os.environ.setdefault("TORCH_HOME", str(project_path("data/cache/torch")))


SUPPORTED_MODELS = {
    "fasterrcnn_resnet50_fpn_v2",
    "maskrcnn_resnet50_fpn_v2",
    "maskrcnn_resnet50_fpn",
}


class ClassAgnosticMaskRCNNPredictor(nn.Module):
    """Mask predictor with one trainable geometry channel.

    TorchVision's stock ``RoIHeads`` indexes mask logits by the predicted
    foreground label during both loss computation and inference.  Returning a
    literal ``[N, 1, H, W]`` tensor would therefore make labels 1..K index
    past the channel dimension.  This predictor keeps one trainable mask
    channel and expands it across the class dimension only at the interface
    expected by the stock TorchVision head.  The selected class channel is
    consequently identical for every foreground class while the checkpoint
    stores only one mask-output channel.
    """

    def __init__(self, in_channels: int, dim_reduced: int, num_classes: int) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must include background and at least one foreground class")
        self.num_classes = int(num_classes)
        self.conv5_mask = nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)
        self.relu = nn.ReLU(inplace=True)
        self.mask_fcn_logits = nn.Conv2d(dim_reduced, 1, 1, 1, 0)
        for parameter in self.parameters():
            if parameter.ndim >= 2:
                nn.init.kaiming_normal_(parameter, mode="fan_out", nonlinearity="relu")
            else:
                nn.init.zeros_(parameter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.mask_fcn_logits(self.relu(self.conv5_mask(x)))
        return logits.expand(-1, self.num_classes, -1, -1)


def build_instance_segmentation_model(
    model_config: dict[str, Any],
    *,
    num_classes: int,
    min_size: int,
    max_size: int,
) -> torch.nn.Module:
    """Build Mask R-CNN and replace COCO prediction heads.

    ``num_classes`` includes class 0 (background), matching TorchVision's
    documented convention.
    """

    try:
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_V2_Weights,
            MaskRCNN_ResNet50_FPN_V2_Weights,
            MaskRCNN_ResNet50_FPN_Weights,
            fasterrcnn_resnet50_fpn_v2,
            maskrcnn_resnet50_fpn,
            maskrcnn_resnet50_fpn_v2,
        )
        from torchvision.models.detection.anchor_utils import AnchorGenerator
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "TorchVision detection operators are unavailable. Install torch and torchvision from the same "
            "official wheel index, then run scripts/preflight_environment.py."
        ) from error

    name = str(model_config["name"])
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model {name!r}; choose one of {sorted(SUPPORTED_MODELS)}")
    pretrained = bool(model_config.get("pretrained", True))
    trainable_layers = int(model_config.get("trainable_backbone_layers", 3))
    if not pretrained and trainable_layers != 5:
        raise ValueError(
            "A randomly initialized backbone must train all 5 backbone layers; "
            "set trainable_backbone_layers=5 or enable pretrained weights."
        )
    detections_per_image = int(model_config.get("detections_per_image", 100))

    shared_kwargs: dict[str, Any] = {
        "min_size": int(min_size),
        "max_size": int(max_size),
        "box_score_thresh": float(model_config.get("internal_score_threshold", 0.0)),
        "box_detections_per_img": detections_per_image,
    }
    if pretrained:
        shared_kwargs["trainable_backbone_layers"] = trainable_layers
    # Reduced proposal counts make the explicit smoke profile fast without
    # changing defaults for the scientific baseline.
    optional_kwargs = {
        "rpn_pre_nms_top_n_train": "rpn_pre_nms_top_n_train",
        "rpn_post_nms_top_n_train": "rpn_post_nms_top_n_train",
        "rpn_pre_nms_top_n_test": "rpn_pre_nms_top_n_test",
        "rpn_post_nms_top_n_test": "rpn_post_nms_top_n_test",
        "box_batch_size_per_image": "box_batch_size_per_image",
    }
    for config_key, argument_name in optional_kwargs.items():
        if model_config.get(config_key) is not None:
            shared_kwargs[argument_name] = int(model_config[config_key])

    if name == "fasterrcnn_resnet50_fpn_v2":
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_resnet50_fpn_v2(
            weights=weights,
            weights_backbone=None,
            **shared_kwargs,
        )
    elif name == "maskrcnn_resnet50_fpn_v2":
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
        # Pass ``weights_backbone`` explicitly so builder defaults cannot cause
        # a surprise network download in offline smoke/checkpoint-loading runs.
        model = maskrcnn_resnet50_fpn_v2(
            weights=weights,
            weights_backbone=None,
            **shared_kwargs,
        )
    else:
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model = maskrcnn_resnet50_fpn(
            weights=weights,
            weights_backbone=None,
            **shared_kwargs,
        )

    box_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, num_classes)
    if hasattr(model.roi_heads, "mask_predictor") and model.roi_heads.mask_predictor is not None:
        mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
        if bool(model_config.get("class_agnostic_mask", False)):
            model.roi_heads.mask_predictor = ClassAgnosticMaskRCNNPredictor(
                mask_features,
                256,
                num_classes,
            )
        else:
            model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, 256, num_classes)

    if bool(model_config.get("small_object_anchors", False)):
        # Keep three anchors per location, so the existing RPN head remains
        # compatible while shifting scales toward small text/faces/documents.
        model.rpn.anchor_generator = AnchorGenerator(
            sizes=((16,), (32,), (64,), (128,), (256,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        )
    return model


def model_summary(model: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"parameters": total, "trainable_parameters": trainable}
