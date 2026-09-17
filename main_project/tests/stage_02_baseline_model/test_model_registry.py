from __future__ import annotations

import torch

from consentguard.stage_02_baseline_model.models import (
    ClassAgnosticMaskRCNNPredictor,
    build_instance_segmentation_model,
)


def test_maskrcnn_v2_builds_custom_heads_and_small_anchors() -> None:
    model = build_instance_segmentation_model(
        {
            "name": "maskrcnn_resnet50_fpn_v2",
            "pretrained": False,
            "trainable_backbone_layers": 5,
            "small_object_anchors": True,
            "detections_per_image": 20,
            "rpn_pre_nms_top_n_train": 100,
            "rpn_post_nms_top_n_train": 50,
            "rpn_pre_nms_top_n_test": 50,
            "rpn_post_nms_top_n_test": 20,
            "box_batch_size_per_image": 32,
        },
        num_classes=29,
        min_size=128,
        max_size=192,
    )
    assert model.roi_heads.box_predictor.cls_score.out_features == 29
    assert model.roi_heads.mask_predictor.mask_fcn_logits.out_channels == 29
    assert model.rpn.anchor_generator.sizes[0] == (16,)


def test_class_agnostic_predictor_has_one_trainable_mask_channel() -> None:
    predictor = ClassAgnosticMaskRCNNPredictor(8, 16, num_classes=4)
    features = torch.randn(2, 8, 7, 7)
    logits = predictor(features)

    assert predictor.mask_fcn_logits.out_channels == 1
    assert tuple(logits.shape) == (2, 4, 14, 14)
    assert torch.equal(logits[:, 0], logits[:, 1])
    assert torch.equal(logits[:, 1], logits[:, 3])


def test_class_agnostic_model_builds_with_torchvision_compatible_interface() -> None:
    model = build_instance_segmentation_model(
        {
            "name": "maskrcnn_resnet50_fpn_v2",
            "pretrained": False,
            "trainable_backbone_layers": 5,
            "class_agnostic_mask": True,
            "small_object_anchors": True,
            "rpn_pre_nms_top_n_train": 50,
            "rpn_post_nms_top_n_train": 20,
            "rpn_pre_nms_top_n_test": 50,
            "rpn_post_nms_top_n_test": 20,
            "box_batch_size_per_image": 16,
        },
        num_classes=4,
        min_size=64,
        max_size=96,
    )
    assert model.roi_heads.box_predictor.cls_score.out_features == 4
    assert model.roi_heads.mask_predictor.mask_fcn_logits.out_channels == 1

    image = torch.rand(3, 64, 64)
    masks = torch.zeros((2, 64, 64), dtype=torch.uint8)
    masks[0, 8:24, 8:24] = 1
    masks[1, 32:52, 32:54] = 1
    target = {
        "boxes": torch.tensor([[8.0, 8.0, 24.0, 24.0], [32.0, 32.0, 54.0, 52.0]]),
        "labels": torch.tensor([1, 3], dtype=torch.int64),
        "masks": masks,
        "image_id": torch.tensor([1]),
        "area": masks.flatten(1).sum(1).float(),
        "iscrowd": torch.zeros(2, dtype=torch.int64),
    }
    model.train()
    losses = model([image], [target])
    assert torch.isfinite(sum(losses.values()))

    model.eval()
    with torch.inference_mode():
        predictions = model([image])
    assert len(predictions) == 1
    assert predictions[0]["masks"].shape[1] == 1
