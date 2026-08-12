from __future__ import annotations

from consentguard.perception.models import build_instance_segmentation_model


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
