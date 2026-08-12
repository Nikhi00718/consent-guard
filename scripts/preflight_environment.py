"""Verify the Python, CUDA, detection-operator, metric, and dataset stack."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from consentguard.config import load_training_config, project_path
from consentguard.perception.dataset import VisualRedactionsDataset
from consentguard.runtime import atomic_json_dump, environment_snapshot


def check(name: str, action, checks: list[dict]) -> None:
    try:
        detail = action()
        checks.append({"name": name, "passed": True, "detail": detail})
    except Exception as error:  # preflight intentionally aggregates failures
        checks.append({"name": name, "passed": False, "error": f"{type(error).__name__}: {error}"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_smoke.yaml")
    parser.add_argument("--output", type=Path, default=Path("reports/training_environment_preflight.json"))
    args = parser.parse_args()

    checks: list[dict] = []

    def python_check():
        if sys.version_info[:2] != (3, 11):
            raise RuntimeError("Python 3.11 is required")
        return sys.version

    check("python_version", python_check, checks)

    def torchvision_ops():
        import torchvision
        from torchvision.ops import nms, roi_align

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the training environment preflight")
        boxes = torch.tensor(
            [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0]],
            device="cuda",
        )
        scores = torch.tensor([0.9, 0.8], device="cuda")
        kept = nms(boxes, scores, 0.5)
        feature = torch.arange(64, dtype=torch.float32, device="cuda").reshape(1, 1, 8, 8)
        rois = torch.tensor([[0.0, 1.0, 1.0, 6.0, 6.0]], device="cuda")
        pooled = roi_align(feature, rois, output_size=(2, 2), spatial_scale=1.0)
        torch.cuda.synchronize()
        return {
            "torchvision": torchvision.__version__,
            "device": "cuda",
            "nms_kept": kept.cpu().tolist(),
            "roi_align_shape": list(pooled.shape),
        }

    check("torchvision_detection_ops", torchvision_ops, checks)

    def cuda_check():
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this environment")
        tensor = torch.ones(8, device="cuda")
        return {"device": torch.cuda.get_device_name(0), "sum": float(tensor.sum().cpu())}

    check("cuda_tensor", cuda_check, checks)

    def metrics_check():
        import pycocotools
        from pycocotools import mask as mask_utils

        mask = torch.zeros((16, 16), dtype=torch.uint8).numpy()
        mask[2:10, 3:12] = 1
        encoded = mask_utils.encode(mask.copy(order="F"))
        decoded = mask_utils.decode(encoded)
        if not (decoded == mask).all():
            raise RuntimeError("pycocotools mask RLE round trip failed")
        return {
            "rle_area": float(mask_utils.area(encoded)),
            "pycocotools": getattr(pycocotools, "__version__", "installed"),
        }

    check("coco_metrics", metrics_check, checks)

    def dataset_check():
        config = load_training_config(args.config)
        data = config.section("data")
        dataset = VisualRedactionsDataset(
            data["train_records"],
            short_side=data["short_side"],
            max_long_side=data["max_long_side"],
            crop_size=data["crop_size"],
            crop_probability=data["crop_probability"],
            crop_context_factor=data["crop_context_factor"],
            min_crop_visibility=data["min_crop_visibility"],
            training=True,
            limit=1,
        )
        image, target = dataset[0]
        return {"image_shape": list(image.shape), "instances": int(target["labels"].numel())}

    check("real_dataset_sample", dataset_check, checks)
    report = {
        "passed": all(item["passed"] for item in checks),
        "environment": environment_snapshot(),
        "checks": checks,
    }
    output = project_path(args.output)
    atomic_json_dump(report, output)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
