"""Prove that the preprocessed image/mask path produces a learning signal."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.consentguard.perception.dataset import VisualRedactionsDataset


class TinyUnionMaskNet(nn.Module):
    """CPU smoke model only; not the research baseline."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, image: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        logits = self.encoder(image.unsqueeze(0))
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False).squeeze(0)


def main() -> None:
    torch.manual_seed(7)
    dataset = VisualRedactionsDataset(
        ROOT / "data" / "processed" / "visual_redactions_verified_visual" / "records_train2017.jsonl",
        short_side=128,
        max_long_side=256,
        limit=1,
    )
    image, target = dataset[0]
    union_mask = (target["masks"].any(dim=0).float().unsqueeze(0))
    positive = float(union_mask.sum())
    negative = float(union_mask.numel() - positive)
    pos_weight = torch.tensor([max(1.0, min(20.0, negative / max(positive, 1.0)))])

    model = TinyUnionMaskNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    losses = []
    for _step in range(25):
        optimizer.zero_grad(set_to_none=True)
        logits = model(image, union_mask.shape[-2:])
        loss = criterion(logits, union_mask)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    with torch.no_grad():
        prediction = (torch.sigmoid(model(image, union_mask.shape[-2:])) > 0.5).float()
        intersection = float((prediction * union_mask).sum())
        union = float(((prediction + union_mask) > 0).sum())
        iou = intersection / union if union else 1.0

    report = {
        "model": "TinyUnionMaskNet_cpu_smoke_only",
        "image_id": target["image_key"],
        "steps": len(losses),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_decreased": losses[-1] < losses[0],
        "final_union_mask_iou": iou,
        "passed": losses[-1] < losses[0],
        "note": "This validates the data/gradient path, not Mask R-CNN performance.",
    }
    output = ROOT / "reports" / "learning_smoke_test.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
