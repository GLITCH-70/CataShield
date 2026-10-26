"""
metrics.py
==========
Segmentation metrics for Cataract-1K: per-class IoU / Dice, class-group means, pixel accuracy.

Counting:
  Every evaluated frame contributes per-class (TP, FP, FN) pixel counts, stored per frame as an
  array of shape [N, C, 3]. Dataset-level metrics pool these counts over frames (equivalent to a
  global confusion matrix); keeping them per frame lets the analysis bootstrap confidence intervals
  and run paired per-frame tests without re-running inference.

Class sets used for the means:
  - eval classes:  foreground classes that occur in the ground truth of the evaluated split.
                   Ground truth does not change under attack, so clean and attacked runs average
                   over exactly the same classes.
  - headline mIoU: eval classes minus the spec's rare classes (Slit/Incision Knife, Katena
                   Forceps), which are reported separately (spec section 25).
  - anatomy / instruments: the same rule restricted to each group; "instruments_all" keeps the
                   rare instrument classes.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

from cataract_dataset import (
    ANATOMY_CLASSES,
    CLASS_NAMES,
    IGNORE_INDEX,
    INSTRUMENT_CLASSES,
    NUM_CLASSES,
    RARE_CLASSES,
)


@torch.no_grad()
def frame_stats(pred: torch.Tensor, target: torch.Tensor, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """
    Per-frame, per-class (TP, FP, FN) counts.

    pred   : [B, H, W] class ids, or [B, C, H, W] logits
    target : [B, H, W] class ids (IGNORE_INDEX pixels are skipped)
    returns: int64 array [B, num_classes, 3]
    """
    if pred.dim() == 4:
        pred = pred.argmax(1)
    b = pred.shape[0]
    pred = pred.reshape(b, -1).long()
    target = target.reshape(b, -1).long().to(pred.device)
    valid = target != IGNORE_INDEX
    frame_idx = torch.arange(b, device=pred.device).unsqueeze(1).expand_as(pred)
    idx = (frame_idx * num_classes + target) * num_classes + pred
    cm = torch.bincount(idx[valid], minlength=b * num_classes * num_classes)
    cm = cm.reshape(b, num_classes, num_classes)          # [B, true, pred]
    tp = cm.diagonal(dim1=1, dim2=2)
    fp = cm.sum(1) - tp
    fn = cm.sum(2) - tp
    return torch.stack([tp, fp, fn], dim=-1).cpu().numpy().astype(np.int64)


def _nanmean(values: Iterable[float]) -> float:
    vals = [v for v in values if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def class_sets(gt_pixels: np.ndarray) -> Dict[str, List[int]]:
    """Class groups for the means, given per-class ground-truth pixel counts [C]."""
    present = [c for c in range(1, NUM_CLASSES) if gt_pixels[c] > 0]
    return {
        "eval": present,
        "headline": [c for c in present if c not in RARE_CLASSES],
        "anatomy": [c for c in present if c in ANATOMY_CLASSES],
        "instruments": [c for c in present if c in INSTRUMENT_CLASSES and c not in RARE_CLASSES],
        "instruments_all": [c for c in present if c in INSTRUMENT_CLASSES],
        "rare": [c for c in present if c in RARE_CLASSES],
        "absent": [c for c in range(1, NUM_CLASSES) if gt_pixels[c] == 0],
    }


def summarize_counts(counts: np.ndarray, sets: Optional[Dict[str, List[int]]] = None) -> Dict[str, object]:
    """
    counts: pooled [C, 3] (TP, FP, FN) or per-frame [N, C, 3] (pooled here).
    sets:   class groups from class_sets(); derived from these counts if not given.
    """
    if counts.ndim == 3:
        counts = counts.sum(0)
    tp, fp, fn = counts[:, 0].astype(np.float64), counts[:, 1].astype(np.float64), counts[:, 2].astype(np.float64)
    gt_pixels = tp + fn
    sets = sets or class_sets(gt_pixels)

    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(gt_pixels > 0, tp / (tp + fp + fn), np.nan)
        dice = np.where(gt_pixels > 0, 2 * tp / (2 * tp + fp + fn), np.nan)

    out: Dict[str, object] = {
        "per_class_iou": {c: float(iou[c]) for c in range(NUM_CLASSES)},
        "per_class_dice": {c: float(dice[c]) for c in range(NUM_CLASSES)},
        "pixel_accuracy": float(tp.sum() / max(gt_pixels.sum(), 1)),
        "class_sets": sets,
    }
    for group in ("headline", "eval", "anatomy", "instruments", "instruments_all"):
        suffix = "" if group == "headline" else f"_{group}"
        out[f"mIoU{suffix}"] = _nanmean(iou[c] for c in sets[group])
        out[f"mDice{suffix}"] = _nanmean(dice[c] for c in sets[group])
    return out


class SegmentationMetricTracker:
    """Accumulates per-frame (TP, FP, FN) counts across batches."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        self.num_classes = num_classes
        self._stats: List[np.ndarray] = []

    def reset(self) -> None:
        self._stats = []

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
        s = frame_stats(pred, target, self.num_classes)
        self._stats.append(s)
        return s

    @property
    def stats(self) -> np.ndarray:
        if not self._stats:
            return np.zeros((0, self.num_classes, 3), dtype=np.int64)
        return np.concatenate(self._stats, axis=0)

    def summary(self, sets: Optional[Dict[str, List[int]]] = None) -> Dict[str, object]:
        return summarize_counts(self.stats, sets)


def format_report(s: Dict[str, object], title: str = "Segmentation report") -> str:
    sets = s["class_sets"]
    lines = [
        "=" * 78,
        title,
        "=" * 78,
        f"Headline mIoU {s['mIoU'] * 100:6.2f}%   mDice {s['mDice'] * 100:6.2f}%   "
        f"(classes: {', '.join(CLASS_NAMES[c] for c in sets['headline'])})",
        f"  Anatomy      mIoU {s['mIoU_anatomy'] * 100:6.2f}%   mDice {s['mDice_anatomy'] * 100:6.2f}%",
        f"  Instruments  mIoU {s['mIoU_instruments'] * 100:6.2f}%   mDice {s['mDice_instruments'] * 100:6.2f}%"
        f"   (incl. rare: mIoU {s['mIoU_instruments_all'] * 100:6.2f}%)",
        f"  All eval fg  mIoU {s['mIoU_eval'] * 100:6.2f}%   Pixel accuracy {s['pixel_accuracy'] * 100:6.2f}%",
        "-" * 78,
        f"{'ID':>2}  {'Class':<25} {'Group':<11} {'IoU %':>8} {'Dice %':>8}",
        "-" * 78,
    ]
    for c in range(NUM_CLASSES):
        group = "Background" if c == 0 else ("Anatomy" if c in ANATOMY_CLASSES else "Instrument")
        if c in sets["rare"]:
            group += "*"
        iou, dice = s["per_class_iou"][c], s["per_class_dice"][c]
        iou_s = "   n/a" if np.isnan(iou) else f"{iou * 100:8.2f}"
        dice_s = "   n/a" if np.isnan(dice) else f"{dice * 100:8.2f}"
        lines.append(f"{c:2d}  {CLASS_NAMES[c]:<25} {group:<11} {iou_s:>8} {dice_s:>8}")
    lines.append("-" * 78)
    lines.append("* rare class, excluded from headline means. n/a = absent from this split's ground truth.")
    return "\n".join(lines)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    target = torch.from_numpy(rng.integers(0, NUM_CLASSES, size=(4, 64, 64)))
    pred = target.clone()
    noise = torch.from_numpy(rng.random((4, 64, 64)) > 0.85)
    pred[noise] = torch.from_numpy(rng.integers(0, NUM_CLASSES, size=int(noise.sum())))
    tracker = SegmentationMetricTracker()
    tracker.update(pred, target)

    # Cross-check against a direct confusion-matrix computation.
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(cm, (target.numpy().ravel(), pred.numpy().ravel()), 1)
    tp = np.diag(cm)
    ref_iou = tp / (cm.sum(0) + cm.sum(1) - tp)
    s = tracker.summary()
    got = np.array([s["per_class_iou"][c] for c in range(NUM_CLASSES)])
    assert np.allclose(got, ref_iou), "per-class IoU mismatch"
    print(format_report(s, "Synthetic check (85% agreement)"))
    print("[SUCCESS] metrics verified against a reference confusion matrix")
