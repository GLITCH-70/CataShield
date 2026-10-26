"""
train_baseline.py
=================
Clean-baseline training for DeepLabV3+ (ResNet-50, OS16) on Cataract-1K.

Protocol
  - Split: official Fold 1 = test (never seen during training or model selection);
    20 videos train / 4 videos validation from the remaining folds (cataract_dataset.py).
  - Loss: CE + 0.8 * soft Dice.
  - Optimiser: AdamW, encoder lr 1e-4, decoder lr 5e-4, weight decay 1e-4,
    1-epoch linear warm-up then cosine decay (per iteration).
  - Augmentation (GPU, train only): horizontal / vertical flip, random affine
    (scale 0.75-1.25, rotation +/-15 deg, shift +/-10%), brightness / contrast / saturation
    jitter, occasional Gaussian blur. Pixels moved in from outside the frame are ignored by the loss.
  - Mixed precision (bf16 autocast) for training; all evaluation runs in fp32.
  - Model selection: best validation headline mIoU. The test fold is evaluated once, at the end.

Outputs (runs/baseline/)
  best_model.pth, history.csv, training_curves.png, clean_metrics.json, clean_report.txt,
  qualitative_test_panel.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from cataract_dataset import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES, get_cataract_dataloaders
from loss import CataractCombinedLoss
from metrics import SegmentationMetricTracker, class_sets, format_report
from model_builder import build_cataract_model, count_parameters, load_trained_model
from visualize import colorize_mask, to_jsonable

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# GPU augmentation
# ----------------------------------------------------------------------------
def augment_batch(images: torch.Tensor, masks: torch.Tensor):
    """images [B,3,H,W] in [0,1], masks [B,H,W] long -> augmented copies (same device)."""
    b, _, h, w = images.shape
    dev = images.device

    # Random affine: flips, scale, rotation, translation (output -> input sampling grid).
    scale = torch.empty(b, device=dev).uniform_(0.75, 1.25)
    angle = torch.empty(b, device=dev).uniform_(-15, 15) * math.pi / 180
    shift = torch.empty(b, 2, device=dev).uniform_(-0.1, 0.1)
    hflip = torch.where(torch.rand(b, device=dev) < 0.5, -1.0, 1.0)
    vflip = torch.where(torch.rand(b, device=dev) < 0.25, -1.0, 1.0)
    # theta = [R(angle) / scale] @ diag(hflip, vflip), plus translation.
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([
        torch.stack([cos * hflip, -sin * vflip, shift[:, 0]], dim=1),
        torch.stack([sin * hflip, cos * vflip, shift[:, 1]], dim=1),
    ], dim=1)
    grid = F.affine_grid(theta, (b, 3, h, w), align_corners=False)
    images = F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    shifted = (masks.float() + 1.0).unsqueeze(1)
    shifted = F.grid_sample(shifted, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    shifted = shifted.squeeze(1).round().long()
    masks = torch.where(shifted == 0, torch.full_like(shifted, IGNORE_INDEX), shifted - 1)

    # Photometric jitter.
    bright = torch.empty(b, 1, 1, 1, device=dev).uniform_(0.75, 1.25)
    contrast = torch.empty(b, 1, 1, 1, device=dev).uniform_(0.75, 1.25)
    sat = torch.empty(b, 1, 1, 1, device=dev).uniform_(0.8, 1.2)
    images = images * bright
    mean = images.mean(dim=(1, 2, 3), keepdim=True)
    images = (images - mean) * contrast + mean
    gray = (0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3])
    images = gray + (images - gray) * sat

    # Occasional Gaussian blur (motion / focus blur is common in surgical video).
    blur = torch.rand(b, device=dev) < 0.2
    if blur.any():
        k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=dev)
        k = (k[:, None] * k[None, :]) / 256.0
        k = k.expand(3, 1, 5, 5)
        blurred = F.conv2d(F.pad(images[blur], (2, 2, 2, 2), mode="reflect"), k, groups=3)
        images = images.clone()
        images[blur] = blurred
    return images.clamp_(0.0, 1.0), masks


# ----------------------------------------------------------------------------
# Train / evaluate
# ----------------------------------------------------------------------------
def lr_factor(it: int, warmup_iters: int, total_iters: int, min_factor: float = 0.01) -> float:
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    progress = (it - warmup_iters) / max(1, total_iters - warmup_iters)
    return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, device, sets=None, criterion=None) -> Dict[str, object]:
    """fp32 evaluation; returns summary dict plus per-frame stats ('stats') and mean 'loss'."""
    model.eval()
    tracker = SegmentationMetricTracker()
    loss_sum, n = 0.0, 0
    for batch in loader:
        logits = model(batch["image"].to(device, non_blocking=True))
        masks = batch["mask"].to(device)
        tracker.update(logits.argmax(1), masks)
        if criterion is not None:
            loss_sum += float(criterion(logits, masks)[0]) * masks.shape[0]
            n += masks.shape[0]
    summary = tracker.summary(sets)
    summary["stats"] = tracker.stats
    summary["loss"] = loss_sum / max(n, 1)
    return summary


def save_qualitative_panel(model, loader, device, path: str, n: int = 4) -> None:
    """Frame | ground truth | prediction | overlay, for n test frames spread across the split."""
    ds = loader.dataset
    idxs = np.linspace(0, len(ds) - 1, n).astype(int)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    model.eval()
    for row, i in enumerate(idxs):
        s = ds[i]
        with torch.no_grad():
            pred = model(s["image"][None].to(device)).argmax(1)[0].cpu().numpy()
        img = s["image"].permute(1, 2, 0).numpy()
        gt_c, pred_c = colorize_mask(s["mask"].numpy()), colorize_mask(pred)
        overlay = img.copy()
        fg = pred > 0
        overlay[fg] = 0.5 * overlay[fg] + 0.5 * pred_c[fg] / 255.0
        for col, (im, title) in enumerate([(img, "Frame"), (gt_c, "Ground truth"),
                                           (pred_c, "Prediction"), (overlay, "Overlay")]):
            axes[row, col].imshow(im)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(title, fontsize=13)
        axes[row, 0].set_ylabel(s["frame_name"])
        axes[row, 0].text(5, 25, s["frame_name"], color="white", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_history(history: List[Dict[str, float]], path: str) -> None:
    ep = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(ep, [h["train_loss"] for h in history], label="train loss (CE + 0.8 Dice)")
    a1.plot(ep, [h["val_loss"] for h in history], label="val loss")
    a1.set_xlabel("epoch"); a1.set_ylabel("loss"); a1.legend(); a1.grid(alpha=0.3)
    a2.plot(ep, [100 * h["val_mIoU"] for h in history], label="val headline mIoU")
    a2.plot(ep, [100 * h["val_mIoU_anatomy"] for h in history], label="val anatomy mIoU")
    a2.plot(ep, [100 * h["val_mIoU_instruments"] for h in history], label="val instrument mIoU")
    a2.set_xlabel("epoch"); a2.set_ylabel("mIoU (%)"); a2.legend(); a2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train DeepLabV3+ (ResNet-50) on Cataract-1K.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr-encoder", type=float, default=1e-4)
    parser.add_argument("--lr-decoder", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iters-per-epoch", type=int, default=None, help="for quick smoke tests")
    parser.add_argument("--save-dir", type=str, default="runs/baseline")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    save_dir = os.path.join(BASE_DIR, args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 78)
    print("Cataract-1K | DeepLabV3+ (ResNet-50, OS16) clean-baseline training")
    print("=" * 78)
    print(f"device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    print(f"args:   {vars(args)}")

    train_loader, val_loader, test_loader = get_cataract_dataloaders(
        BASE_DIR, batch_size=args.batch_size, eval_batch_size=8)
    print(f"train {len(train_loader.dataset)} | val {len(val_loader.dataset)} | "
          f"test {len(test_loader.dataset)} frames")

    model = build_cataract_model(pretrained=True, device=device).to(memory_format=torch.channels_last)
    print(f"parameters: {count_parameters(model)[0] / 1e6:.2f}M")
    criterion = CataractCombinedLoss(lambda_dice=0.8)
    optimizer = torch.optim.AdamW([
        {"params": list(model.net.encoder_parameters()), "lr": args.lr_encoder},
        {"params": list(model.net.decoder_parameters()), "lr": args.lr_decoder},
    ], weight_decay=args.weight_decay)
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    iters_per_epoch = len(train_loader)
    if args.max_iters_per_epoch:
        iters_per_epoch = min(iters_per_epoch, args.max_iters_per_epoch)
    total_iters = args.epochs * iters_per_epoch
    warmup_iters = iters_per_epoch

    # Class groups for validation means are fixed from the validation ground truth.
    val_gt = np.zeros(NUM_CLASSES)
    for m in val_loader.dataset.masks:
        val_gt += np.bincount(m.flatten().numpy(), minlength=NUM_CLASSES)
    val_sets = class_sets(val_gt)

    history: List[Dict[str, float]] = []
    best_miou, best_epoch, it = -1.0, -1, 0
    ckpt_path = os.path.join(save_dir, "best_model.pth")
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        sums = {"loss_total": 0.0, "loss_ce": 0.0, "loss_dice": 0.0}
        n_batches = 0
        for b_idx, batch in enumerate(train_loader):
            if b_idx >= iters_per_epoch:
                break
            factor = lr_factor(it, warmup_iters, total_iters)
            for g, base in zip(optimizer.param_groups, base_lrs):
                g["lr"] = base * factor

            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            images, masks = augment_batch(images, masks)
            images = images.contiguous(memory_format=torch.channels_last)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(images)
            loss, parts = criterion(logits.float(), masks)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            for k in sums:
                sums[k] += parts[k]
            n_batches += 1
            it += 1

        val = evaluate(model, val_loader, device, val_sets, criterion)
        vloss = val["loss"]
        record = {
            "epoch": epoch,
            "lr_encoder": optimizer.param_groups[0]["lr"],
            "train_loss": sums["loss_total"] / n_batches,
            "train_ce": sums["loss_ce"] / n_batches,
            "train_dice": sums["loss_dice"] / n_batches,
            "val_loss": vloss,
            "val_mIoU": val["mIoU"],
            "val_mDice": val["mDice"],
            "val_mIoU_anatomy": val["mIoU_anatomy"],
            "val_mIoU_instruments": val["mIoU_instruments"],
            "epoch_seconds": time.time() - t0,
        }
        history.append(record)
        is_best = val["mIoU"] > best_miou
        if is_best:
            best_miou, best_epoch = val["mIoU"], epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_mIoU": best_miou,
                "num_classes": NUM_CLASSES,
                "args": vars(args),
            }, ckpt_path)
        mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0
        print(f"epoch {epoch:3d}/{args.epochs} {record['epoch_seconds']:5.1f}s | "
              f"train {record['train_loss']:.4f} (CE {record['train_ce']:.4f}, Dice {record['train_dice']:.4f}) | "
              f"val {vloss:.4f} | val mIoU {100 * val['mIoU']:5.2f} "
              f"(anat {100 * val['mIoU_anatomy']:5.2f}, tools {100 * val['mIoU_instruments']:5.2f}) "
              f"| peak mem {mem:.2f} GB{'  *best*' if is_best else ''}", flush=True)

        with open(os.path.join(save_dir, "history.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    print(f"\ntraining finished in {(time.time() - t_start) / 60:.1f} min; "
          f"best val mIoU {100 * best_miou:.2f}% at epoch {best_epoch}")
    plot_history(history, os.path.join(save_dir, "training_curves.png"))

    # ---- Final clean evaluation of the selected checkpoint (val + held-out test fold) ----
    model = load_trained_model(ckpt_path, device)
    results = {"best_epoch": best_epoch}
    reports = []
    for name, loader in (("val", val_loader), ("test", test_loader)):
        s = evaluate(model, loader, device)
        s.pop("stats")
        s.pop("loss")
        results[name] = s
        reports.append(format_report(s, f"Clean {name} split (best epoch {best_epoch})"))
    report = "\n\n".join(reports)
    print("\n" + report)
    with open(os.path.join(save_dir, "clean_report.txt"), "w") as f:
        f.write(report + "\n")
    with open(os.path.join(save_dir, "clean_metrics.json"), "w") as f:
        json.dump(to_jsonable(results), f, indent=2)
    save_qualitative_panel(model, test_loader, device,
                           os.path.join(save_dir, "qualitative_test_panel.png"))
    print(f"\nartefacts written to {save_dir}")


if __name__ == "__main__":
    main()
