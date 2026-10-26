"""
loss.py
=======
Combined cross-entropy + soft Dice loss for Cataract-1K semantic segmentation.

    L_total = L_CE + lambda_dice * L_Dice,   lambda_dice = 0.8 (Cataract-1K benchmark convention)

Pixels labelled IGNORE_INDEX (padding introduced by geometric augmentation) are excluded from both
terms.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cataract_dataset import IGNORE_INDEX, NUM_CLASSES


class SoftDiceLoss(nn.Module):
    """
    Multi-class soft Dice loss. Intersection and cardinality are pooled over the whole batch, so a
    class contributes a meaningful gradient whenever it appears anywhere in the batch.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        smooth: float = 1.0,
        ignore_index: int = IGNORE_INDEX,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits.float(), dim=1)                       # [B, C, H, W]
        valid = (targets != self.ignore_index).unsqueeze(1).float()     # [B, 1, H, W]
        safe_targets = torch.where(targets == self.ignore_index, torch.zeros_like(targets), targets)
        one_hot = F.one_hot(safe_targets, self.num_classes).permute(0, 3, 1, 2).float()

        probs = probs * valid
        one_hot = one_hot * valid
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        cardinality = (probs + one_hot).sum(dims)
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (cardinality + self.smooth)  # [C]

        if self.class_weights is not None:
            w = self.class_weights.to(dice_loss.device)
            return (dice_loss * w).sum() / w.sum()
        return dice_loss.mean()


class CataractCombinedLoss(nn.Module):
    """L_total = L_CE + lambda_dice * L_Dice."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        lambda_dice: float = 0.8,
        smooth: float = 1.0,
        ignore_index: int = IGNORE_INDEX,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.lambda_dice = lambda_dice
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.dice_loss = SoftDiceLoss(num_classes, smooth, ignore_index, class_weights)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        ce = self.ce_loss(logits.float(), targets)
        dice = self.dice_loss(logits, targets)
        total = ce + self.lambda_dice * dice
        return total, {
            "loss_total": float(total.detach()),
            "loss_ce": float(ce.detach()),
            "loss_dice": float(dice.detach()),
        }


if __name__ == "__main__":
    criterion = CataractCombinedLoss()
    logits = torch.randn(2, NUM_CLASSES, 64, 64, requires_grad=True)
    targets = torch.randint(0, NUM_CLASSES, (2, 64, 64))
    targets[:, :8] = IGNORE_INDEX
    loss, parts = criterion(logits, targets)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[:, :, :8].abs().max() == 0, "ignored pixels must receive no gradient"
    print("[SUCCESS] loss forward/backward and ignore-index masking verified:", parts)
