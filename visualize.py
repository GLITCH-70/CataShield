"""
visualize.py
============
Shared helpers: class colour palette (identical to convert_supervisely_to_masks.py), mask
colourisation, and JSON sanitising.
"""

from __future__ import annotations

import math

import numpy as np

from cataract_dataset import NUM_CLASSES

PALETTE = np.array([
    [0, 0, 0],          # 0  Background
    [34, 180, 216],     # 1  Cornea
    [190, 42, 244],     # 2  Pupil
    [52, 49, 114],      # 3  Lens (IOL)
    [223, 129, 147],    # 4  Slit/Incision Knife
    [252, 229, 85],     # 5  Gauge
    [192, 17, 128],     # 6  Capsulorhexis Cystotome
    [117, 39, 36],      # 7  Capsulorhexis Forceps
    [12, 228, 30],      # 8  Katena Forceps
    [113, 224, 174],    # 9  Spatula
    [212, 92, 58],      # 10 Phacoemulsification Tip
    [171, 10, 7],       # 11 Irrigation-Aspiration
    [140, 190, 14],     # 12 Lens Injector
], dtype=np.uint8)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """[H, W] class ids -> [H, W, 3] uint8 RGB."""
    return PALETTE[np.clip(mask, 0, NUM_CLASSES - 1).astype(np.int64)]


def to_jsonable(obj):
    """Recursively convert numpy types / NaN to plain JSON values (NaN -> None)."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if math.isnan(float(obj)) else float(obj)
    return obj
