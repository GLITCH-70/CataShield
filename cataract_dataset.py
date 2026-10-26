"""
cataract_dataset.py
===================
PyTorch Dataset for Cataract-1K surgical scene segmentation (3 anatomy + 9 instrument classes).

Split protocol (Version 1 spec, section 25):
  - The five official patient-disjoint folds of Ghamsarian et al. (2024), numbered 1-5 as in the
    paper. Fold sizes in annotated frames are 561 / 459 / 420 / 385 / 431, which this module
    reproduces exactly from the Supervisely JSON files.
  - Fold 1 (6 videos) is the held-out test set.
  - The remaining 24 videos are split 20 train / 4 validation. The 4 validation videos are one per
    non-test fold, chosen for instrument-class coverage (fixed before the test fold was evaluated).
    On the full 2,256 frames this gives 1,412 train / 283 val / 561 test, and every foreground
    class occurs in all three splits.

Pixel space:
  Images are returned as float tensors in [0, 1] RGB, NOT ImageNet-normalised. Normalisation lives
  inside the model (model_builder.CataractSegmenter), so the adversarial threat model is defined
  directly in [0, 1] pixel space, as the spec requires.

Storage:
  Only frames whose PNG is present in dataset/ are used (download_synapse_dataset.py fetches them).
  Frames are resized once into uint8 .npy files under cache/ and memory-mapped, so the ~2.4 GB of
  resized frames is paged from disk on demand rather than held in RAM.
"""

from __future__ import annotations

import glob
import hashlib
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset


# ============================================================================
# Official Cataract-1K 5-fold patient-disjoint partition (paper numbering, 1-5)
# ============================================================================
OFFICIAL_FOLDS: Dict[int, List[str]] = {
    1: ["case5013", "case5014", "case5057", "case5058", "case5104", "case5334"],
    2: ["case5063", "case5299", "case5316", "case5317", "case5319", "case5335"],
    3: ["case5017", "case5032", "case5301", "case5309", "case5325", "case_53153"],
    4: ["case5015", "case5300", "case5303", "case5304", "case5305", "case5329"],
    5: ["case5016", "case5051", "case5072", "case5180", "case5340", "case5353"],
}
# Annotated-frame counts per fold reported in the Cataract-1K paper; checked by verify_fold_sizes().
PAPER_FOLD_SIZES = {1: 561, 2: 459, 3: 420, 4: 385, 5: 431}

TEST_FOLD = 1
# One validation video from each non-test fold (folds 2, 3, 4, 5).
DEFAULT_VAL_CASES = ["case5317", "case5325", "case5300", "case5072"]

NUM_CLASSES = 13  # 0 = background, 1-3 anatomy, 4-12 instruments
IGNORE_INDEX = 255  # used only for padding introduced by geometric augmentation

CLASS_NAMES = {
    0:  "Background",
    1:  "Cornea",
    2:  "Pupil",
    3:  "Lens (IOL)",
    4:  "Slit/Incision Knife",
    5:  "Gauge",
    6:  "Capsulorhexis Cystotome",
    7:  "Capsulorhexis Forceps",
    8:  "Katena Forceps",
    9:  "Spatula",
    10: "Phacoemulsification Tip",
    11: "Irrigation-Aspiration",
    12: "Lens Injector",
}
ANATOMY_CLASSES = [1, 2, 3]
INSTRUMENT_CLASSES = [4, 5, 6, 7, 8, 9, 10, 11, 12]
# Rare classes named in the spec (Katena forceps 1.29%, slit knife 2.22% of frames); reported
# separately from the headline means.
RARE_CLASSES = [4, 8]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def extract_case_id(filename: str) -> str:
    """'case_53153_01.png' -> 'case_53153', 'case5013_01.png' -> 'case5013'."""
    base = os.path.basename(filename)
    match = re.match(r"(case_?\d+)_\d+\.png", base)
    if not match:
        raise ValueError(f"Cannot extract case ID from filename: {filename}")
    return match.group(1)


def get_split_cases(
    test_fold: int = TEST_FOLD,
    val_cases: Sequence[str] = DEFAULT_VAL_CASES,
) -> Dict[str, List[str]]:
    """Return {'train': [...], 'val': [...], 'test': [...]} patient-case lists."""
    test_cases = list(OFFICIAL_FOLDS[test_fold])
    val_cases = list(val_cases)
    train_cases = [
        c for f, cases in OFFICIAL_FOLDS.items() if f != test_fold
        for c in cases if c not in val_cases
    ]
    for c in val_cases:
        assert c not in test_cases, f"Validation case {c} is in the test fold"
        assert any(c in cases for cases in OFFICIAL_FOLDS.values()), f"Unknown case {c}"
    verify_patient_disjointness(train_cases, val_cases, test_cases)
    return {"train": train_cases, "val": val_cases, "test": test_cases}


def verify_patient_disjointness(train_cases, val_cases, test_cases) -> None:
    """Assert that no patient video appears in more than one split."""
    s_train, s_val, s_test = set(train_cases), set(val_cases), set(test_cases)
    assert not s_train & s_val, f"LEAKAGE between train and val: {s_train & s_val}"
    assert not s_train & s_test, f"LEAKAGE between train and test: {s_train & s_test}"
    assert not s_val & s_test, f"LEAKAGE between val and test: {s_val & s_test}"


def verify_fold_sizes(dataset_dir: str) -> Dict[int, int]:
    """Count annotated JSON frames per fold and assert they match the paper."""
    counts = {}
    for fold, cases in OFFICIAL_FOLDS.items():
        counts[fold] = sum(
            len(glob.glob(os.path.join(dataset_dir, f"{c}_*.png.json"))) for c in cases
        )
    for fold, n in counts.items():
        assert n == PAPER_FOLD_SIZES[fold], (
            f"Fold {fold} has {n} annotated frames, paper reports {PAPER_FOLD_SIZES[fold]}"
        )
    return counts


class _TensorView:
    """Indexable view over a (memory-mapped) uint8 array that returns torch tensors."""

    def __init__(self, array: np.ndarray):
        self.array = array

    def __len__(self) -> int:
        return len(self.array)

    @property
    def shape(self):
        return self.array.shape

    def __getitem__(self, idx) -> torch.Tensor:
        if isinstance(idx, (list, tuple)):
            idx = np.asarray(idx)
        # np.array copies out of the read-only memmap, so callers get an ordinary writable tensor.
        return torch.from_numpy(np.array(self.array[idx]))

    def __iter__(self):
        for i in range(len(self.array)):
            yield self[i]


class CataractDataset(Dataset):
    """
    Cataract-1K segmentation frames for one split, cached on disk (memory-mapped) at the target resolution.

    Each item is a dict with
        image      float32 [3, H, W] in [0, 1]
        mask       int64   [H, W] with values 0-12
        case_id    str
        frame_name str
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        test_fold: int = TEST_FOLD,
        val_cases: Sequence[str] = DEFAULT_VAL_CASES,
        img_size: Tuple[int, int] = (512, 512),
    ):
        super().__init__()
        self.root_dir = os.path.abspath(root_dir)
        self.images_dir = os.path.join(self.root_dir, "dataset")
        self.masks_dir = os.path.join(self.root_dir, "masks")
        self.split = split.lower()
        self.img_size = img_size

        splits = get_split_cases(test_fold, val_cases)
        if self.split == "all":
            target = splits["train"] + splits["val"] + splits["test"]
        elif self.split in splits:
            target = splits[self.split]
        else:
            raise ValueError(f"Unknown split '{split}'")
        self.target_cases = sorted(target)

        self.samples: List[Tuple[str, str, str]] = []
        for img_path in sorted(glob.glob(os.path.join(self.images_dir, "*.png"))):
            name = os.path.basename(img_path)
            case_id = extract_case_id(name)
            mask_path = os.path.join(self.masks_dir, name)
            if case_id in self.target_cases and os.path.isfile(mask_path):
                self.samples.append((img_path, mask_path, case_id))
        if not self.samples:
            raise RuntimeError(f"No image/mask pairs found for split '{self.split}'")

        # Resize once into uint8 .npy files under cache/, then memory-map them. The full dataset at
        # 512x512 is ~2.4 GB, so frames are paged in from disk on demand instead of held in RAM.
        images, masks = self._load_or_build_cache()
        self.images = _TensorView(images)
        self.masks = _TensorView(masks)

    def _load_or_build_cache(self):
        h, w = self.img_size
        names = "|".join(os.path.basename(s[0]) for s in self.samples)
        key = hashlib.sha1(f"{names}|{h}x{w}".encode()).hexdigest()[:12]
        cache_dir = os.path.join(self.root_dir, "cache")
        img_path = os.path.join(cache_dir, f"{self.split}_{key}_images.npy")
        mask_path = os.path.join(cache_dir, f"{self.split}_{key}_masks.npy")
        n = len(self.samples)
        if not (os.path.isfile(img_path) and os.path.isfile(mask_path)):
            os.makedirs(cache_dir, exist_ok=True)
            tmp_img, tmp_mask = img_path + ".tmp", mask_path + ".tmp"
            images = np.lib.format.open_memmap(tmp_img, mode="w+", dtype=np.uint8, shape=(n, 3, h, w))
            masks = np.lib.format.open_memmap(tmp_mask, mode="w+", dtype=np.uint8, shape=(n, h, w))
            for i, (img_file, mask_file, _) in enumerate(self.samples):
                img = Image.open(img_file).convert("RGB").resize((w, h), Image.BILINEAR)
                mask = np.asarray(Image.open(mask_file).resize((w, h), Image.NEAREST))
                assert mask.max() < NUM_CLASSES, f"{mask_file}: out-of-range class id"
                images[i] = np.asarray(img).transpose(2, 0, 1)
                masks[i] = mask
            images.flush(); masks.flush()
            del images, masks
            os.replace(tmp_img, img_path)
            os.replace(tmp_mask, mask_path)
        images = np.load(img_path, mmap_mode="r")
        masks = np.load(mask_path, mmap_mode="r")
        assert images.shape == (n, 3, h, w) and masks.shape == (n, h, w), "stale cache; delete cache/"
        return images, masks

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return {
            "image": self.images[idx].float().div_(255.0),
            "mask": self.masks[idx].long(),
            "case_id": self.samples[idx][2],
            "frame_name": os.path.basename(self.samples[idx][0]),
        }

    def class_frame_counts(self) -> np.ndarray:
        """Number of frames in which each class appears, shape [NUM_CLASSES]."""
        counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        for m in self.masks:
            counts += (torch.bincount(m.flatten().long(), minlength=NUM_CLASSES) > 0).numpy()
        return counts


def get_cataract_dataloaders(
    root_dir: str,
    batch_size: int = 8,
    eval_batch_size: Optional[int] = None,
    img_size: Tuple[int, int] = (512, 512),
    test_fold: int = TEST_FOLD,
    val_cases: Sequence[str] = DEFAULT_VAL_CASES,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """(train_loader, val_loader, test_loader). Augmentation is applied on the GPU by the trainer."""
    eval_batch_size = eval_batch_size or batch_size
    pin = torch.cuda.is_available()
    kw = dict(test_fold=test_fold, val_cases=val_cases, img_size=img_size)
    train_ds = CataractDataset(root_dir, "train", **kw)
    val_ds = CataractDataset(root_dir, "val", **kw)
    test_ds = CataractDataset(root_dir, "test", **kw)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False,
                            num_workers=0, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False,
                             num_workers=0, pin_memory=pin)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("=" * 78)
    print("Cataract-1K dataset & patient-disjoint split validator")
    print("=" * 78)

    counts = verify_fold_sizes(os.path.join(base_dir, "dataset"))
    print("[PASSED] Annotated frames per fold match the paper:", counts)

    splits = get_split_cases()
    print("[PASSED] Zero patient leakage across train / val / test")
    for name in ("train", "val", "test"):
        ds = CataractDataset(base_dir, name)
        fc = ds.class_frame_counts()
        print(f"\n{name.upper():5s}: {len(ds):4d} frames from {len(splits[name])} videos {splits[name]}")
        print("  frames per class: " + ", ".join(
            f"{CLASS_NAMES[c]}={fc[c]}" for c in range(1, NUM_CLASSES)))
    s = ds[0]
    print(f"\nSample {s['frame_name']}: image {tuple(s['image'].shape)} "
          f"range [{s['image'].min():.3f}, {s['image'].max():.3f}], mask {tuple(s['mask'].shape)} "
          f"classes {s['mask'].unique().tolist()}")
