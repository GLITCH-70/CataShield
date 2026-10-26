"""
convert_supervisely_to_masks.py
===============================
Converts the Supervisely polygon annotations of the Cataract-1K segmentation subset
(Synapse syn53395479) into dense single-channel label masks for semantic segmentation.

Class mapping (13 ids = background + 3 anatomy + 9 instruments):
     0  Background
     1  Cornea                    (Supervisely labels "Cornea" and "cornea1"; called "iris" in the paper)
     2  Pupil                     ("Pupil" and "pupil1")
     3  Lens (IOL)
     4  Slit/Incision Knife       ("Slit Knife" + "Incision Knife", merged as in the Cataract-1K paper,
                                   which lists 9 instrument classes; 28 + 22 = 50 frames, matching the
                                   paper's "slit knife: 50 frames")
     5  Gauge
     6  Capsulorhexis Cystotome
     7  Capsulorhexis Forceps
     8  Katena Forceps
     9  Spatula
    10  Phacoemulsification Tip
    11  Irrigation-Aspiration
    12  Lens Injector

Rasterisation:
    Polygons are filled in z-order: anatomy first (cornea, then pupil, then lens), instruments last,
    so an instrument occluding the eye keeps its own label. All 7,585 annotation objects are
    hole-free polygons on 1024x768 frames.

Usage:
    python convert_supervisely_to_masks.py

Output:
    masks/          uint8 PNG label masks, values 0-12, one per annotation JSON
    masks_colored/  RGB visualisations of the same masks
    Class frequency statistics printed to stdout.
"""

import json
import os
import glob
import sys
from collections import Counter
from PIL import Image, ImageDraw
import numpy as np

# ============================================================================
# Class mapping: Supervisely classTitle -> integer class ID
# ============================================================================
# Z-order priority: higher number = drawn later (on top).
# Anatomy classes are background layers; instruments are foreground.

CLASS_MAP = {
    # Anatomy (drawn first — background layers)
    "Cornea":   1,
    "cornea1":  1,   # variant label, same semantic class
    "Pupil":    2,
    "pupil1":   2,   # variant label, same semantic class
    "Lens":     3,
    # Instruments (drawn on top of anatomy)
    "Slit Knife":              4,
    "Incision Knife":          4,   # merged with Slit Knife per Cataract-1K paper
    "Gauge":                   5,
    "Capsulorhexis Cystotome": 6,
    "Capsulorhexis Forceps":   7,
    "Katena Forceps":          8,
    "Spatula":                 9,
    "Phacoemulsification Tip": 10,
    "Irrigation-Aspiration":   11,
    "Lens Injector":           12,
}

# Canonical class names for reporting (class ID -> human-readable name)
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

NUM_CLASSES = 13  # 0-12 inclusive

# RGB palette for colored visualization masks
CLASS_COLORS = {
    0:  (0,   0,   0),      # Background — black
    1:  (34,  180, 216),     # Cornea — cyan
    2:  (190, 42,  244),     # Pupil — purple
    3:  (52,  49,  114),     # Lens — dark blue
    4:  (223, 129, 147),     # Slit/Incision Knife — pink
    5:  (252, 229, 85),      # Gauge — yellow
    6:  (192, 17,  128),     # Capsulorhexis Cystotome — magenta
    7:  (117, 39,  36),      # Capsulorhexis Forceps — dark red
    8:  (12,  228, 30),      # Katena Forceps — green
    9:  (113, 224, 174),     # Spatula — mint
    10: (212, 92,  58),      # Phacoemulsification Tip — orange
    11: (171, 10,  7),       # Irrigation-Aspiration — red
    12: (140, 190, 14),      # Lens Injector — lime
}

# Z-order: anatomy first, then instruments.
# Objects are drawn in this order; later objects overwrite earlier ones.
Z_ORDER = {
    1: 0,   # Cornea (background anatomy)
    2: 1,   # Pupil (inside cornea)
    3: 2,   # Lens (inside pupil)
    4: 3,   # Slit/Incision Knife
    5: 3,   # Gauge
    6: 3,   # Capsulorhexis Cystotome
    7: 3,   # Capsulorhexis Forceps
    8: 3,   # Katena Forceps
    9: 3,   # Spatula
    10: 3,  # Phacoemulsification Tip
    11: 3,  # Irrigation-Aspiration
    12: 3,  # Lens Injector
}


def convert_annotation(json_path, mask_dir, colored_dir):
    """
    Convert a single Supervisely annotation JSON to a label mask PNG.

    Parameters
    ----------
    json_path : str
        Path to the .png.json annotation file.
    mask_dir : str
        Output directory for single-channel label masks.
    colored_dir : str
        Output directory for RGB colored visualization masks.

    Returns
    -------
    dict
        Per-class pixel counts for this frame.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    height = data["size"]["height"]
    width = data["size"]["width"]

    # Create blank mask (all background = 0)
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    # Sort objects by z-order so anatomy is drawn first, instruments on top
    objects = data.get("objects", [])
    sorted_objects = sorted(objects, key=lambda obj: Z_ORDER.get(
        CLASS_MAP.get(obj.get("classTitle", ""), 0), 99
    ))

    skipped_classes = set()
    for obj in sorted_objects:
        class_title = obj.get("classTitle", "")
        class_id = CLASS_MAP.get(class_title)

        if class_id is None:
            skipped_classes.add(class_title)
            continue

        geometry_type = obj.get("geometryType", "")
        if geometry_type != "polygon":
            continue

        points = obj.get("points", {})
        exterior = points.get("exterior", [])
        if len(exterior) < 3:
            continue

        # Convert [[x, y], ...] to [(x, y), ...] for PIL
        polygon = [(int(p[0]), int(p[1])) for p in exterior]
        draw.polygon(polygon, fill=class_id)

    if skipped_classes:
        print(f"  WARNING: Skipped unknown classes in {os.path.basename(json_path)}: {skipped_classes}")

    # Save single-channel label mask
    base_name = os.path.basename(json_path).replace(".png.json", ".png")
    mask.save(os.path.join(mask_dir, base_name))

    # Create and save colored visualization
    mask_np = np.array(mask)
    colored = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        colored[mask_np == class_id] = color
    colored_img = Image.fromarray(colored)
    colored_img.save(os.path.join(colored_dir, base_name))

    # Count pixels per class
    unique, counts = np.unique(mask_np, return_counts=True)
    pixel_counts = {int(u): int(c) for u, c in zip(unique, counts)}
    return pixel_counts


def main():
    # Paths
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    mask_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "masks")
    colored_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "masks_colored")

    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(colored_dir, exist_ok=True)

    # Find all annotation JSONs
    json_files = sorted(glob.glob(os.path.join(dataset_dir, "*.png.json")))
    if not json_files:
        print("ERROR: No .png.json files found in", dataset_dir)
        sys.exit(1)

    print(f"Found {len(json_files)} annotation files in {dataset_dir}")
    print(f"Output masks:   {mask_dir}")
    print(f"Output colored: {colored_dir}")
    print(f"Number of classes: {NUM_CLASSES} (IDs 0-{NUM_CLASSES - 1})")
    print()

    # Print class mapping
    print("=== Class Mapping ===")
    for class_id in range(NUM_CLASSES):
        print(f"  {class_id:2d} = {CLASS_NAMES[class_id]}")
    print()

    # Convert all annotations
    total_pixel_counts = Counter()
    total_frame_class_presence = Counter()

    for i, jf in enumerate(json_files):
        if (i + 1) % 100 == 0 or i == 0 or i == len(json_files) - 1:
            print(f"  Processing {i + 1}/{len(json_files)}: {os.path.basename(jf)}")

        pixel_counts = convert_annotation(jf, mask_dir, colored_dir)

        for class_id, count in pixel_counts.items():
            total_pixel_counts[class_id] += count
            total_frame_class_presence[class_id] += 1

    # Report statistics
    total_pixels = sum(total_pixel_counts.values())
    print()
    print("=== Class Frequency Statistics ===")
    print(f"{'ID':>3s}  {'Class Name':<25s}  {'Pixel Count':>12s}  {'% Pixels':>8s}  {'Frames Present':>14s}  {'% Frames':>8s}")
    print("-" * 80)
    for class_id in range(NUM_CLASSES):
        px = total_pixel_counts.get(class_id, 0)
        fr = total_frame_class_presence.get(class_id, 0)
        print(f"{class_id:3d}  {CLASS_NAMES[class_id]:<25s}  {px:12,d}  {100.0 * px / total_pixels:7.2f}%  {fr:14d}  {100.0 * fr / len(json_files):7.2f}%")

    print()
    print(f"Total pixels across all masks: {total_pixels:,d}")
    print(f"Total masks generated: {len(json_files)}")
    print()
    print("DONE. Masks saved to:", mask_dir)
    print("Colored visualizations saved to:", colored_dir)


if __name__ == "__main__":
    main()
