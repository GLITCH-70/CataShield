# Adversarial Attack and Robustness Analysis of AI-Based Cataract Surgical Instrument Segmentation

B.Tech CSE Micro-Project 2026, Bharati Vidyapeeth's College of Engineering (GGSIPU), New Delhi.
Guide: Dr. Jolly Parikh. Team: Parikshat Singh, Venu Kulshreshtha, Sayyad Ahsan.

**Version 1 scope:** train a clean segmentation baseline on Cataract-1K, attack it with white-box
L∞ perturbations (FGSM, PGD-20, SegPGD-20), and measure the degradation. No defences (those are
Version 2).

Results are in [`results/RESULTS_TABLES.md`](results/RESULTS_TABLES.md) (auto-generated) and
[`Results_and_Verification.md`](Results_and_Verification.md) (written analysis and problem-statement
check).

---

## Pipeline

```
dataset/ (Supervisely JSON + PNG frames)
   │  convert_supervisely_to_masks.py     → masks/ (13-class label PNGs)
   ▼
cataract_dataset.py   official 5 patient-disjoint folds; Fold 1 = test; 20 train / 4 val videos
model_builder.py      DeepLabV3+ (ResNet-50, OS16), ImageNet normalisation inside the model
loss.py, metrics.py   CE + 0.8·Dice; per-class IoU/Dice, per-frame TP/FP/FN
   │  train_baseline.py                  → runs/baseline/best_model.pth, clean metrics, curves
   ▼
attacks.py            FGSM, PGD, SegPGD, random-noise control (L∞ in [0,1] pixel space)
   │  run_experiments.py                 → results/raw/*.npz (29 runs)
   ▼
   │  analyze_results.py                 → results/tables, results/figures, RESULTS_TABLES.md
```

## Reproduce

```bash
pip install -r requirements.txt
```

1. Get the Cataract-1K segmentation subset (Synapse `syn53395479`) into `dataset/`: the Supervisely
   JSONs plus all 2,256 frame PNGs (`Annotations/Images-and-Supervisely-Annotations/<case>/img`). With a
   view+download token stored in `~/.synapseConfig`, this fetches any frames still missing (it needs
   `pip install synapseclient`, ideally in a separate virtual environment):

```bash
python download_synapse_dataset.py
```
2. Build the label masks, then check the split and fold sizes:

```bash
python convert_supervisely_to_masks.py
```

```bash
python cataract_dataset.py
```

3. Train the clean baseline (≈1 h on an RTX 3060 Laptop GPU):

```bash
python train_baseline.py --epochs 40
```

4. Run the clean + adversarial grid (41 runs on 561 test frames, ≈5 h; resumable, finished runs are skipped):

```bash
python run_experiments.py
```

5. Build tables and figures (CPU, ~1 min):

```bash
python analyze_results.py
```

Each module also has a self-test: `python loss.py`, `python metrics.py`, `python model_builder.py`,
`python attacks.py`.

## Protocol summary

| Item | Setting |
|---|---|
| Data | Cataract-1K segmentation subset: all 2,256 annotated frames (30 videos), resized to 512×512 and cached in `cache/` |
| Classes | 13 = background + Cornea/iris, Pupil, Lens + 9 instruments (Slit and Incision knife merged, as in the paper) |
| Split | Official patient-disjoint Fold 1 (6 videos, 561 frames) = test; 20 videos train (1,412), 4 videos val (283) |
| Model | DeepLabV3+, ImageNet ResNet-50, output stride 16, 512×512, 40.35M parameters |
| Training | CE + 0.8 Dice, AdamW (1e-4 encoder / 5e-4 decoder), cosine schedule, 40 epochs, batch 8, GPU augmentation, best epoch by val mIoU |
| Threat model | White-box, untargeted, L∞, perturbation in [0,1] RGB before normalisation, clipped to [0,1] |
| Attacks | FGSM; PGD-20 (α = ε/4, random start); SegPGD-20 (same, Gu et al. 2022 loss schedule); random ±ε noise control |
| Budgets | ε ∈ {1, 2, 4, 8}/255; step ablation T ∈ {5, 10, 20, 40} at ε = 4/255 (spec) and 1/255 (supplementary) |
| Metrics | Per-class IoU/Dice, headline mIoU/mDice (rare classes reported separately), ΔmIoU, relative fall, robustness ratio, correct→wrong pixel rate, PSNR; paired bootstrap 95% CIs |

## Repository layout

| Path | Contents |
|---|---|
| `*.py` | pipeline code (see above) and `visualize.py` (palette, helpers) |
| `runs/baseline/` | checkpoint, `history.csv`, `training_curves.png`, `clean_report.txt`, test panel |
| `results/raw/` | per-run per-frame counts (the source of every number) |
| `results/tables/`, `results/figures/` | generated tables (CSV) and figures (PNG) |
| `runs/legacy_sanity_deeplabv3/` | the earlier 3-epoch torchvision-DeepLabV3 sanity run (superseded) |
| `runs/baseline_908frames/`, `results_908frames/` | the first full run on the 908-frame partial download (80 epochs), kept as a replication check |
