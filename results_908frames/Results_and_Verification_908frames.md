# Results and Verification — Version 1

**Project:** Adversarial Attack and Robustness Analysis of AI-Based Cataract Surgical Instrument Segmentation
**Date of experiments:** 2 October 2026
**Companions:** [Version 1 spec](Version_1_Cataract_Adversarial_Robustness_Project.md) · [Implementation plan](Implementation_Plan_and_Roadmap.md) · [Gap matrix](Paper_Gap_Matrix.md) · auto-generated tables [results/RESULTS_TABLES.md](results/RESULTS_TABLES.md) · code [README.md](README.md)

Every number below is reproducible from `results/raw/*.npz` with `python analyze_results.py`. Percentages
are IoU/Dice × 100; "pp" = percentage points; ε is in 8-bit units of the [0, 1] RGB input.

---

## 1. Summary of findings

1. **Clean baseline is competent.** DeepLabV3+ (ResNet-50) reaches **77.08% headline mIoU (95% CI 75.06–78.86)
   and 86.72% mean Dice** on the held-out official Fold 1. Its anatomy IoUs are within ±3 points of the published
   Cataract-1K DeepLabV3+ numbers (iris 82.8 vs 80.0, pupil 89.2 vs 86.2, lens 77.1 vs 77.2).
2. **The model is extremely fragile to adversarial perturbation.** With a change of **at most one intensity level
   per pixel (ε = 1/255, PSNR 51 dB)**, PGD-20 and SegPGD-20 cut headline mIoU from 77.1% to **3.9% and 3.7%**
   (−95% relative). At ε = 8/255 the model is at 0.5–0.7% mIoU.
3. **The damage comes from the adversarial direction, not from noise.** Random ±ε noise of the same budget costs
   0.4% (ε = 1) to 10.3% (ε = 8) of mIoU. Across the same spec budgets (1–8/255), FGSM costs 32–35% at every budget
   and the iterative attacks cost 95–99%.
4. **Instruments are more fragile than anatomy under noise and single-step attacks; under iterative attacks the
   evidence is weaker.** Under FGSM, instruments lose +6 to +18 pp more of their clean IoU at every budget (bootstrap
   CIs exclude zero, all 6 test videos agree). Under random noise the gap grows with ε (+11.8 pp at 8/255). Under
   PGD/SegPGD the pooled gap is still positive (+0.6 to +8.7 pp) but is not consistent frame by frame or across
   videos, and object size does not explain it. Iterative attacks break every structure.
5. **The attack survives saving to an 8-bit image.** Rounding the adversarial frame to 8-bit pixel values changes
   PGD/SegPGD mIoU by at most 0.4 pp (e.g. PGD-20 at 1/255: 3.85% float → 3.48% 8-bit). The iterative attacks remain
   strong even below one 8-bit step: at ε = ¼/255, PGD-20 still removes 78% of mIoU.
6. **Under iterative attack the model hallucinates rather than going blank.** With SegPGD-20 at 4/255 the model labels
   41% of all pixels as pupil (ground truth 6.5%) and 7.6% as lens injector (ground truth 0.2%).
7. **SegPGD vs PGD.** At equal steps and budget the two reach the same mIoU (within CI), but SegPGD consistently flips
   more correctly-classified pixels (71–75% vs 61–69%), the behaviour its loss is designed for.

---

## 2. What was built, and what was corrected

### 2.1 Pipeline (all code in the project root)

| Stage | File | Notes |
|---|---|---|
| Masks | `convert_supervisely_to_masks.py` | 2,256 Supervisely JSONs → 13-class PNG masks (z-ordered: anatomy, then instruments) |
| Data / split | `cataract_dataset.py` | Official 5 patient-disjoint folds, **fold sizes verified against the paper (561/459/420/385/431)**; Fold 1 = test; 20 train / 4 val videos; zero patient leakage asserted |
| Model | `model_builder.py` | DeepLabV3+ (ResNet-50 ImageNet, OS16, ASPP 6/12/18, low-level decoder), 40.35M params; ImageNet normalisation **inside** the model |
| Loss / metrics | `loss.py`, `metrics.py` | CE + 0.8·Dice; per-frame per-class TP/FP/FN; headline means over a fixed class set |
| Training | `train_baseline.py` | 80 epochs, AdamW, cosine, GPU augmentation, bf16; best epoch by validation mIoU |
| Attacks | `attacks.py` | FGSM, PGD, SegPGD (Gu et al. 2022 schedule), random-sign noise control; L∞ in [0,1] pixel space |
| Experiments | `run_experiments.py` | 41 runs on Fold 1, budget check on every batch, resumable |
| Analysis | `analyze_results.py` | Tables, paired bootstrap CIs, sign tests, per-video consistency, figures |

### 2.2 Corrections to the pre-existing code (found during the audit before training)

| Issue in the earlier code | Effect if left in place | Fix |
|---|---|---|
| `test_fold=1` indexed a 0-based dict, so the test set was the paper's **Fold 2** (459 frames), not Fold 1 (561) | Results not comparable to the published benchmark; spec violated | Folds renumbered 1–5 as in the paper; fold sizes asserted against the paper |
| `model_builder.py` built torchvision **DeepLabV3**, not DeepLabV3+ | Spec / synopsis architecture not implemented | DeepLabV3+ implemented from torchvision ResNet-50 + ASPP with the low-level decoder |
| ImageNet normalisation applied in the dataset | Attack ε would be defined in normalised space, not the spec's [0,1] RGB space | Normalisation moved inside the model; datasets return [0,1] images |
| Dice loss clamped labels before one-hot, so an ignore label (255) became class 12 (Lens Injector) | Padding from geometric augmentation would be trained as Lens Injector | Ignore pixels masked out of both loss terms (unit-tested) |
| Validation used a whole fold (6 videos) | Spec says 20 train / 4 val | 4 validation videos, one per non-test fold |
| mIoU averaged over every class with any prediction, incl. background and classes absent from the ground truth | Under attack, hallucinated absent classes would add zeros and change the class set between clean and attacked runs | Class set fixed from test ground truth; rare classes reported separately (spec §25) |

### 2.3 Data actually available

The local copy of Synapse `syn53395479` contains **908 of the 2,256 annotated frames** as images (all 2,256 JSONs are
present). Every result below uses those 908 frames: **524 train / 125 val / 259 test** (Fold 1, 6 videos). Consequences:

- Capsulorhexis Cystotome does not occur in the available test frames, so it cannot be evaluated.
- Slit/Incision Knife and Katena Forceps occur in only 2 test frames each. They are reported separately and never
  enter a headline mean (spec §25).
- If the remaining images are downloaded into `dataset/`, the loader picks them up automatically; re-running the four
  commands in the README regenerates every number.

---

## 3. Experimental setup

| Item | Setting |
|---|---|
| Classes | Background, 3 anatomy (Cornea/iris, Pupil, Lens), 9 instruments (Slit+Incision knife merged as in the paper) |
| Headline class set | 9 classes present in test GT, minus the rare ones: Cornea, Pupil, Lens, Gauge, Capsulorhexis Forceps, Spatula, Phaco Tip, Irrigation-Aspiration, Lens Injector |
| Training | 512×512, batch 8, CE + 0.8 Dice, AdamW (1e-4 encoder / 5e-4 decoder, wd 1e-4), 1-epoch warm-up + cosine, 80 epochs (40.7 min), flips / affine / colour jitter / blur, seed 42; best epoch 68 (val mIoU 73.54%) |
| Threat model | White-box, untargeted, L∞; δ applied in [0,1] RGB before normalisation; x+δ clipped to [0,1]; attacker uses the ground-truth mask |
| Attacks | FGSM (1 step of size ε); PGD-T (random start, α = ε/4); SegPGD-T (same, loss reweighted with λ_t = (t−1)/2T); random ±ε sign noise (control) |
| Budgets | Spec: ε ∈ {1, 2, 4, 8}/255. Supplementary: ε ∈ {¼, ½}/255 (below one 8-bit step) and 8-bit-rounded versions of ε ∈ {1, 2}/255 |
| Step ablation | T ∈ {5, 10, 20, 40} at ε = 4/255 (spec) and 1/255 (supplementary) |
| Precision | Attacks and all evaluation in fp32 (only training used bf16) |
| Budget check | `check_perturbation` asserts ‖δ‖∞ ≤ ε and x+δ ∈ [0,1] on every batch of every run; measured max ‖δ‖∞ equals ε exactly in every attacked run |
| Statistics | Paired frame bootstrap (2,000 resamples) for CIs; per-frame sign test; per-video consistency count; Spearman ρ across classes |

---

## 4. Results

### 4.1 Clean baseline (Fold 1, 259 frames)

| Metric | Value |
|---|---:|
| Headline mIoU (95% CI) | **77.08** (75.06–78.86) |
| Headline mean Dice | **86.72** |
| Anatomy mIoU / Dice | 83.04 / 90.66 |
| Instrument mIoU / Dice (headline instruments) | 74.10 / 84.75 |
| Pixel accuracy | 98.12 |

Per class (IoU %): Cornea 82.8 · Pupil 89.2 · Lens 77.1 · Gauge 54.6 · Capsulorhexis Forceps 76.4 · Spatula 70.8 ·
Phaco Tip 79.5 · Irrigation-Aspiration 84.0 · Lens Injector 79.2 · *rare:* Slit/Incision Knife 74.9 (2 frames),
Katena Forceps 0.0 (2 frames).

**Comparison with the published benchmark** (Cataract-1K, *Sci. Data* 2024, Table 7, DeepLabV3+/ResNet-50):

| Structure | Paper IoU | Ours IoU | Paper Dice | Ours Dice |
|---|---:|---:|---:|---:|
| Iris (dataset label "Cornea") | 79.97 | 82.76 | 88.83 | 90.56 |
| Pupil | 86.24 | 89.23 | 92.61 | 94.31 |
| Lens | 77.23 | 77.14 | 87.12 | 87.09 |
| Instruments (all merged into one class) | 53.53 | 78.45 | 69.67 | 87.92 |

The anatomy results sit inside the published range, which meets the plan's "baseline within published benchmark range"
check. Our merged-instrument IoU is much higher than the paper's. Per-frame averaging does not explain this (76.0%),
and it is consistent across all six test videos (74–82%). The comparison is not like-for-like: the paper reports a
5-fold average with a different training recipe, and we test on the 259 Fold-1 frames that have images. **We therefore
do not claim to outperform the benchmark.** The conclusion we draw is narrower: the baseline is well converged, so
the vulnerability measured below cannot be blamed on an under-trained model.

![Training curves](results/figures/fig0_training_curves.png)

### 4.2 Main grid: clean vs. attacked (spec: 1 clean + 12 attacked runs, plus noise control)

| Attack | ε | mIoU % (95% CI) | ΔmIoU pp | Rel. fall % | Robustness ratio | mDice % | Anatomy | Instruments | Correct→wrong % | PSNR dB |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Clean** | — | 77.08 (75.06–78.86) | — | — | 1.000 | 86.72 | 83.04 | 74.10 | — | — |
| Noise (control) | 1/255 | 76.80 (74.83–78.62) | 0.27 | 0.4 | 0.996 | 86.53 | 83.17 | 73.62 | 0.1 | 48.2 |
| Noise (control) | 2/255 | 76.56 (74.50–78.44) | 0.52 | 0.7 | 0.993 | 86.38 | 83.03 | 73.32 | 0.3 | 42.2 |
| Noise (control) | 4/255 | 74.71 (72.44–76.78) | 2.37 | 3.1 | 0.969 | 85.13 | 82.64 | 70.75 | 0.6 | 36.2 |
| Noise (control) | 8/255 | 69.14 (66.85–71.54) | 7.94 | 10.3 | 0.897 | 81.23 | 80.75 | 63.34 | 1.2 | 30.2 |
| FGSM | 1/255 | 50.84 (47.96–53.95) | 26.24 | 34.0 | 0.660 | 66.83 | 59.96 | 46.28 | 3.7 | 48.2 |
| FGSM | 2/255 | 50.80 (47.83–54.03) | 26.28 | 34.1 | 0.659 | 66.76 | 60.56 | 45.92 | 3.4 | 42.2 |
| FGSM | 4/255 | 52.11 (49.00–55.51) | 24.97 | 32.4 | 0.676 | 67.88 | 62.56 | 46.89 | 3.1 | 36.2 |
| FGSM | 8/255 | 50.33 (46.90–53.62) | 26.75 | 34.7 | 0.653 | 66.16 | 63.66 | 43.66 | 3.4 | 30.2 |
| PGD-20 | 1/255 | 3.85 (3.19–4.56) | 73.23 | 95.0 | 0.050 | 7.29 | 6.08 | 2.73 | 66.4 | 51.3 |
| PGD-20 | 2/255 | 1.81 (1.36–2.27) | 75.27 | 97.7 | 0.023 | 3.52 | 3.20 | 1.12 | 68.8 | 45.6 |
| PGD-20 | 4/255 | 1.05 (0.72–1.38) | 76.03 | 98.6 | 0.014 | 2.05 | 2.16 | 0.49 | 66.5 | 39.8 |
| PGD-20 | 8/255 | 0.67 (0.39–1.02) | 76.40 | 99.1 | 0.009 | 1.33 | 1.60 | 0.21 | 61.3 | 33.9 |
| SegPGD-20 | 1/255 | 3.74 (3.14–4.38) | 73.34 | 95.1 | 0.049 | 7.06 | 5.07 | 3.08 | 71.2 | 51.3 |
| SegPGD-20 | 2/255 | 1.97 (1.50–2.45) | 75.11 | 97.4 | 0.026 | 3.82 | 2.79 | 1.56 | 74.8 | 45.6 |
| SegPGD-20 | 4/255 | 1.03 (0.72–1.34) | 76.05 | 98.7 | 0.013 | 2.02 | 1.95 | 0.56 | 72.0 | 39.8 |
| SegPGD-20 | 8/255 | 0.52 (0.32–0.78) | 76.55 | 99.3 | 0.007 | 1.04 | 0.86 | 0.36 | 68.3 | 33.9 |

![mIoU vs budget](results/figures/fig1_miou_vs_eps.png)

**Reading the grid.**

- **Noise vs. gradient attacks.** At ε = 1/255 the noise control and FGSM have *identical* PSNR (48.2 dB, every pixel
  moved by exactly ±1/255), yet noise costs 0.4% of mIoU and FGSM 34%. The vulnerability is directional, not a
  sensitivity to small perturbations in general.
- **FGSM plateaus.** FGSM's damage does not grow with ε (34.0 → 34.1 → 32.4 → 34.7%). A single linear step stops
  gaining once the loss surface curves, which is the expected single-step behaviour. A larger budget does not
  substitute for iterations.
- **Iterative attacks saturate at the smallest specified budget.** PGD-20 and SegPGD-20 already remove 95% of mIoU at
  ε = 1/255. That is why the supplementary sub-8-bit budgets were added (§4.6).
- **PGD perturbations are less visible than FGSM's at the same ε** (51.3 vs 48.2 dB at 1/255): PGD does not push every
  pixel to ±ε, yet it does roughly 3× the damage.

### 4.3 Which attack is stronger, and does iteration count matter?

**Step-count ablation at ε = 4/255 (spec)** — headline mIoU %:

| Steps | 1 (FGSM) | 5 | 10 | 20 | 40 |
|---|---:|---:|---:|---:|---:|
| PGD | 52.11 | 7.62 | 2.23 | 1.05 | 0.66 |
| SegPGD | — | 7.20 | 2.46 | 1.03 | 0.67 |
| PGD correct→wrong % | 3.1 | 23.0 | 43.5 | 66.5 | 82.7 |
| SegPGD correct→wrong % | — | 26.0 | 47.7 | 72.0 | 88.6 |

**Same ablation at ε = 1/255 (supplementary)**, where the attacks are less saturated:

| Steps | 1 (FGSM) | 5 | 10 | 20 | 40 |
|---|---:|---:|---:|---:|---:|
| PGD | 50.84 | 16.83 | 7.14 | 3.85 | 2.56 |
| SegPGD | — | 16.76 | 7.40 | 3.74 | 2.67 |
| PGD correct→wrong % | 3.7 | 20.4 | 43.7 | 66.4 | 82.6 |
| SegPGD correct→wrong % | — | 22.5 | 46.8 | 71.2 | 87.4 |

Five PGD steps at one intensity level already do more damage (−78% relative) than FGSM at eight levels (−35%).

![Step ablation](results/figures/fig5_step_ablation.png)

- Damage keeps increasing up to 40 steps, so **20-step results are conservative**: they are an upper bound on the
  model's robustness, as the SEA paper (P11) requires robustness numbers to be read.
- SegPGD and PGD reach statistically indistinguishable mIoU at every step count. SegPGD consistently converts 3–6 pp
  more of the correctly-classified pixels. On this model plain PGD is already strong enough to collapse mIoU, so the
  segmentation-aware reweighting shows up in pixel-flip rate rather than in mIoU.

### 4.4 Anatomy vs. instruments (spec RQ3; the project's novelty claim)

| Attack | ε | Anatomy rel. fall % | Instrument rel. fall % | Difference pp (95% CI) | Videos where instruments fall further |
|---|---:|---:|---:|---|---:|
| Noise | 4/255 | 0.5 | 4.5 | +4.0 (+1.8 to +6.3) | 4/6 |
| Noise | 8/255 | 2.8 | 14.5 | +11.8 (+8.0 to +15.5) | 6/6 |
| FGSM | 1/255 | 27.8 | 37.5 | +9.7 (+4.7 to +14.7) | 6/6 |
| FGSM | 2/255 | 27.1 | 38.0 | +11.0 (+5.7 to +16.0) | 6/6 |
| FGSM | 4/255 | 24.7 | 36.7 | +12.1 (+6.5 to +17.2) | 6/6 |
| FGSM | 8/255 | 23.3 | 41.1 | +17.7 (+12.1 to +23.6) | 6/6 |
| PGD-20 | 1/255 | 92.7 | 96.3 | +3.6 (+2.0 to +5.3) | 5/6 |
| PGD-20 | 8/255 | 98.1 | 99.7 | +1.6 (+0.7 to +2.8) | 4/6 |
| SegPGD-20 | 1/255 | 93.9 | 95.8 | +2.0 (+0.5 to +3.6) | 3/6 |
| SegPGD-20 | 8/255 | 99.0 | 99.5 | +0.6 (+0.0 to +1.0) | 4/6 |

The supplementary budgets give the iterative attacks unsaturated points (per-frame sign test p in brackets):

| Attack | ε | Anatomy rel. fall % | Instrument rel. fall % | Difference pp (95% CI) | Frames (sign-test p) | Videos |
|---|---:|---:|---:|---|---|---:|
| FGSM | ¼/255 | 23.5 | 29.6 | +6.1 (+1.3 to +10.6) | 126/219 (p = 0.015) | 6/6 |
| FGSM | ½/255 | 26.7 | 34.6 | +7.9 (+3.0 to +12.7) | 129/219 (p = 0.005) | 6/6 |
| PGD-20 | ¼/255 | 72.4 | 81.0 | +8.7 (+4.6 to +12.4) | 109/219 (p = 0.55) | 4/6 |
| PGD-20 | ½/255 | 86.3 | 90.0 | +3.7 (+1.0 to +6.4) | 109/210 (p = 0.32) | 4/6 |
| SegPGD-20 | ¼/255 | 74.0 | 79.8 | +5.8 (+1.6 to +9.6) | 93/219 (p = 0.99) | 4/6 |
| SegPGD-20 | ½/255 | 87.7 | 89.6 | +1.9 (−0.5 to +4.3) | 115/216 (p = 0.19) | 4/6 |

Full table (all budgets, absolute drops, per-frame sign tests): `results/RESULTS_TABLES.md` §T4.

![Anatomy vs instruments](results/figures/fig2_anatomy_vs_instruments.png)

**Interpretation.**

- **Noise and single-step attacks: clear, consistent effect.** Instruments lose a larger *share* of their clean IoU
  than anatomy at every FGSM budget (¼ to 8/255) and under noise at 4–8/255. The bootstrap CIs exclude zero, the
  per-frame sign tests are significant, and every one of the six test videos agrees for FGSM.
- **Iterative attacks: weak, inconsistent effect.** At the unsaturated ¼/255 budget, the *pooled* instrument fall
  exceeds the anatomy fall by 6–9 pp. However, instruments do not fall further in a majority of individual frames
  (sign tests p = 0.32–0.99), and only 4 of 6 videos agree. A follow-up check explains the discrepancy. Pooled IoU
  counts false positives in every frame, including frames where that instrument is absent. When only frames
  containing each class are scored, the gap shrinks from +8.7 to +5.8 pp (PGD) and from +5.8 to +1.8 pp (SegPGD),
  while the FGSM gap is unchanged (+9.7 → +11.1 pp). Most of the iterative "instrument fragility" is therefore
  **hallucinated instruments where none exist** (§4.7), not worse segmentation of the real ones. At ε ≥ 1/255 both
  groups collapse to near zero and the gap shrinks to ≤ 3.6 pp.
- **Conclusion for the project hypothesis.** Instrument segmentation is the more fragile part of the scene against
  random and single-step perturbations. A strong iterative attacker does not need to exploit that asymmetry: it
  breaks anatomy and instruments alike.
- **Absolute vs. relative.** Under PGD/SegPGD, anatomy loses *more absolute* IoU (≈ 77–82 pp vs ≈ 71–74 pp) simply
  because it starts higher. The project's hypothesis concerns how fast each group degrades, so relative fall is the
  primary measure; both are tabulated.
- **Statistical caution.** Frames from one video are correlated, so per-frame tests overstate certainty. The per-video
  counts (6 videos) are the conservative evidence. They are unanimous only for FGSM and for noise at 8/255.

### 4.5 Per-class vulnerability and the "thin metallic tools" hypothesis

![Per-class heatmap](results/figures/fig3_per_class_iou_heatmap.png)

![Per-class relative fall](results/figures/fig4_per_class_relative_fall.png)

- **Under FGSM at 1/255**, the two large, opaque anatomy classes are the least affected: Pupil keeps 72.1 of 89.2
  IoU (−19%) and Cornea 59.1 of 82.8 (−29%). Small instruments fall further: Lens Injector −44%,
  Irrigation-Aspiration −42%, Phaco Tip −40%, Gauge −39%. Two classes break the group pattern. **Lens (IOL)**,
  an anatomy class that is transparent and seen in only 57 frames, falls 37%, as much as the instruments.
  **Capsulorhexis Forceps** (−28%) behaves like the robust anatomy. Group membership is a strong but not perfect
  predictor; visibility and size matter too.
- **Under random noise at 8/255**, anatomy is almost unaffected (Lens +1%, Pupil −2%, Cornea −7%), while instruments
  lose 8–23% (Irrigation-Aspiration −23%, Gauge −16%, Lens Injector −15%, Spatula −14%).
- **Under SegPGD-20 at ¼/255** (iterative, unsaturated) the ordering changes. The two thinnest instruments, Spatula
  (−53%) and Gauge (−65%), are the *most* robust classes. Lens Injector (−98%), Capsulorhexis Forceps (−92%) and
  Pupil (−83%) fall furthest.
- **Object-size correlation is attack-dependent.** Across the 9 headline classes, the Spearman correlation between
  mean object area and relative fall is **negative under noise and FGSM** (ρ from −0.17 to −0.55): smaller
  structures fall further. It is **positive under PGD/SegPGD at sub-8-bit budgets** (ρ = +0.40 to +0.60): larger
  structures fall further. With only 9 classes **none of these correlations is statistically significant**
  (smallest p = 0.09).

![Object size vs fall](results/figures/fig6_object_size_vs_fall.png)

**Verdict on the hypothesis "thin, specular metallic tools degrade faster under small noise caps":**

- *Supported for noise and single-step attacks at the group level.* Instruments lose a larger share of IoU than
  anatomy with CIs excluding zero, unanimously across videos for FGSM (§4.4).
- *Not supported as a size effect.* The size–vulnerability correlation is weak, never significant, and **reverses
  sign** under iterative attacks, where the thinnest tools (Spatula, Gauge) are the most robust classes.
- The stronger, simpler finding is that an iterative attacker removes almost all structure regardless of class. What
  separates classes is mainly which ones the attack *hallucinates* (Pupil, Lens Injector; §4.7).

### 4.6 Supplementary: below one 8-bit step, and 8-bit quantisation

**Why these runs exist.** The spec's smallest budget (1/255) already saturates PGD/SegPGD. Two questions follow:
where does the collapse actually begin, and does the attack still work once the adversarial frame is saved as an
ordinary 8-bit image?

**(a) Budgets below one 8-bit step.** These perturbations exist only in floating point, i.e. for an attacker who can
modify decoded frames in memory.

| Attack | ε | mIoU % (95% CI) | Rel. fall % | Correct→wrong % |
|---|---:|---|---:|---:|
| FGSM | ¼/255 | 55.94 (53.17–58.79) | 27.4 | 3.0 |
| FGSM | ½/255 | 52.60 (49.70–55.58) | 31.8 | 3.5 |
| PGD-20 | ¼/255 | 17.01 (15.39–18.80) | 77.9 | 33.9 |
| PGD-20 | ½/255 | 8.74 (7.68–9.82) | 88.7 | 54.8 |
| SegPGD-20 | ¼/255 | 17.17 (15.67–18.78) | 77.7 | 36.4 |
| SegPGD-20 | ½/255 | 8.54 (7.61–9.55) | 88.9 | 58.2 |

Even a quarter of one intensity level removes 78% of mIoU with 20 iterations. The ε curve (Fig. 1) is therefore
steepest *below* the spec's range: the model has no meaningful robustness margin at any budget an 8-bit attacker can
use.

**(b) 8-bit quantisation.** The adversarial frame is rounded to the 8-bit grid before inference. Clean frames are
already on that grid, so ‖δ‖∞ ≤ ε still holds (measured exactly ε).

| Attack | ε | mIoU % float | mIoU % 8-bit |
|---|---:|---:|---:|
| FGSM | 1/255 | 50.84 | 50.85 |
| FGSM | 2/255 | 50.80 | 50.81 |
| PGD-20 | 1/255 | 3.85 | 3.48 |
| PGD-20 | 2/255 | 1.81 | 1.94 |
| SegPGD-20 | 1/255 | 3.74 | 3.93 |
| SegPGD-20 | 2/255 | 1.97 | 2.07 |

Rounding changes mIoU by at most 0.4 pp. **The headline result holds for adversarial frames stored as standard
8-bit images**, so it is not an artefact of floating-point precision.

### 4.7 What the model predicts under attack

| Class | Ground truth | Clean | Noise 8/255 | FGSM 4/255 | PGD-20 4/255 | SegPGD-20 4/255 |
|---|---:|---:|---:|---:|---:|---:|
| Background | 84.4 | 84.5 | 85.1 | 85.1 | 41.4 | 35.4 |
| Cornea | 5.5 | 5.6 | 5.4 | 5.6 | 8.8 | 9.7 |
| Pupil | 6.5 | 6.3 | 6.6 | 6.1 | 35.5 | 41.2 |
| Lens (IOL) | 1.0 | 1.1 | 1.0 | 1.2 | 1.9 | 1.8 |
| Capsulorhexis Forceps | 0.1 | 0.1 | 0.1 | 0.1 | 1.0 | 0.9 |
| Phaco Tip | 0.7 | 0.7 | 0.6 | 0.7 | 2.2 | 2.6 |
| Lens Injector | 0.2 | 0.2 | 0.2 | 0.2 | 8.2 | 7.6 |

(Share of all test pixels, %.) **FGSM** leaves the class proportions almost unchanged while removing a third of the
mIoU: it *mislocalises and confuses* structures. **PGD/SegPGD** make the model *hallucinate*. Over a third of the
frame becomes "pupil", and about 8% becomes "lens injector", an instrument present in only 8 test frames. For any
downstream use (tool tracking, phase recognition, skill metrics), confident false structures are a worse failure than
an empty mask.

### 4.8 Qualitative examples

![SegPGD panel, eps 4](results/figures/fig7_qualitative_segpgd_eps4.png)

![Attack comparison](results/figures/fig8_qualitative_attack_comparison_eps4.png)

![Budget progression](results/figures/fig9_qualitative_budget_progression.png)

Columns in Fig. 7: clean frame, adversarial frame, perturbation (rescaled from ±ε to the full range so it is
visible), ground truth, clean prediction, prediction under attack. The adversarial frames are visually
indistinguishable from the clean ones, yet the predictions turn into whole-frame "Lens Injector" or sprawling
"Pupil" regions, the hallucination pattern quantified in §4.7. Row 4 also shows a clean-model miss: the Katena
forceps (top right of the ground truth) is not detected even without attack, which is why that rare class scores
0% clean IoU.

---

## 5. Answers to the research questions (spec §6)

**Primary — How do adversarial attacks affect the performance and robustness of the model?**
Catastrophically and at very small budgets. A white-box iterative attack with ε = 1/255 (one intensity level, 51 dB
PSNR) removes 95% of the headline mIoU (77.1% → 3.7–3.9%). At 8/255 the model is at 0.5–0.7%. A single-step attack
removes about a third. Random noise of the same budget removes at most 10%.

| # | Supporting question | Answer | Evidence |
|---|---|---|---|
| 1 | How much does performance decrease? | ΔmIoU 26–27 pp (FGSM), 73–77 pp (PGD-20 / SegPGD-20); mDice falls from 86.7% to 66–68% (FGSM) and 1–7% (iterative) | §4.2 |
| 2 | Which attack settings degrade most? | Iterations matter far more than budget: SegPGD ≈ PGD ≫ FGSM ≫ noise. More steps keep helping up to 40. FGSM does not improve with ε | §4.2, §4.3 |
| 3 | Are some instruments / conditions more vulnerable? | Under noise and FGSM, yes: instruments lose +6 to +18 pp more of their IoU than anatomy (CIs exclude 0; all 6 videos for FGSM). Under iterative attacks the gap is small and mostly driven by hallucinated instruments in frames where they are absent. Object size does not explain vulnerability (ρ changes sign between attack types, never significant). Under FGSM the transparent Lens (IOL) is as fragile as the instruments | §4.4, §4.5, §4.7 |
| 4 | Clean vs. adversarial comparison? | 77.08% → 50.3–52.1% (FGSM), 0.5–3.9% (iterative) headline mIoU; full per-class tables | §4.2, T3 |
| 5 | Perturbation magnitude vs. degradation trade-off? | Strongly non-linear. Iterative attacks remove 78% of mIoU at ¼/255 and 95% at 1/255, then flatten. FGSM reaches its ~33% plateau by ½/255 and does not improve with ε. Noise grows slowly (0.4% → 10.3% from 1 to 8/255). Results are unchanged after 8-bit rounding | §4.2, §4.6, Fig. 1 |

---

## 6. Verification against the problem statement and plan

**Problem statement (spec §3):** *"How vulnerable is an AI-based cataract surgical instrument segmentation model to
adversarial perturbations, and how significantly do such attacks affect its performance?"* — **Answered** (§1, §5),
quantitatively, on the official held-out fold, with confidence intervals and a no-gradient control.

### 6.1 Specific objectives (spec §7)

| # | Objective | Status | Where |
|---|---|---|---|
| 1 | Study existing research | Done earlier | Literature pool, gap matrix |
| 2 | Select dataset and model | Done | Cataract-1K seg. subset; DeepLabV3+ / ResNet-50 |
| 3 | Clean baseline | **Done**: 77.08% mIoU, 86.72% mDice; within the published range for anatomy | §4.1 |
| 4 | Implement attacks | **Done**: FGSM, PGD, SegPGD (+ noise control), unit-tested | `attacks.py` |
| 5 | Generate adversarial examples | **Done**: 40 attacked runs × 259 frames | `results/raw/` |
| 6 | Evaluate on adversarial inputs | **Done**: same model, same metrics, fp32 | §4.2 |
| 7 | Quantify degradation | **Done**: ΔmIoU, relative fall, robustness ratio, mDice, correct→wrong, paired CIs | §4.2 |
| 8 | Attack strength vs. degradation | **Done**: 4 spec + 2 supplementary budgets, step ablation at two budgets | §4.2, §4.3, §4.6 |
| 9 | Vulnerability patterns | **Done**: anatomy vs instruments, per class, object size, prediction composition | §4.4–§4.7 |
| 10 | Document findings and limitations | **Done**: this document | §7 |

### 6.2 Locked decisions (spec §25) and plan milestones

| Requirement | Status |
|---|---|
| DeepLabV3+, ImageNet ResNet-50, OS16, 13 channels, 512×512 | ✔ (earlier code used DeepLabV3; corrected) |
| Official patient-disjoint folds, Fold 1 test, 20 train / 4 val videos | ✔ (earlier code tested on Fold 2; corrected) |
| White-box, untargeted, L∞, perturbation in [0,1] RGB before normalisation, clipped | ✔ checked on every batch |
| FGSM; PGD-20 (α = ε/4, random start); SegPGD-20 as primary | ✔ |
| ε ∈ {1, 2, 4, 8}/255 → 12 attacked + 1 clean | ✔ (+4 noise controls) |
| Step ablation {5, 10, 20, 40} at 4/255 for PGD and SegPGD | ✔ (+ same at 1/255) |
| mIoU and mean Dice headline; per-class IoU always reported; rare classes separate | ✔ |
| Absolute and relative degradation, robustness ratio, mIoU-vs-ε curves | ✔ |
| Anatomy vs instrument comparison | ✔ with CIs and per-video consistency |
| Clean baseline within published range (plan, Phase 2) | ✔ for anatomy; instrument comparison not like-for-like (§4.1) |
| Unit-test perturbation clipping (plan, 18 Oct) | ✔ `python attacks.py` + runtime assertion |
| Publication-quality panels: clean, perturbed, difference map, clean pred, attacked pred (plan, 4 Nov) | ✔ Fig. 7–9 |
| Code documentation, requirements, fixed seeds, README (plan, Phase 5) | ✔ |
| No defences / adversarial training (spec §28) | ✔ none added |
| Final thesis-format report and slide deck (plan, Phase 5–6) | **Not done here.** This document and `results/` supply the material |

---

## 7. Limitations and threats to validity

1. **Data subset.** Only 908 of 2,256 annotated frames have images locally. Training used 524 frames; three
   instrument classes cannot be evaluated reliably (Cystotome absent from test; Knife and Katena 2 frames each).
2. **One fold, one seed, one architecture.** Fold 1 only, as the spec specifies. No cross-fold variance and no
   second architecture (the optional UNet++ arm was not run).
3. **Correlated frames.** Bootstrap CIs and per-frame sign tests treat frames as independent; frames from the same
   video are not. The per-video consistency counts are the conservative evidence.
4. **Attack strength is a lower bound.** Damage was still rising from 20 to 40 steps. Measured mIoU under attack is an
   upper bound on robustness, never proof of it (P11, SEA).
5. **Threat model.** White-box, digital, untargeted, with ground-truth labels available to the attacker. This is the
   worst case by design, not a realistic operating-room attack. Black-box, transfer, physical and temporal (video)
   attacks are Version 2.
6. **Sub-8-bit budgets** (¼, ½ /255) cannot be stored in an 8-bit image; they apply only to an attacker who modifies
   decoded frames in memory. The 8-bit-rounded runs address the stored-image case.
7. **Benchmark comparison** with the paper is not like-for-like (single fold vs 5-fold mean, image subset, recipe).
8. **No clinical claims.** These are laboratory measurements on recorded frames. They say nothing directly about
   patient safety.

---

## 8. Deviations from the plan, and why

| Deviation | Reason |
|---|---|
| Random ±ε noise control added | Separates "any small perturbation" from "adversarial direction"; without it the degradation is uninterpretable |
| Sub-8-bit budgets (¼, ½ /255) and 8-bit-rounded runs added | Iterative attacks saturated already at the smallest spec budget, which left the ε-curve and the anatomy/instrument comparison with no unsaturated iterative points |
| Step ablation also at 1/255 | Same reason: to see the step effect where the attack has not yet saturated |
| Prediction-composition table added | Explains *how* the model fails (hallucination vs mislocalisation) |
| 80 training epochs instead of 40 | Validation mIoU was still improving at 40; the best epoch was 68 |

All additions are measurements; none is a defence, so Version-1 scope (spec §28) is preserved.

---

## 9. Reproduce

```bash
python train_baseline.py --epochs 80
```

```bash
python run_experiments.py
```

```bash
python analyze_results.py
```

Training ≈ 41 min and the 41-run grid ≈ 2 h (117 min of attack compute) on an RTX 3060 Laptop GPU (6 GB).
Analysis ≈ 1 min on the CPU.
