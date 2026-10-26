"""
analyze_results.py
==================
Turns results/raw/*.npz (written by run_experiments.py) into the Version 1 robustness analysis:

  Tables (results/tables/*.csv and results/RESULTS_TABLES.md)
    T1 clean baseline per class                T4 anatomy vs instrument degradation (+ tests)
    T2 main grid: clean + 12 attacks + control T5 step-count ablation (eps = 4/255)
    T3 per-class IoU and relative fall         T6 rare classes, reported separately
  Figures (results/figures/*.png)
    mIoU vs eps, anatomy vs instruments, per-class heatmap, per-class relative fall,
    step ablation, object size vs vulnerability, qualitative panels.

Statistics
  - Degradation:   dmIoU = mIoU_clean - mIoU_adv;  relative fall = dmIoU / mIoU_clean * 100;
                   robustness ratio = mIoU_adv / mIoU_clean.
  - 95% CIs:       paired frame bootstrap (2,000 resamples). The same resampled frames are used for
                   clean and attacked runs, so CIs on differences are paired. Frames from one video
                   are correlated, so these CIs are optimistic; a per-video consistency count is
                   reported next to them.
  - Anatomy vs instruments: bootstrap CI of (instrument relative fall - anatomy relative fall),
                   a per-frame sign test on robustness ratios, and the number of test videos (of 6)
                   in which instruments fall further.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from scipy import stats as sps

from cataract_dataset import (
    ANATOMY_CLASSES,
    CLASS_NAMES,
    NUM_CLASSES,
    RARE_CLASSES,
    extract_case_id,
)
from metrics import class_sets, summarize_counts
from visualize import PALETTE, colorize_mask, to_jsonable

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE_DIR, os.environ.get("RESULTS_DIR", "results"))
FIG = os.path.join(RES, "figures")
TAB = os.path.join(RES, "tables")

EPS_LIST = [1, 2, 4, 8]
ATTACKS = ["noise", "fgsm", "pgd", "segpgd"]
ATTACK_LABEL = {"clean": "Clean", "noise": "Random ±ε noise (control)", "fgsm": "FGSM",
                "pgd": "PGD-20", "segpgd": "SegPGD-20"}
SHORT_LABEL = {"noise": "Noise", "fgsm": "FGSM", "pgd": "PGD-20", "segpgd": "SegPGD-20"}
# Validated categorical slots (dataviz reference palette, light mode): slot 1 blue, 2 orange,
# 3 aqua; the no-gradient control is a neutral grey. Markers give a second, non-colour channel.
COLORS = {"segpgd": "#2a78d6", "pgd": "#eb6834", "fgsm": "#1baf7a", "noise": "#8f8e8a"}
MARKERS = {"segpgd": "^", "pgd": "s", "fgsm": "o", "noise": "D"}
GROUP_COLORS = {"Anatomy": "#2a78d6", "Instruments": "#eb6834"}
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE_RAMP = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
N_BOOT = 2000


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": TEXT,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlecolor": TEXT, "axes.grid": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linewidth": 0.8, "xtick.color": TEXT2, "ytick.color": TEXT2,
        "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.dpi": 200, "savefig.bbox": "tight", "lines.linewidth": 2, "lines.markersize": 7,
    })


# ----------------------------------------------------------------------------
# Loading and metric helpers
# ----------------------------------------------------------------------------
def load_runs() -> Dict[str, dict]:
    with open(os.path.join(RES, "run_config.json")) as f:
        cfg = json.load(f)
    runs = {}
    for r in cfg["runs"]:
        path = os.path.join(RES, "raw", f"{r['id']}.npz")
        if not os.path.isfile(path):
            print(f"[warn] missing run {r['id']}")
            continue
        d = np.load(path, allow_pickle=False)
        runs[r["id"]] = {**r, **{k: d[k] for k in d.files}}
    return runs


def run_id(attack: str, eps: float, steps: int = 20, q8: bool = False) -> str:
    if attack == "clean":
        return "clean"
    rid = f"{attack}_eps{eps:g}" + (f"_T{steps}" if attack in ("pgd", "segpgd") else "")
    return rid + ("_q8" if q8 else "")


EPS_SUB = [0.25, 0.5]           # supplementary, below the 8-bit quantisation step
EPS_ALL = EPS_SUB + EPS_LIST


def iou_matrix(pooled: np.ndarray) -> np.ndarray:
    """pooled [..., C, 3] -> IoU [..., C], NaN where the class has no ground truth."""
    tp, fp, fn = pooled[..., 0], pooled[..., 1], pooled[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(tp + fn > 0, tp / (tp + fp + fn), np.nan)


def group_mean(iou: np.ndarray, classes: List[int]) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(iou[..., classes], axis=-1)


class Bootstrap:
    """Paired frame bootstrap: one fixed set of resampling weights shared by every run."""

    def __init__(self, n_frames: int, n_boot: int = N_BOOT, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w = rng.multinomial(n_frames, np.full(n_frames, 1.0 / n_frames), size=n_boot).astype(np.float64)

    def iou(self, stats: np.ndarray) -> np.ndarray:
        pooled = np.einsum("rn,nck->rck", self.w, stats.astype(np.float64))
        return iou_matrix(pooled)  # [R, C]


def ci(samples: np.ndarray) -> tuple:
    return float(np.nanpercentile(samples, 2.5)), float(np.nanpercentile(samples, 97.5))


def rel_fall(clean: float, attacked: float) -> float:
    """Relative fall in % ((clean - attacked) / clean); NaN when undefined (clean is 0 or NaN)."""
    if clean is None or attacked is None or np.isnan(clean) or np.isnan(attacked) or clean <= 0:
        return float("nan")
    return 100.0 * (clean - attacked) / clean


def pct(x: float, digits: int = 2) -> str:
    return "—" if x is None or np.isnan(x) else f"{100 * x:.{digits}f}"


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    style()
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(TAB, exist_ok=True)
    runs = load_runs()
    clean = runs["clean"]
    n_frames = clean["stats"].shape[0]
    frame_names = [str(x) for x in clean["frame_names"]]
    videos = np.array([extract_case_id(f) for f in frame_names])
    gt_pixels = clean["stats"].sum(0)[:, 0] + clean["stats"].sum(0)[:, 2]
    sets = class_sets(gt_pixels)
    frames_present = ((clean["stats"][:, :, 0] + clean["stats"][:, :, 2]) > 0).sum(0)
    boot = Bootstrap(n_frames)
    groups = {"headline": sets["headline"], "anatomy": sets["anatomy"],
              "instruments": sets["instruments"], "instruments_all": sets["instruments_all"]}

    summaries, boot_groups = {}, {}
    for rid, r in runs.items():
        summaries[rid] = summarize_counts(r["stats"], sets)
        b_iou = boot.iou(r["stats"])
        boot_groups[rid] = {g: group_mean(b_iou, cls) for g, cls in groups.items()}
        boot_groups[rid]["per_class"] = b_iou
    c_sum = summaries["clean"]
    md: List[str] = []
    key: Dict[str, object] = {"n_test_frames": n_frames, "n_test_videos": int(len(set(videos))),
                              "class_sets": {k: [CLASS_NAMES[c] for c in v] for k, v in sets.items()}}

    # ---------------- T1: clean baseline ----------------
    rows = []
    for c in range(NUM_CLASSES):
        if c != 0 and c not in sets["eval"] and c not in sets["absent"]:
            continue
        group = "Background" if c == 0 else ("Anatomy" if c in ANATOMY_CLASSES else "Instrument")
        rows.append({
            "id": c, "class": CLASS_NAMES[c], "group": group, "rare": c in RARE_CLASSES,
            "test_frames": int(frames_present[c]),
            "IoU": c_sum["per_class_iou"][c], "Dice": c_sum["per_class_dice"][c],
        })
    t1 = pd.DataFrame(rows)
    t1.to_csv(os.path.join(TAB, "T1_clean_baseline_per_class.csv"), index=False)
    b = boot_groups["clean"]
    key["clean"] = {
        "mIoU": c_sum["mIoU"], "mIoU_ci": ci(b["headline"]), "mDice": c_sum["mDice"],
        "mIoU_anatomy": c_sum["mIoU_anatomy"], "mIoU_instruments": c_sum["mIoU_instruments"],
        "mIoU_instruments_all": c_sum["mIoU_instruments_all"], "mIoU_eval": c_sum["mIoU_eval"],
        "mDice_anatomy": c_sum["mDice_anatomy"], "mDice_instruments": c_sum["mDice_instruments"],
        "pixel_accuracy": c_sum["pixel_accuracy"],
    }
    md += ["## T1 — Clean baseline on the held-out test fold (Fold 1)", "",
           f"{n_frames} frames from {len(set(videos))} patient videos. Headline means use "
           f"{len(sets['headline'])} classes ({', '.join(CLASS_NAMES[c] for c in sets['headline'])}); "
           f"rare classes ({', '.join(CLASS_NAMES[c] for c in sets['rare'])}) are reported separately; "
           f"absent from test ground truth: {', '.join(CLASS_NAMES[c] for c in sets['absent']) or 'none'}.", "",
           f"| Metric | Value |", "|---|---:|",
           f"| Headline mIoU (95% CI) | {pct(c_sum['mIoU'])} ({pct(ci(b['headline'])[0])}–{pct(ci(b['headline'])[1])}) |",
           f"| Headline mean Dice | {pct(c_sum['mDice'])} |",
           f"| Anatomy mIoU / Dice | {pct(c_sum['mIoU_anatomy'])} / {pct(c_sum['mDice_anatomy'])} |",
           f"| Instrument mIoU / Dice (excl. rare) | {pct(c_sum['mIoU_instruments'])} / {pct(c_sum['mDice_instruments'])} |",
           f"| Instrument mIoU (incl. rare) | {pct(c_sum['mIoU_instruments_all'])} |",
           f"| mIoU over all evaluable foreground classes | {pct(c_sum['mIoU_eval'])} |",
           f"| Pixel accuracy | {pct(c_sum['pixel_accuracy'])} |", "",
           "| ID | Class | Group | Test frames | IoU % | Dice % |", "|---:|---|---|---:|---:|---:|"]
    for _, r in t1.iterrows():
        name = r["class"] + (" *(rare)*" if r["rare"] else "")
        md.append(f"| {r['id']} | {name} | {r['group']} | {r['test_frames']} | {pct(r['IoU'])} | {pct(r['Dice'])} |")
    md.append("")

    # Comparison with the published benchmark (Cataract-1K, Sci. Data 2024, Table 7,
    # "anatomy plus instrument" segmentation, DeepLabV3+ / ResNet-50 row). The paper scores all
    # instruments as ONE merged class, so ours is merged the same way for this comparison.
    paper = {"Iris (dataset label: Cornea)": (79.97, 88.83), "Pupil": (86.24, 92.61),
             "Lens (IOL)": (77.23, 87.12), "Instruments (merged)": (53.53, 69.67)}
    if "inst_binary" in clean:
        tp, fp, fn = clean["inst_binary"].sum(0).astype(float)
        inst_iou, inst_dice = tp / (tp + fp + fn), 2 * tp / (2 * tp + fp + fn)
    else:
        inst_iou = inst_dice = np.nan
    ours = {"Iris (dataset label: Cornea)": (c_sum["per_class_iou"][1], c_sum["per_class_dice"][1]),
            "Pupil": (c_sum["per_class_iou"][2], c_sum["per_class_dice"][2]),
            "Lens (IOL)": (c_sum["per_class_iou"][3], c_sum["per_class_dice"][3]),
            "Instruments (merged)": (inst_iou, inst_dice)}
    key["paper_comparison"] = {k: {"paper_iou": paper[k][0], "paper_dice": paper[k][1],
                                   "ours_iou": 100 * ours[k][0], "ours_dice": 100 * ours[k][1]} for k in paper}
    md += ["**Comparison with the published benchmark.** Cataract-1K paper (Ghamsarian et al., *Sci. Data* 2024), "
           "Table 7, DeepLabV3+ with ResNet-50, anatomy-plus-instrument task. The paper merges all instruments "
           "into one class and reports a 5-fold cross-validation average; we evaluate the single official Fold 1 "
           f"({n_frames} frames) with a 20-train / 4-validation video split, so the comparison is indicative, "
           "not like-for-like.", "",
           "| Structure | Paper IoU % | Ours IoU % | Paper Dice % | Ours Dice % |", "|---|---:|---:|---:|---:|"]
    for k in paper:
        md.append(f"| {k} | {paper[k][0]:.2f} | {pct(ours[k][0])} | {paper[k][1]:.2f} | {pct(ours[k][1])} |")
    md.append("")

    # ---------------- T2: main grid ----------------
    rows = []
    order = [("clean", 0)] + [(a, e) for a in ATTACKS for e in EPS_LIST]
    for attack, eps in order:
        rid = run_id(attack, eps)
        if rid not in runs:
            continue
        s, r, bg = summaries[rid], runs[rid], boot_groups[rid]
        d_boot = boot_groups["clean"]["headline"] - bg["headline"]
        rel_boot = d_boot / boot_groups["clean"]["headline"]
        mse = r["mse"] if attack != "clean" else None
        psnr = float(np.mean(10 * np.log10(1.0 / np.maximum(mse, 1e-12)))) if mse is not None else np.nan
        rows.append({
            "run": rid, "attack": ATTACK_LABEL[attack], "eps_255": eps,
            "steps": int(r["steps"]),
            "mIoU": s["mIoU"], "mIoU_lo": ci(bg["headline"])[0], "mIoU_hi": ci(bg["headline"])[1],
            "delta_mIoU": c_sum["mIoU"] - s["mIoU"], "delta_lo": ci(d_boot)[0], "delta_hi": ci(d_boot)[1],
            "rel_fall_pct": 100 * (c_sum["mIoU"] - s["mIoU"]) / c_sum["mIoU"],
            "rel_fall_lo": 100 * ci(rel_boot)[0], "rel_fall_hi": 100 * ci(rel_boot)[1],
            "robustness_ratio": s["mIoU"] / c_sum["mIoU"],
            "mDice": s["mDice"], "delta_mDice": c_sum["mDice"] - s["mDice"],
            "mIoU_anatomy": s["mIoU_anatomy"], "mIoU_instruments": s["mIoU_instruments"],
            "mIoU_instruments_all": s["mIoU_instruments_all"], "mIoU_eval": s["mIoU_eval"],
            "pixel_accuracy": s["pixel_accuracy"],
            "correct_to_wrong_pct": 100 * r["flipped_correct"].sum() / r["clean_correct"].sum(),
            "pixels_changed_pct": 100 * r["changed"].sum() / (n_frames * 512 * 512),
            "linf_max_255": float(r["linf_max"]) * 255, "psnr_db": psnr,
            "seconds": float(r["seconds"]),
        })
    t2 = pd.DataFrame(rows)
    t2.to_csv(os.path.join(TAB, "T2_main_grid.csv"), index=False)
    key["main_grid"] = t2.to_dict(orient="records")
    md += ["## T2 — Main grid: clean vs. attacked performance (test fold)", "",
           "ε in 8-bit units of the [0, 1] RGB input. PGD/SegPGD: 20 steps, α = ε/4, random start. "
           "95% CIs from a paired frame bootstrap. *Correct→wrong* = share of pixels the clean model "
           "classified correctly that the attack made wrong.", "",
           "| Attack | ε | mIoU % (95% CI) | ΔmIoU (pp) | Rel. fall % | Robustness ratio | mDice % | Anatomy mIoU % | Instrument mIoU % | Pixel acc. % | Correct→wrong % | PSNR dB |",
           "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in t2.iterrows():
        eps_s = "—" if r["eps_255"] == 0 else f"{r['eps_255']}/255"
        if r["run"] == "clean":
            md.append(f"| **Clean** | — | {pct(r['mIoU'])} ({pct(r['mIoU_lo'])}–{pct(r['mIoU_hi'])}) | — | — | 1.000 | "
                      f"{pct(r['mDice'])} | {pct(r['mIoU_anatomy'])} | {pct(r['mIoU_instruments'])} | "
                      f"{pct(r['pixel_accuracy'])} | — | — |")
            continue
        md.append(
            f"| {r['attack']} | {eps_s} | {pct(r['mIoU'])} ({pct(r['mIoU_lo'])}–{pct(r['mIoU_hi'])}) | "
            f"{100 * r['delta_mIoU']:.2f} | {r['rel_fall_pct']:.1f} | {r['robustness_ratio']:.3f} | "
            f"{pct(r['mDice'])} | {pct(r['mIoU_anatomy'])} | {pct(r['mIoU_instruments'])} | "
            f"{pct(r['pixel_accuracy'])} | {r['correct_to_wrong_pct']:.1f} | {r['psnr_db']:.1f} |")
    md.append("")

    # ---------------- T2b: eps curve incl. sub-8-bit budgets; 8-bit quantisation check ----------------
    curve_rows = []
    for a in ATTACKS:
        for e in EPS_ALL:
            for q8 in (False, True):
                rid = run_id(a, e, q8=q8)
                if rid not in summaries:
                    continue
                s, bg, r = summaries[rid], boot_groups[rid], runs[rid]
                curve_rows.append({
                    "run": rid, "attack": a, "eps_255": e, "quantised_8bit": q8,
                    "mIoU": s["mIoU"], "mIoU_lo": ci(bg["headline"])[0], "mIoU_hi": ci(bg["headline"])[1],
                    "rel_fall_pct": 100 * (c_sum["mIoU"] - s["mIoU"]) / c_sum["mIoU"],
                    "mIoU_anatomy": s["mIoU_anatomy"], "mIoU_instruments": s["mIoU_instruments"],
                    "correct_to_wrong_pct": 100 * r["flipped_correct"].sum() / r["clean_correct"].sum(),
                    "linf_max_255": float(r["linf_max"]) * 255,
                })
    curve = pd.DataFrame(curve_rows)
    curve.to_csv(os.path.join(TAB, "T2b_eps_curve_and_quantisation.csv"), index=False)
    key["eps_curve"] = curve.to_dict(orient="records")
    sub8 = curve[(~curve.quantised_8bit) & (curve.eps_255 < 1) & (curve.attack != "noise")]
    if not sub8.empty:
        md += ["## T2b — Supplementary: budgets below one 8-bit step (ε = 0.25, 0.5 / 255)", "",
               "Added because PGD-20 and SegPGD-20 already saturate at the smallest specified budget (1/255). "
               "These perturbations cannot be stored in an 8-bit image; they show where the transition happens "
               "for an attacker who can modify decoded frames in memory.", "",
               "| Attack | ε | mIoU % (95% CI) | Rel. fall % | Anatomy mIoU % | Instrument mIoU % | Correct→wrong % |",
               "|---|---:|---|---:|---:|---:|---:|"]
        for a in ("fgsm", "pgd", "segpgd"):
            for e in EPS_ALL[:3]:
                row = curve[(curve.attack == a) & (curve.eps_255 == e) & (~curve.quantised_8bit)]
                if row.empty:
                    continue
                r = row.iloc[0]
                md.append(f"| {SHORT_LABEL[a]} | {e:g}/255 | {pct(r['mIoU'])} ({pct(r['mIoU_lo'])}–{pct(r['mIoU_hi'])}) | "
                          f"{r['rel_fall_pct']:.1f} | {pct(r['mIoU_anatomy'])} | {pct(r['mIoU_instruments'])} | "
                          f"{r['correct_to_wrong_pct']:.1f} |")
        md.append("")
    q = curve[curve.quantised_8bit]
    if not q.empty:
        md += ["## T2c — Supplementary: do the attacks survive 8-bit quantisation?", "",
               "Adversarial frames rounded to the 8-bit grid (as when saved to PNG) before inference, "
               "compared with the float perturbation of the same run settings.", "",
               "| Attack | ε | mIoU % float | mIoU % 8-bit | Rel. fall % float | Rel. fall % 8-bit | max ‖δ‖∞ after rounding |",
               "|---|---:|---:|---:|---:|---:|---:|"]
        for _, r in q.iterrows():
            f_row = curve[(curve.attack == r["attack"]) & (curve.eps_255 == r["eps_255"]) & (~curve.quantised_8bit)]
            if f_row.empty:
                continue
            f = f_row.iloc[0]
            md.append(f"| {SHORT_LABEL[r['attack']]} | {r['eps_255']:g}/255 | {pct(f['mIoU'])} | {pct(r['mIoU'])} | "
                      f"{f['rel_fall_pct']:.1f} | {r['rel_fall_pct']:.1f} | {r['linf_max_255']:.2f}/255 |")
        md.append("")

    # ---------------- T3: per-class IoU and relative fall ----------------
    eval_cls = sets["anatomy"] + [c for c in sets["instruments_all"]]
    rows = []
    for c in eval_cls:
        row = {"class": CLASS_NAMES[c], "group": "Anatomy" if c in ANATOMY_CLASSES else "Instrument",
               "rare": c in RARE_CLASSES, "test_frames": int(frames_present[c]),
               "clean": c_sum["per_class_iou"][c]}
        for a in ATTACKS:
            for e in EPS_ALL:
                rid = run_id(a, e)
                if rid in summaries:
                    row[f"{a}_eps{e:g}"] = summaries[rid]["per_class_iou"][c]
        rows.append(row)
    t3 = pd.DataFrame(rows)
    t3.to_csv(os.path.join(TAB, "T3_per_class_iou.csv"), index=False)
    rel = t3.copy()
    for col in rel.columns:
        if "_eps" in col:
            rel[col] = [rel_fall(c0, c1) for c0, c1 in zip(t3["clean"], t3[col])]
    rel.to_csv(os.path.join(TAB, "T3b_per_class_relative_fall_pct.csv"), index=False)
    md += ["## T3 — Per-class IoU % (clean and attacked)", "",
           "| Class | Group | Frames | Clean | Noise 8 | FGSM 1 | FGSM 8 | PGD ¼ | PGD 1 | SegPGD ¼ | SegPGD ½ | SegPGD 1 | SegPGD 8 |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in t3.iterrows():
        name = r["class"] + (" *" if r["rare"] else "")
        md.append(f"| {name} | {r['group']} | {r['test_frames']} | {pct(r['clean'], 1)} | "
                  + " | ".join(pct(r.get(k, np.nan), 1) for k in
                               ["noise_eps8", "fgsm_eps1", "fgsm_eps8", "pgd_eps0.25", "pgd_eps1",
                                "segpgd_eps0.25", "segpgd_eps0.5", "segpgd_eps1", "segpgd_eps8"]) + " |")
    md += ["", "\\* rare class (excluded from headline means). Columns are attack + ε (in /255). "
           "Full grid in `tables/T3_per_class_iou.csv`.", "",
           "**Per-class relative fall % under SegPGD-20** (primary attack)", "",
           "| Class | Group | ε=¼ † | ε=½ † | ε=1 | ε=2 | ε=4 | ε=8 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in rel.iterrows():
        name = r["class"] + (" *" if r["rare"] else "")
        md.append(f"| {name} | {r['group']} | " + " | ".join(
            "—" if np.isnan(r.get(f"segpgd_eps{e:g}", np.nan)) else f"{r[f'segpgd_eps{e:g}']:.1f}" for e in EPS_ALL) + " |")
    md.append("")

    # ---------------- T4: anatomy vs instruments ----------------
    frame_iou = {rid: iou_matrix(r["stats"].astype(np.float64)) for rid, r in runs.items()}
    anat_f = {rid: group_mean(v, sets["anatomy"]) for rid, v in frame_iou.items()}
    inst_f = {rid: group_mean(v, sets["instruments"]) for rid, v in frame_iou.items()}
    rows = []
    for a in ATTACKS:
        for e in EPS_ALL:
            rid = run_id(a, e)
            if rid not in runs:
                continue
            bc, ba = boot_groups["clean"], boot_groups[rid]
            ra = (bc["anatomy"] - ba["anatomy"]) / bc["anatomy"]
            ri = (bc["instruments"] - ba["instruments"]) / bc["instruments"]
            diff = ri - ra
            s = summaries[rid]
            rel_a = (c_sum["mIoU_anatomy"] - s["mIoU_anatomy"]) / c_sum["mIoU_anatomy"]
            rel_i = (c_sum["mIoU_instruments"] - s["mIoU_instruments"]) / c_sum["mIoU_instruments"]
            # Per-frame sign test on robustness ratios (frames containing a headline instrument).
            ok = (~np.isnan(inst_f["clean"])) & (inst_f["clean"] > 0) & (anat_f["clean"] > 0)
            ratio_i = inst_f[rid][ok] / inst_f["clean"][ok]
            ratio_a = anat_f[rid][ok] / anat_f["clean"][ok]
            k_more = int(np.sum(ratio_i < ratio_a))
            n_nontie = int(np.sum(ratio_i != ratio_a))
            p_sign = float(sps.binomtest(k_more, n_nontie, 0.5, alternative="greater").pvalue) if n_nontie else np.nan
            # Per-video consistency.
            vids_more, vids_valid = 0, 0
            for v in sorted(set(videos)):
                m = videos == v
                pc = summarize_counts(runs["clean"]["stats"][m], sets)
                pa = summarize_counts(runs[rid]["stats"][m], sets)
                fa = rel_fall(pc["mIoU_anatomy"], pa["mIoU_anatomy"])
                fi = rel_fall(pc["mIoU_instruments"], pa["mIoU_instruments"])
                if np.isnan(fa) or np.isnan(fi):
                    continue
                vids_valid += 1
                vids_more += int(fi > fa)
            rows.append({
                "attack": SHORT_LABEL[a], "eps_255": e,
                "anatomy_rel_fall_pct": 100 * rel_a, "instrument_rel_fall_pct": 100 * rel_i,
                "diff_pp": 100 * (rel_i - rel_a), "diff_lo": 100 * ci(diff)[0], "diff_hi": 100 * ci(diff)[1],
                "anatomy_abs_drop_pp": 100 * (c_sum["mIoU_anatomy"] - s["mIoU_anatomy"]),
                "instrument_abs_drop_pp": 100 * (c_sum["mIoU_instruments"] - s["mIoU_instruments"]),
                "frames_inst_worse": k_more, "frames_compared": n_nontie, "sign_test_p": p_sign,
                "videos_inst_worse": vids_more, "videos": vids_valid,
            })
    t4 = pd.DataFrame(rows)
    t4.to_csv(os.path.join(TAB, "T4_anatomy_vs_instruments.csv"), index=False)
    key["anatomy_vs_instruments"] = t4.to_dict(orient="records")
    md += ["## T4 — Anatomy vs. instrument vulnerability", "",
           "Relative fall = (clean − attacked) / clean of the group mIoU. *Diff* = instrument − anatomy "
           "relative fall (percentage points) with a paired bootstrap 95% CI; a CI above 0 means instruments "
           "lose a larger share of their clean IoU. *Frames* = per-frame sign test on robustness ratios "
           "(frames that contain a headline instrument class; one-sided p for 'instruments fall further'). "
           "*Videos* = test videos in which instruments fall further.", "",
           "| Attack | ε | Anatomy rel. fall % | Instrument rel. fall % | Diff pp (95% CI) | Anatomy ΔmIoU pp | Instrument ΔmIoU pp | Frames inst. worse | Sign-test p | Videos inst. worse |",
           "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|"]
    for _, r in t4.iterrows():
        p = r["sign_test_p"]
        p_s = "<0.001" if p < 1e-3 else f"{p:.3f}"
        md.append(f"| {r['attack']} | {r['eps_255']:g}/255{' †' if r['eps_255'] < 1 else ''} | {r['anatomy_rel_fall_pct']:.1f} | "
                  f"{r['instrument_rel_fall_pct']:.1f} | {r['diff_pp']:+.1f} ({r['diff_lo']:+.1f} to {r['diff_hi']:+.1f}) | "
                  f"{r['anatomy_abs_drop_pp']:.1f} | {r['instrument_abs_drop_pp']:.1f} | "
                  f"{r['frames_inst_worse']}/{r['frames_compared']} | {p_s} | {r['videos_inst_worse']}/{r['videos']} |")
    md += ["", "† supplementary budget below one 8-bit step (see T2b).", ""]

    # Same comparison, scoring each class only on frames where it is present in the ground truth.
    # Pooled IoU also charges a class for pixels hallucinated in frames where it does not occur;
    # this variant separates "segments real instruments worse" from "hallucinates instruments".
    present = (clean["stats"][:, :, 0] + clean["stats"][:, :, 2]) > 0      # [N, C]

    def present_group_iou(stats, classes):
        vals = []
        for c in classes:
            tp, fp, fn = stats[present[:, c], c, :].sum(0).astype(float)
            vals.append(tp / (tp + fp + fn) if tp + fp + fn > 0 else np.nan)
        return float(np.nanmean(vals))

    rows = []
    for a in ATTACKS:
        for e in EPS_ALL:
            rid = run_id(a, e)
            if rid not in runs:
                continue
            ca, ci_ = present_group_iou(clean["stats"], sets["anatomy"]), present_group_iou(clean["stats"], sets["instruments"])
            aa, ai = present_group_iou(runs[rid]["stats"], sets["anatomy"]), present_group_iou(runs[rid]["stats"], sets["instruments"])
            pooled = t4[(t4.attack == SHORT_LABEL[a]) & (t4.eps_255 == e)]
            rows.append({"attack": SHORT_LABEL[a], "eps_255": e,
                         "anatomy_rel_fall_pct": rel_fall(ca, aa), "instrument_rel_fall_pct": rel_fall(ci_, ai),
                         "diff_present_only_pp": rel_fall(ci_, ai) - rel_fall(ca, aa),
                         "diff_pooled_pp": float(pooled["diff_pp"].iloc[0]) if not pooled.empty else np.nan})
    t4c = pd.DataFrame(rows)
    t4c.to_csv(os.path.join(TAB, "T4c_present_frames_only.csv"), index=False)
    key["anatomy_vs_instruments_present_only"] = t4c.to_dict(orient="records")
    md += ["**Present-frames-only check.** Each class scored only on frames where it occurs, so hallucinated "
           "pixels in frames without that class are not counted. If the instrument–anatomy gap shrinks here, "
           "part of the pooled gap was hallucination rather than worse segmentation of real instruments.", "",
           "| Attack | ε | Anatomy rel. fall % | Instrument rel. fall % | Diff pp (present-only) | Diff pp (pooled, T4) |",
           "|---|---:|---:|---:|---:|---:|"]
    for _, r in t4c.iterrows():
        md.append(f"| {r['attack']} | {r['eps_255']:g}/255 | {r['anatomy_rel_fall_pct']:.1f} | "
                  f"{r['instrument_rel_fall_pct']:.1f} | {r['diff_present_only_pp']:+.1f} | {r['diff_pooled_pp']:+.1f} |")
    md.append("")

    # Object size vs vulnerability (the "thin metallic tools" hypothesis). Evaluated in the regimes
    # where attacks are still discriminative (not saturated): see VULN_PANELS.
    area = {c: gt_pixels[c] / max(frames_present[c], 1) for c in eval_cls}
    size_rows = []
    for a in ATTACKS:
        for e in EPS_ALL:
            rid = run_id(a, e)
            if rid not in summaries:
                continue
            for c in eval_cls:
                size_rows.append({"run": rid, "attack": a, "eps_255": e, "class": CLASS_NAMES[c],
                                  "rare": c in RARE_CLASSES,
                                  "group": "Anatomy" if c in ANATOMY_CLASSES else "Instrument",
                                  "mean_area_px": area[c], "clean_iou": c_sum["per_class_iou"][c],
                                  "attacked_iou": summaries[rid]["per_class_iou"][c],
                                  "rel_fall_pct": rel_fall(c_sum["per_class_iou"][c], summaries[rid]["per_class_iou"][c])})
    size_df = pd.DataFrame(size_rows, columns=["run", "attack", "eps_255", "class", "rare", "group", "mean_area_px",
                                               "clean_iou", "attacked_iou", "rel_fall_pct"])
    size_df.to_csv(os.path.join(TAB, "T4b_object_size_vs_fall.csv"), index=False)
    spearman = {}
    for rid in size_df.run.unique():
        sub = size_df[(size_df.run == rid) & (~size_df.rare)]
        if len(sub) > 2 and sub.rel_fall_pct.std() > 0:
            rho, p = sps.spearmanr(sub["mean_area_px"], sub["rel_fall_pct"])
            spearman[rid] = (float(rho), float(p), len(sub))
    key["size_vs_fall_spearman"] = spearman
    md += ["**Object size vs. vulnerability.** Spearman correlation across the headline classes between mean "
           "ground-truth area per frame and per-class relative IoU fall (negative ρ = smaller structures fall "
           "further). Saturated runs (every class near 100 % fall) carry no information about size and are "
           "omitted.", "",
           "| Attack | ε | Spearman ρ | p | classes | mean rel. fall % |", "|---|---:|---:|---:|---:|---:|"]
    for rid, (rho, p, n) in spearman.items():
        sub = size_df[(size_df.run == rid) & (~size_df.rare)]
        if sub.rel_fall_pct.mean() > 95:
            continue
        a, e = sub.attack.iloc[0], sub.eps_255.iloc[0]
        md.append(f"| {SHORT_LABEL[a]} | {e:g}/255 | {rho:+.2f} | {p:.3f} | {n} | {sub.rel_fall_pct.mean():.1f} |")
    md.append("")

    # ---------------- T5: step-count ablation ----------------
    rows = []
    for e in (4, 1):
        rid = run_id("fgsm", e)
        if rid in summaries:
            s = summaries[rid]
            rows.append({"eps_255": e, "attack": "FGSM", "steps": 1, "mIoU": s["mIoU"],
                         "mIoU_anatomy": s["mIoU_anatomy"], "mIoU_instruments": s["mIoU_instruments"],
                         "seconds": float(runs[rid]["seconds"])})
        for a in ("pgd", "segpgd"):
            for t in (5, 10, 20, 40):
                rid = run_id(a, e, t)
                if rid in summaries:
                    s = summaries[rid]
                    rows.append({"eps_255": e, "attack": SHORT_LABEL[a].replace("-20", ""), "steps": t,
                                 "mIoU": s["mIoU"], "mIoU_anatomy": s["mIoU_anatomy"],
                                 "mIoU_instruments": s["mIoU_instruments"],
                                 "seconds": float(runs[rid]["seconds"])})
    t5 = pd.DataFrame(rows, columns=["eps_255", "attack", "steps", "mIoU", "mIoU_anatomy",
                                     "mIoU_instruments", "seconds"])
    t5.to_csv(os.path.join(TAB, "T5_step_ablation.csv"), index=False)
    key["ablation"] = t5.to_dict(orient="records")
    md += ["## T5 — Step-count ablation (ε = 4/255 per spec; ε = 1/255 supplementary)", "",
           f"Clean headline mIoU = {pct(c_sum['mIoU'])}%. α = ε/4 for every step count.", "",
           "| ε | Attack | Steps | mIoU % | Anatomy mIoU % | Instrument mIoU % | Time (s, all test frames) |",
           "|---:|---|---:|---:|---:|---:|---:|"]
    # Wall-clock times: a run that spans a system sleep shows an inflated time. Flag any run whose
    # time per attack step is more than 5x the median of the iterative runs.
    per_step = (t5.seconds / t5.steps.clip(lower=1))[t5.steps > 1]
    slow_limit = 5 * float(per_step.median()) if len(per_step) else np.inf
    flagged = False
    for _, r in t5.iterrows():
        slow = r["steps"] > 1 and r["seconds"] / r["steps"] > slow_limit
        flagged |= bool(slow)
        md.append(f"| {r['eps_255']}/255 | {r['attack']} | {r['steps']} | {pct(r['mIoU'])} | "
                  f"{pct(r['mIoU_anatomy'])} | {pct(r['mIoU_instruments'])} | {r['seconds']:.0f}{' ‡' if slow else ''} |")
    if flagged:
        md += ["", "‡ wall-clock time inflated by a system sleep during the run; the attack result is unaffected."]
    md.append("")

    # ---------------- T6: rare classes ----------------
    md += ["## T6 — Rare classes (reported separately)", "",
           "Too few test frames for a reliable estimate; shown for completeness only.", "",
           "| Class | Test frames | Clean IoU % | FGSM 4 | PGD-20 4 | SegPGD-20 4 | SegPGD-20 8 |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for c in sets["rare"]:
        vals = [summaries.get(run_id(a, e), {}).get("per_class_iou", {}).get(c, np.nan)
                for a, e in (("fgsm", 4), ("pgd", 4), ("segpgd", 4), ("segpgd", 8))]
        md.append(f"| {CLASS_NAMES[c]} | {frames_present[c]} | {pct(c_sum['per_class_iou'][c], 1)} | "
                  + " | ".join(pct(v, 1) for v in vals) + " |")
    for c in sets["absent"]:
        md.append(f"| {CLASS_NAMES[c]} | 0 | — | — | — | — | — |")
    md.append("")

    # ---------------- T7: what the attacked model predicts instead ----------------
    comp_runs = [("Ground truth", None), ("Clean", "clean")] + [
        (f"{SHORT_LABEL[a]} {e}/255", run_id(a, e)) for a, e in (("noise", 8), ("fgsm", 4), ("pgd", 4), ("segpgd", 4))]
    comp = {}
    for label, rid in comp_runs:
        pooled = clean["stats"].sum(0) if rid is None else runs[rid]["stats"].sum(0) if rid in runs else None
        if pooled is None:
            continue
        pixels = (pooled[:, 0] + pooled[:, 2]) if rid is None else (pooled[:, 0] + pooled[:, 1])
        comp[label] = 100.0 * pixels / pixels.sum()
    comp_df = pd.DataFrame(comp, index=[CLASS_NAMES[c] for c in range(NUM_CLASSES)])
    comp_df.to_csv(os.path.join(TAB, "T7_predicted_class_composition.csv"))
    md += ["## T7 — What the attacked model predicts instead", "",
           "Share of all test pixels assigned to each class (ground truth vs. predictions). Iterative attacks do not "
           "blank the output; they make the model hallucinate large anatomy / instrument regions.", "",
           "| Class | " + " | ".join(comp_df.columns) + " |", "|---|" + "---:|" * len(comp_df.columns)]
    for name, row in comp_df.iterrows():
        md.append(f"| {name} | " + " | ".join(f"{v:.1f}" for v in row) + " |")
    md.append("")

    # ---------------- Figures ----------------
    fig_training_curves()
    fig_miou_vs_eps(curve, c_sum)
    fig_groups_vs_eps(summaries, c_sum)
    fig_heatmap(t3)
    fig_class_fall(size_df)
    fig_ablation(t5, c_sum)
    fig_size(size_df, spearman)
    fig_qualitative(runs)
    md += ["## Figures", ""] + [f"- `figures/{f}`" for f in sorted(os.listdir(FIG))] + [""]

    header = ["# Version 1 results — adversarial robustness of cataract surgical segmentation", "",
              "Auto-generated by `analyze_results.py` from `results/raw/`. Model: DeepLabV3+ (ResNet-50, OS16), "
              "trained on 20 Cataract-1K videos; evaluated on the official Fold 1 (6 videos). "
              "Threat model: white-box, untargeted, L∞ in [0, 1] RGB before normalisation.", ""]
    with open(os.path.join(RES, "RESULTS_TABLES.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(header + md))
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(to_jsonable(key), f, indent=2)
    print("\n".join(md))
    print(f"\nwrote {RES}/RESULTS_TABLES.md, summary.json, tables/, figures/")


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig_training_curves() -> None:
    path = os.path.join(BASE_DIR, "runs", "baseline", "history.csv")
    if not os.path.isfile(path):
        return
    h = pd.read_csv(path)
    best = h.loc[h.val_mIoU.idxmax()]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(h.epoch, h.train_loss, color=COLORS["segpgd"], label="Train loss (CE + 0.8 Dice)")
    a1.plot(h.epoch, h.val_loss, color=COLORS["pgd"], label="Validation loss")
    a1.set_xlabel("Epoch"); a1.set_ylabel("Loss"); a1.legend(fontsize=9)
    a1.set_title("Loss", loc="left")
    for key, label, color, marker in (("val_mIoU_anatomy", "Anatomy", GROUP_COLORS["Anatomy"], None),
                                      ("val_mIoU", "Headline", TEXT2, None),
                                      ("val_mIoU_instruments", "Instruments", GROUP_COLORS["Instruments"], None)):
        a2.plot(h.epoch, 100 * h[key], color=color, label=label)
    a2.axvline(best.epoch, color=GRID, linewidth=1.5)
    a2.text(best.epoch - 1, 3, f"selected epoch {int(best.epoch)}", ha="right", fontsize=9, color=TEXT2)
    a2.set_xlabel("Epoch"); a2.set_ylabel("Validation mIoU (%)"); a2.set_ylim(0, 100)
    a2.legend(fontsize=9, loc="center right")
    a2.set_title("Validation mIoU (4 held-out training-fold videos)", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig0_training_curves.png"))
    plt.close(fig)


EPS_AXIS = [0] + EPS_ALL                     # clean, 0.25, 0.5, 1, 2, 4, 8
EPS_X = {e: i for i, e in enumerate(EPS_AXIS)}
EPS_TICKS = ["clean", "¼", "½", "1", "2", "4", "8"]
# Regimes in which the attacks are not saturated, used for per-class / object-size plots.
VULN_PANELS = [("noise", 8), ("fgsm", 1), ("segpgd", 0.25)]


def _eps_axis(ax, label: bool = True) -> None:
    """Shared x-axis for eps curves, with the sub-8-bit (supplementary) region shaded."""
    ax.axvspan(0.5, 2.5, color="#f0efec", zorder=0, linewidth=0)
    ax.set_xticks(range(len(EPS_AXIS)), EPS_TICKS)
    ax.set_xlim(-0.3, len(EPS_AXIS) - 0.4)
    if label:
        ax.set_xlabel("Perturbation budget ε (/255, L∞)")


def _declutter(values: List[float], min_gap: float) -> List[float]:
    """Spread label y-positions so that they are at least min_gap apart, keeping order."""
    order = np.argsort(values)
    out = np.array(values, dtype=float)
    for i in range(1, len(order)):
        if out[order[i]] - out[order[i - 1]] < min_gap:
            out[order[i]] = out[order[i - 1]] + min_gap
    return out.tolist()


def fig_miou_vs_eps(curve: pd.DataFrame, c_sum) -> None:
    if curve.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    _eps_axis(ax)
    ends, labels = [], []
    for a in ATTACKS:
        sub = curve[(curve.attack == a) & (~curve.quantised_8bit)].sort_values("eps_255")
        if sub.empty:
            continue
        x = [0] + [EPS_X[e] for e in sub.eps_255]
        y = [100 * c_sum["mIoU"]] + list(100 * sub.mIoU)
        lo = [100 * c_sum["mIoU"]] + list(100 * sub.mIoU_lo)
        hi = [100 * c_sum["mIoU"]] + list(100 * sub.mIoU_hi)
        ax.fill_between(x, lo, hi, color=COLORS[a], alpha=0.14, linewidth=0)
        ax.plot(x, y, color=COLORS[a], marker=MARKERS[a], label=ATTACK_LABEL[a],
                markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        ends.append(y[-1])
        labels.append((a, x[-1]))
    for (a, x_end), yp in zip(labels, _declutter(ends, 4.0)):
        ax.text(x_end + 0.12, yp, SHORT_LABEL[a], va="center", color=TEXT2, fontsize=9)
    ax.text(1.5, 2, "below one 8-bit step\n(supplementary)", ha="center", va="bottom", fontsize=8, color=TEXT2)
    ax.set_ylim(0, max(100 * c_sum["mIoU"] + 6, 10))
    ax.set_ylabel("Headline mIoU on test fold (%)")
    ax.set_title("Headline mIoU vs. perturbation budget (shaded bands: 95% bootstrap CI)", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, fontsize=9)
    fig.savefig(os.path.join(FIG, "fig1_miou_vs_eps.png"))
    plt.close(fig)


def fig_groups_vs_eps(summaries, c_sum) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
    for ax, a in zip(axes, ["noise", "fgsm", "pgd", "segpgd"]):
        _eps_axis(ax, label=False)
        for g, key, color, marker in (("Anatomy", "mIoU_anatomy", GROUP_COLORS["Anatomy"], "o"),
                                      ("Instruments", "mIoU_instruments", GROUP_COLORS["Instruments"], "s")):
            xs, ys = [0], [100.0]
            for e in EPS_ALL:
                rid = run_id(a, e)
                if rid in summaries:
                    xs.append(EPS_X[e])
                    ys.append(100 * summaries[rid][key] / c_sum[key])
            ax.plot(xs, ys, color=color, marker=marker, label=g, markeredgecolor="white", markeredgewidth=1.2)
        ax.set_title(ATTACK_LABEL[a].replace(" (control)", ""), loc="left")
        ax.set_xlabel("ε (/255)")
        ax.set_ylim(-3, 108)
    axes[0].set_ylabel("Group mIoU, % of its clean value")
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle("Anatomy vs. instruments: share of clean mIoU retained under attack "
                 "(grey band: supplementary budgets below one 8-bit step)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_anatomy_vs_instruments.png"))
    plt.close(fig)


def fig_heatmap(t3: pd.DataFrame) -> None:
    cols = ["clean"] + [f"{a}_eps{e:g}" for a in ["noise", "fgsm", "pgd", "segpgd"] for e in EPS_ALL]
    cols = [c for c in cols if c in t3.columns]
    data = 100 * t3[cols].to_numpy(dtype=float)
    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(0.62 * len(cols) + 3.5, 0.45 * len(t3) + 1.9))
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 55 else TEXT)
    ylabels = [f"{r['class']}{' *' if r['rare'] else ''}" for _, r in t3.iterrows()]
    ax.set_yticks(range(len(t3)), ylabels)
    xlabels = ["Clean"]
    for c in cols[1:]:
        a, e = c.split("_eps")
        e_s = {"0.25": "¼", "0.5": "½"}.get(e, e)
        xlabels.append(f"{SHORT_LABEL[a].replace('-20', '')}\n{e_s}")
    ax.set_xticks(range(len(cols)), xlabels, fontsize=7.5)
    ax.grid(False)
    attacks_in_cols = [c.split("_eps")[0] for c in cols]
    for j in range(1, len(cols)):
        if attacks_in_cols[j] != attacks_in_cols[j - 1]:
            ax.axvline(j - 0.5, color="white", linewidth=3)
    n_anat = sum(t3.group == "Anatomy")
    ax.axhline(n_anat - 0.5, color="white", linewidth=3)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("IoU (%)")
    cb.outline.set_visible(False)
    ax.set_title("Per-class IoU (%) on the test fold, clean and under attack. Column = attack and ε (/255); "
                 "¼ and ½ are supplementary; * rare class", loc="left", fontsize=10)
    fig.savefig(os.path.join(FIG, "fig3_per_class_iou_heatmap.png"))
    plt.close(fig)


def fig_class_fall(size_df: pd.DataFrame) -> None:
    panels = [(a, e) for a, e in VULN_PANELS if ((size_df.attack == a) & (size_df.eps_255 == e)).any()]
    if not panels:
        return
    last = size_df[(size_df.attack == panels[-1][0]) & (size_df.eps_255 == panels[-1][1]) & (~size_df.rare)]
    order = list(last.sort_values("rel_fall_pct")["class"])
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels) + 1.5, 0.42 * len(order) + 1.7),
                             sharey=True, squeeze=False)
    for ax, (a, e) in zip(axes[0], panels):
        sub = size_df[(size_df.attack == a) & (size_df.eps_255 == e) & (~size_df.rare)].set_index("class").loc[order]
        colors = [GROUP_COLORS["Anatomy"] if g == "Anatomy" else GROUP_COLORS["Instruments"] for g in sub.group]
        ax.barh(range(len(sub)), sub.rel_fall_pct.clip(lower=0), color=colors, height=0.68)
        for i, v in enumerate(sub.rel_fall_pct):
            ax.text(max(v, 0) + 1.5, i, f"{v:.0f}%", va="center", fontsize=8, color=TEXT2)
        ax.set_title(f"{SHORT_LABEL[a]}, ε = {e:g}/255", loc="left")
        ax.set_xlabel("Relative IoU fall (%)")
        ax.set_xlim(0, 115)
        ax.grid(axis="y", visible=False)
    axes[0, 0].set_yticks(range(len(order)), order)
    axes[0, -1].legend(handles=[Patch(color=GROUP_COLORS["Anatomy"], label="Anatomy"),
                                Patch(color=GROUP_COLORS["Instruments"], label="Instrument")],
                       loc="lower right", fontsize=9)
    fig.suptitle("Per-class vulnerability in three unsaturated regimes (headline classes)", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_per_class_relative_fall.png"))
    plt.close(fig)


def fig_ablation(t5: pd.DataFrame, c_sum) -> None:
    if t5.empty:
        return
    budgets = [e for e in (4, 1) if (t5.eps_255 == e).any()]
    fig, axes = plt.subplots(1, len(budgets), figsize=(6 * len(budgets), 4.2), sharey=True, squeeze=False)
    for ax, e in zip(axes[0], budgets):
        sub_e = t5[t5.eps_255 == e]
        ax.axhline(100 * c_sum["mIoU"], color=TEXT2, linewidth=1)
        ax.text(1, 100 * c_sum["mIoU"] + 1.2, "clean", color=TEXT2, fontsize=9)
        for name, key in (("PGD", "pgd"), ("SegPGD", "segpgd")):
            sub = sub_e[sub_e.attack == name].sort_values("steps")
            ax.plot(sub.steps, 100 * sub.mIoU, color=COLORS[key], marker=MARKERS[key],
                    label=f"{name} (α = ε/4)", markeredgecolor="white", markeredgewidth=1.2)
        fg = sub_e[sub_e.attack == "FGSM"]
        if not fg.empty:
            ax.plot(fg.steps, 100 * fg.mIoU, linestyle="none", color=COLORS["fgsm"], marker=MARKERS["fgsm"],
                    label="FGSM (1 step)", markeredgecolor="white", markeredgewidth=1.2, markersize=9)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 5, 10, 20, 40], ["1", "5", "10", "20", "40"])
        ax.set_xlabel("Attack iterations")
        ax.set_ylim(0, 100 * c_sum["mIoU"] + 6)
        ax.set_title(f"ε = {e}/255" + (" (spec)" if e == 4 else " (supplementary)"), loc="left")
    axes[0, 0].set_ylabel("Headline mIoU (%)")
    axes[0, -1].legend(fontsize=9)
    fig.suptitle("Step-count ablation", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_step_ablation.png"))
    plt.close(fig)


def _place_labels(fig, ax, points) -> None:
    """Annotate (x, y, text) points, trying four offsets per label until it overlaps no earlier label."""
    renderer = fig.canvas.get_renderer()
    candidates = [((5, 3), "left"), ((5, -10), "left"), ((-5, 3), "right"), ((-5, -10), "right"),
                  ((5, 12), "left"), ((-5, 12), "right")]
    boxes = []
    for x, y, text in sorted(points, key=lambda p: p[0]):
        for offset, ha in candidates:
            ann = ax.annotate(text, (x, y), xytext=offset, textcoords="offset points", ha=ha,
                              fontsize=7.5, color=TEXT2)
            box = ann.get_window_extent(renderer).expanded(1.05, 1.15)
            if not any(box.overlaps(b) for b in boxes):
                boxes.append(box)
                break
            ann.remove()
        else:  # every candidate collides: keep the default placement
            ann = ax.annotate(text, (x, y), xytext=candidates[0][0], textcoords="offset points",
                              fontsize=7.5, color=TEXT2)
            boxes.append(ann.get_window_extent(renderer))


def fig_size(size_df: pd.DataFrame, spearman) -> None:
    panels = [(a, e) for a, e in VULN_PANELS if ((size_df.attack == a) & (size_df.eps_255 == e)).any()]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.4), sharey=True, squeeze=False)
    for ax, (a, e) in zip(axes[0], panels):
        sub = size_df[(size_df.attack == a) & (size_df.eps_255 == e)]
        for g, color, marker in (("Anatomy", GROUP_COLORS["Anatomy"], "o"),
                                 ("Instrument", GROUP_COLORS["Instruments"], "s")):
            s = sub[(sub.group == g) & (~sub.rare)]
            ax.scatter(s.mean_area_px, s.rel_fall_pct, s=60, color=color, marker=marker, label=g,
                       edgecolor="white", linewidth=1.5, zorder=3)
        ax.set_xscale("log")
        ax.set_ylim(-5, 105)
        _place_labels(fig, ax, [(r.mean_area_px, r.rel_fall_pct,
                                 r["class"].replace("Phacoemulsification", "Phaco.").replace("Capsulorhexis", "Caps."))
                                for _, r in sub[~sub.rare].iterrows()])
        ax.set_xlabel("Mean GT area per frame (px, log)")
        rho = spearman.get(run_id(a, e))
        rho_s = f"ρ = {rho[0]:+.2f}, p = {rho[1]:.2f}" if rho else ""
        ax.set_title(f"{SHORT_LABEL[a]}, ε = {e:g}/255   {rho_s}", loc="left", fontsize=10)
    axes[0, 0].set_ylabel("Relative IoU fall (%)")
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle("Object size vs. vulnerability (headline classes; Spearman ρ across classes)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_object_size_vs_fall.png"))
    plt.close(fig)


def _class_legend(fig, classes: List[int], y: float = 0.0) -> None:
    handles = [Patch(color=PALETTE[c] / 255.0, label=CLASS_NAMES[c]) for c in classes]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)), fontsize=8,
               bbox_to_anchor=(0.5, y), frameon=False)


def fig_qualitative(runs) -> None:
    qd = os.path.join(RES, "qualitative")
    ref_path = os.path.join(qd, "reference.npz")
    if not os.path.isfile(ref_path):
        return
    ref = np.load(ref_path)
    names, imgs, gts, cps = ref["frame_names"], ref["images"], ref["gt"], ref["clean_pred"]
    k = len(names)
    present = list(range(1, NUM_CLASSES))  # attacked predictions may contain any class

    def load(rid):
        p = os.path.join(qd, f"{rid}.npz")
        return np.load(p) if os.path.isfile(p) else None

    # (a) Full panel for SegPGD-20 at eps = 8/255 and 4/255.
    for e in (4, 8):
        q = load(run_id("segpgd", e))
        if q is None:
            continue
        fig, axes = plt.subplots(k, 6, figsize=(18, 3.1 * k + 0.6))
        axes = np.atleast_2d(axes)
        titles = ["Clean frame", f"Adversarial frame (ε = {e}/255)", "Perturbation δ (±ε → full range)",
                  "Ground truth", "Clean prediction", "Prediction under attack"]
        for i in range(k):
            img = imgs[i].transpose(1, 2, 0) / 255.0
            delta = q["delta"][i].astype(np.float32).transpose(1, 2, 0)
            adv = np.clip(img + delta, 0, 1)
            vis_delta = np.clip(delta / (2 * e / 255.0) + 0.5, 0, 1)
            panels = [img, adv, vis_delta, colorize_mask(gts[i]), colorize_mask(cps[i]), colorize_mask(q["pred"][i])]
            for j, p in enumerate(panels):
                axes[i, j].imshow(p)
                axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
                for sp in axes[i, j].spines.values():
                    sp.set_visible(False)
                if i == 0:
                    axes[i, j].set_title(titles[j], fontsize=10)
            axes[i, 0].set_ylabel(str(names[i]).replace(".png", ""), fontsize=9, color=TEXT2)
        _class_legend(fig, present, y=-0.01)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(os.path.join(FIG, f"fig7_qualitative_segpgd_eps{e}.png"), dpi=110)
        plt.close(fig)

    # (b) Attack comparison at eps = 4/255.
    cols = [("Clean frame", None), ("Ground truth", "gt"), ("Clean prediction", "clean")] + \
           [(f"{SHORT_LABEL[a]} ε=4/255", run_id(a, 4)) for a in ATTACKS]
    loaded = {rid: load(rid) for _, rid in cols if rid not in (None, "gt", "clean")}
    fig, axes = plt.subplots(k, len(cols), figsize=(2.9 * len(cols), 2.4 * k + 0.6))
    axes = np.atleast_2d(axes)
    for i in range(k):
        for j, (title, rid) in enumerate(cols):
            if rid is None:
                p = imgs[i].transpose(1, 2, 0)
            elif rid == "gt":
                p = colorize_mask(gts[i])
            elif rid == "clean":
                p = colorize_mask(cps[i])
            else:
                p = colorize_mask(loaded[rid]["pred"][i]) if loaded.get(rid) is not None else np.ones((8, 8, 3))
            axes[i, j].imshow(p)
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            for sp in axes[i, j].spines.values():
                sp.set_visible(False)
            if i == 0:
                axes[i, j].set_title(title, fontsize=9)
    _class_legend(fig, present, y=-0.01)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(FIG, "fig8_qualitative_attack_comparison_eps4.png"), dpi=110)
    plt.close(fig)

    # (c) Budget progression for one frame: adversarial images (top) and predictions (bottom).
    i = 0
    prog = [e for e in EPS_ALL if os.path.isfile(os.path.join(qd, f"{run_id('segpgd', e)}.npz"))]
    qs = {e: load(run_id("segpgd", e)) for e in prog}
    if not prog:
        return
    fig, axes = plt.subplots(2, len(prog) + 1, figsize=(3.2 * (len(prog) + 1), 6.6))
    img = imgs[i].transpose(1, 2, 0) / 255.0
    axes[0, 0].imshow(img); axes[0, 0].set_title("Clean frame", fontsize=10)
    axes[1, 0].imshow(colorize_mask(cps[i])); axes[1, 0].set_title("Clean prediction", fontsize=10)
    for j, e in enumerate(prog, start=1):
        q = qs[e]
        adv = np.clip(img + q["delta"][i].astype(np.float32).transpose(1, 2, 0), 0, 1)
        axes[0, j].imshow(adv); axes[0, j].set_title(f"Attacked, ε = {e:g}/255", fontsize=10)
        axes[1, j].imshow(colorize_mask(q["pred"][i])); axes[1, j].set_title(f"Prediction, ε = {e:g}/255", fontsize=10)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    _class_legend(fig, present, y=-0.01)
    fig.suptitle(f"SegPGD-20 at increasing budgets, frame {str(names[i]).replace('.png', '')} "
                 "(¼ and ½ /255 are supplementary, below one 8-bit step)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(FIG, "fig9_qualitative_budget_progression.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
