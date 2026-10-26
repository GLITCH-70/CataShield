"""
run_experiments.py
==================
Runs the Version 1 evaluation grid on the held-out test fold (Fold 1) with the frozen baseline.

Main grid (spec section 25: 1 clean + 12 attacked runs)
  clean
  FGSM       eps in {1, 2, 4, 8}/255
  PGD-20     eps in {1, 2, 4, 8}/255, alpha = eps/4, random start
  SegPGD-20  eps in {1, 2, 4, 8}/255, alpha = eps/4, random start
Control (added for interpretation; no gradient information)
  random +/-eps sign noise, eps in {1, 2, 4, 8}/255
Step-count ablation (spec section 25)
  PGD and SegPGD at eps = 4/255 with T in {5, 10, 20, 40}  (T = 20 reuses the main-grid run)
  Supplementary: the same at eps = 1/255.
Supplementary runs (added after the spec grid showed PGD/SegPGD saturating already at 1/255)
  Sub-8-bit budgets: FGSM, PGD-20, SegPGD-20 at eps in {0.25, 0.5}/255. These perturbations are
    below 8-bit resolution, so they only exist in a float (decoded-frame) pipeline; they map the
    transition region of the eps curve, where anatomy and instruments separate.
  8-bit quantised: FGSM, PGD-20, SegPGD-20 at eps in {1, 2}/255 with the adversarial frame rounded
    to the 8-bit grid before inference, i.e. as it would be after saving to PNG. Clean frames are
    already on that grid, so the rounded perturbation still satisfies ||delta||_inf <= eps.

For every run, results/raw/<run_id>.npz stores per-frame, per-class (TP, FP, FN) counts plus
per-frame attack-success counts, so analyze_results.py can recompute every metric, bootstrap
confidence intervals and run paired tests without touching the GPU again. Runs whose file exists
are skipped, so the script can be resumed after an interruption.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import zlib
from typing import Dict, List, Tuple

import numpy as np
import torch

from attacks import check_perturbation, run_attack
from cataract_dataset import CataractDataset, INSTRUMENT_CLASSES, NUM_CLASSES
from metrics import format_report, frame_stats, summarize_counts
from model_builder import load_trained_model
from visualize import to_jsonable

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EPS_LIST = [1, 2, 4, 8]
ABLATION_EPS = [4, 1]  # 4/255 per spec; 1/255 supplementary
ABLATION_STEPS = [5, 10, 20, 40]
MAIN_STEPS = 20
SUB8BIT_EPS = [0.25, 0.5]
QUANTISED_EPS = [1, 2]
MIN_INSTRUMENT = min(INSTRUMENT_CLASSES)


def build_run_list() -> List[Dict[str, object]]:
    runs = [{"id": "clean", "attack": "clean", "eps": 0, "steps": 0}]
    for attack in ("noise", "fgsm", "pgd", "segpgd"):
        for e in EPS_LIST:
            steps = MAIN_STEPS if attack in ("pgd", "segpgd") else (1 if attack == "fgsm" else 0)
            runs.append({"id": f"{attack}_eps{e}" + (f"_T{steps}" if attack in ("pgd", "segpgd") else ""),
                         "attack": attack, "eps": e, "steps": steps})
    for e in ABLATION_EPS:
        for attack in ("pgd", "segpgd"):
            for t in ABLATION_STEPS:
                if t != MAIN_STEPS:
                    runs.append({"id": f"{attack}_eps{e}_T{t}", "attack": attack, "eps": e, "steps": t})
    for quantise, eps_list in ((False, SUB8BIT_EPS), (True, QUANTISED_EPS)):
        for e in eps_list:
            for attack in ("fgsm", "pgd", "segpgd"):
                steps = 1 if attack == "fgsm" else MAIN_STEPS
                rid = f"{attack}_eps{e:g}" + (f"_T{steps}" if attack != "fgsm" else "") + ("_q8" if quantise else "")
                runs.append({"id": rid, "attack": attack, "eps": e, "steps": steps, "quantise": quantise})
    return runs


def pick_qualitative_frames(ds: CataractDataset, k: int) -> List[int]:
    """Greedy pick of k test frames that together cover the most instrument classes."""
    present = []
    inst_area = []
    for m in ds.masks:
        counts = torch.bincount(m.flatten().long(), minlength=NUM_CLASSES).numpy()
        present.append({c for c in INSTRUMENT_CLASSES if counts[c] > 0})
        inst_area.append(sum(counts[c] for c in INSTRUMENT_CLASSES))
    chosen: List[int] = []
    covered: set = set()
    cases_used: set = set()
    for _ in range(k):
        def score(i):
            new = len(present[i] - covered)
            other_case = ds.samples[i][2] not in cases_used
            return (new, other_case, len(present[i]), inst_area[i])
        best = max((i for i in range(len(ds)) if i not in chosen), key=score)
        chosen.append(best)
        covered |= present[best]
        cases_used.add(ds.samples[best][2])
    return sorted(chosen)


def main():
    parser = argparse.ArgumentParser(description="Clean + adversarial evaluation grid on Cataract-1K Fold 1.")
    parser.add_argument("--ckpt", default="runs/baseline/best_model.pth")
    parser.add_argument("--out", default="results")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--qual-frames", type=int, default=4)
    parser.add_argument("--only", nargs="*", default=None, help="subset of run ids to execute")
    parser.add_argument("--max-frames", type=int, default=None, help="debug: evaluate only the first N frames")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cpu":
        torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    out_dir = os.path.join(BASE_DIR, args.out)
    raw_dir = os.path.join(out_dir, "raw")
    qual_dir = os.path.join(out_dir, "qualitative")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(qual_dir, exist_ok=True)

    model = load_trained_model(os.path.join(BASE_DIR, args.ckpt), device)
    ds = CataractDataset(BASE_DIR, "test")
    n = len(ds) if args.max_frames is None else min(args.max_frames, len(ds))
    frame_names = [os.path.basename(s[0]) for s in ds.samples[:n]]
    qual_idx = [i for i in pick_qualitative_frames(ds, args.qual_frames) if i < n]
    print(f"test fold: {n} frames from {len(ds.target_cases)} videos | device {device}")
    print(f"qualitative frames: {[frame_names[i] for i in qual_idx]}")

    # Clean predictions are the reference for "pixels changed" / "correct pixels flipped".
    clean_pred = torch.empty((n, *ds.img_size), dtype=torch.uint8)
    with torch.no_grad():
        for s in range(0, n, args.batch_size):
            x = ds.images[s:s + args.batch_size].to(device).float() / 255.0
            clean_pred[s:s + x.shape[0]] = model(x).argmax(1).to(torch.uint8).cpu()
    gt = ds.masks[:n]
    np.savez_compressed(os.path.join(qual_dir, "reference.npz"),
                        frame_names=np.array([frame_names[i] for i in qual_idx]),
                        images=ds.images[qual_idx].numpy(), gt=gt[qual_idx].numpy(),
                        clean_pred=clean_pred[qual_idx].numpy())

    runs = build_run_list()
    if args.only:
        runs = [r for r in runs if r["id"] in args.only]
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump({"runs": runs, "test_frames": n, "ckpt": args.ckpt, "alpha": "eps/4",
                   "random_start": True, "eps_units": "x/255 in [0,1] RGB before normalisation"},
                  f, indent=2)

    for run in runs:
        path = os.path.join(raw_dir, f"{run['id']}.npz")
        if os.path.isfile(path):
            print(f"[skip] {run['id']} (already computed)")
            continue
        eps = run["eps"] / 255.0
        seed = zlib.crc32(run["id"].encode()) & 0x7FFFFFFF
        gen = torch.Generator(device=device).manual_seed(seed)

        stats, changed, clean_correct, flipped_correct, mse, inst_bin = [], [], [], [], [], []
        linf_max = 0.0
        qual_delta, qual_pred = {}, {}
        t0 = time.time()
        for s in range(0, n, args.batch_size):
            x = ds.images[s:s + args.batch_size].to(device).float() / 255.0
            y = gt[s:s + x.shape[0]].to(device).long()
            x_adv = run_attack(run["attack"], model, x, y, eps, run["steps"], gen)
            if run.get("quantise"):
                x_adv = torch.round(x_adv * 255.0) / 255.0
            linf_max = max(linf_max, check_perturbation(x, x_adv, eps, tol=1e-5))
            mse.append((x_adv - x).pow(2).flatten(1).mean(1).cpu().numpy())
            with torch.no_grad():
                pred = model(x_adv).argmax(1)
            stats.append(frame_stats(pred, y))
            # Binary "any instrument" stats (class >= 4), comparable to Cataract-1K Table 7.
            inst_bin.append(frame_stats((pred >= MIN_INSTRUMENT).long(), (y >= MIN_INSTRUMENT).long(), 2)[:, 1])
            cp = clean_pred[s:s + x.shape[0]].to(device).long()
            was_right = cp == y
            changed.append((pred != cp).flatten(1).sum(1).cpu().numpy())
            clean_correct.append(was_right.flatten(1).sum(1).cpu().numpy())
            flipped_correct.append((was_right & (pred != y)).flatten(1).sum(1).cpu().numpy())
            for j in range(x.shape[0]):
                if s + j in qual_idx:
                    qual_delta[s + j] = (x_adv[j] - x[j]).cpu().numpy().astype(np.float16)
                    qual_pred[s + j] = pred[j].cpu().numpy().astype(np.uint8)
        seconds = time.time() - t0

        stats_arr = np.concatenate(stats)
        np.savez_compressed(
            path, stats=stats_arr, changed=np.concatenate(changed),
            clean_correct=np.concatenate(clean_correct), flipped_correct=np.concatenate(flipped_correct),
            mse=np.concatenate(mse), inst_binary=np.concatenate(inst_bin),
            frame_names=np.array(frame_names), linf_max=linf_max, seconds=seconds,
            attack=run["attack"], eps=run["eps"], steps=run["steps"],
        )
        if qual_idx and run["attack"] != "clean":
            np.savez_compressed(os.path.join(qual_dir, f"{run['id']}.npz"),
                                delta=np.stack([qual_delta[i] for i in qual_idx]),
                                pred=np.stack([qual_pred[i] for i in qual_idx]))
        s_ = summarize_counts(stats_arr)
        fr = 100 * np.concatenate(flipped_correct).sum() / max(np.concatenate(clean_correct).sum(), 1)
        print(f"{run['id']:18s} eps {run['eps']}/255 T={run['steps']:2d} | mIoU {100 * s_['mIoU']:6.2f} "
              f"anat {100 * s_['mIoU_anatomy']:6.2f} tools {100 * s_['mIoU_instruments']:6.2f} "
              f"mDice {100 * s_['mDice']:6.2f} | correct->wrong {fr:5.1f}% | "
              f"||d||inf {linf_max * 255:.3f}/255 | {seconds:6.1f}s", flush=True)
        if run["id"] == "clean":
            report = format_report(s_, "Clean test fold (Fold 1)")
            print(report)
            with open(os.path.join(out_dir, "clean_test_report.txt"), "w") as f:
                f.write(report + "\n")
            with open(os.path.join(out_dir, "clean_test_metrics.json"), "w") as f:
                json.dump(to_jsonable(s_), f, indent=2)

    print("all runs complete")


if __name__ == "__main__":
    main()
