"""
build_paper_assets.py
=====================
Generates everything data-dependent in the IEEE paper from the experiment outputs:
  paper/figures/*.pdf|png   figures sized for IEEE columns (3.45 in / 7.16 in), Times font
  paper/tables/*.tex        booktabs tables
  paper/numbers.tex         \\newcommand macros for every number quoted in the text

Run from the project root:  python paper/build_paper_assets.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from cataract_dataset import CLASS_NAMES  # noqa: E402
from visualize import PALETTE, colorize_mask  # noqa: E402

RES, RES908 = os.path.join(ROOT, "results"), os.path.join(ROOT, "results_908frames")
TAB = os.path.join(RES, "tables")
FIGDIR, TABDIR = os.path.join(HERE, "figures"), os.path.join(HERE, "tables")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(TABDIR, exist_ok=True)

COL_W, FULL_W = 3.45, 7.16
COLORS = {"segpgd": "#2a78d6", "pgd": "#eb6834", "fgsm": "#1baf7a", "noise": "#8f8e8a"}
MARKERS = {"segpgd": "^", "pgd": "s", "fgsm": "o", "noise": "D"}
LABEL = {"noise": "Random noise", "fgsm": "FGSM", "pgd": "PGD-20", "segpgd": "SegPGD-20"}
ANAT_C, INST_C = "#2a78d6", "#eb6834"
GRID, TEXT2 = "#e4e3df", "#52514e"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 7.5, "axes.titlesize": 7.8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.8, "axes.edgecolor": "#bdbcb6", "axes.linewidth": 0.6, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.labelcolor": "#2b2b2b", "lines.linewidth": 1.3,
    "lines.markersize": 4, "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})

T2 = pd.read_csv(os.path.join(TAB, "T2_main_grid.csv"))
T2b = pd.read_csv(os.path.join(TAB, "T2b_eps_curve_and_quantisation.csv"))
T1 = pd.read_csv(os.path.join(TAB, "T1_clean_baseline_per_class.csv"))
T3b = pd.read_csv(os.path.join(TAB, "T3b_per_class_relative_fall_pct.csv"))
T4 = pd.read_csv(os.path.join(TAB, "T4_anatomy_vs_instruments.csv"))
T4c = pd.read_csv(os.path.join(TAB, "T4c_present_frames_only.csv"))
T5 = pd.read_csv(os.path.join(TAB, "T5_step_ablation.csv"))
T7 = pd.read_csv(os.path.join(TAB, "T7_predicted_class_composition.csv"), index_col=0)
KEY = json.load(open(os.path.join(RES, "summary.json")))
KEY908 = json.load(open(os.path.join(RES908, "summary.json")))
T2_908 = pd.read_csv(os.path.join(RES908, "tables", "T2_main_grid.csv"))
T2b_908 = pd.read_csv(os.path.join(RES908, "tables", "T2b_eps_curve_and_quantisation.csv"))
T4_908 = pd.read_csv(os.path.join(RES908, "tables", "T4_anatomy_vs_instruments.csv"))
T7_908 = pd.read_csv(os.path.join(RES908, "tables", "T7_predicted_class_composition.csv"), index_col=0)
HIST = pd.read_csv(os.path.join(ROOT, "runs", "baseline", "history.csv"))

EPS_AXIS = [0, 0.25, 0.5, 1, 2, 4, 8]
EPS_TICKS = ["clean", "¼", "½", "1", "2", "4", "8"]


def run(rid, table=T2b):
    r = table[table.run == rid]
    return r.iloc[0] if len(r) else T2[T2.run == rid].iloc[0]


def eps_tex(e):
    e = float(e)
    return {0.25: r"$\frac{1}{4}$", 0.5: r"$\frac{1}{2}$"}.get(e, f"{e:g}")


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------
def fig_miou_eps():
    clean = T2[T2.run == "clean"].iloc[0]
    fig, ax = plt.subplots(figsize=(COL_W, 2.25))
    ax.axvspan(0.5, 2.5, color="#f0efec", zorder=0, lw=0)
    for a in ("noise", "fgsm", "pgd", "segpgd"):
        sub = T2b[(T2b.attack == a) & (~T2b.quantised_8bit)].sort_values("eps_255")
        x = [0] + [EPS_AXIS.index(float(e)) for e in sub.eps_255]
        y = [100 * clean.mIoU] + list(100 * sub.mIoU)
        lo = [100 * clean.mIoU] + list(100 * sub.mIoU_lo)
        hi = [100 * clean.mIoU] + list(100 * sub.mIoU_hi)
        ax.fill_between(x, lo, hi, color=COLORS[a], alpha=0.18, lw=0)
        ax.plot(x, y, color=COLORS[a], marker=MARKERS[a], label=LABEL[a], mec="white", mew=0.6, zorder=3)
    ax.set_xticks(range(len(EPS_AXIS)), EPS_TICKS)
    ax.set_xlim(-0.25, len(EPS_AXIS) - 0.75)
    ax.set_ylim(0, 85)
    ax.set_xlabel(r"Perturbation budget $\varepsilon$ ($\times$1/255, $L_\infty$)")
    ax.set_ylabel("Headline mIoU (%)")
    ax.text(1.5, 44, "below one\n8-bit step", ha="center", va="center", fontsize=6.3, color=TEXT2)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.1), handlelength=1.8)
    fig.savefig(os.path.join(FIGDIR, "fig_miou_eps.pdf"))
    plt.close(fig)


def fig_groups():
    clean = T2[T2.run == "clean"].iloc[0]
    fig, axes = plt.subplots(1, 4, figsize=(FULL_W, 1.72), sharey=True)
    for ax, a in zip(axes, ("noise", "fgsm", "pgd", "segpgd")):
        ax.axvspan(0.5, 2.5, color="#f0efec", zorder=0, lw=0)
        sub = T2b[(T2b.attack == a) & (~T2b.quantised_8bit)].sort_values("eps_255")
        x = [0] + [EPS_AXIS.index(float(e)) for e in sub.eps_255]
        for key, col, mk, lab in (("mIoU_anatomy", ANAT_C, "o", "Anatomy"), ("mIoU_instruments", INST_C, "s", "Instruments")):
            y = [100.0] + list(100 * sub[key] / clean[key])
            ax.plot(x, y, color=col, marker=mk, label=lab, mec="white", mew=0.6)
        ax.set_title(LABEL[a], loc="left", fontweight="bold")
        ax.set_xticks(range(len(EPS_AXIS)), EPS_TICKS)
        ax.set_xlim(-0.3, len(EPS_AXIS) - 0.7)
        ax.set_ylim(-3, 106)
        ax.set_xlabel(r"$\varepsilon$ ($\times$1/255)")
    axes[0].set_ylabel("% of clean group mIoU")
    axes[0].legend(loc="lower left")
    fig.tight_layout(w_pad=0.6)
    fig.savefig(os.path.join(FIGDIR, "fig_groups.pdf"))
    plt.close(fig)


def fig_class_fall():
    panels = [("noise_eps8", "Random noise, $\\varepsilon$ = 8/255"), ("fgsm_eps1", "FGSM, $\\varepsilon$ = 1/255"),
              ("segpgd_eps0.25", "SegPGD-20, $\\varepsilon$ = ¼/255")]
    sub = T3b[~T3b.rare].copy().sort_values("segpgd_eps0.25")
    short = {"Phacoemulsification Tip": "Phaco tip", "Irrigation-Aspiration": "Irrigation-aspiration",
             "Capsulorhexis Cystotome": "Caps. cystotome", "Capsulorhexis Forceps": "Caps. forceps",
             "Lens (IOL)": "Lens (IOL)"}
    names = [short.get(c, c) for c in sub["class"]]
    colors = [ANAT_C if g == "Anatomy" else INST_C for g in sub.group]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.05), sharey=True)
    for ax, (col, title) in zip(axes, panels):
        v = sub[col].clip(lower=0).to_numpy()
        ax.barh(range(len(v)), v, color=colors, height=0.68)
        for i, val in enumerate(sub[col]):
            ax.text(max(val, 0) + 1.5, i, f"{val:.0f}", va="center", fontsize=6, color=TEXT2)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlim(0, 112)
        ax.set_xlabel("Relative IoU fall (%)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(names)), names)
    axes[0].legend(handles=[Patch(color=ANAT_C, label="Anatomy"), Patch(color=INST_C, label="Instrument")],
                   loc="lower right")
    fig.tight_layout(w_pad=0.8)
    fig.savefig(os.path.join(FIGDIR, "fig_class_fall.pdf"))
    plt.close(fig)


def fig_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.6), sharey=True)
    clean = 100 * T2[T2.run == "clean"].iloc[0].mIoU
    for ax, e in zip(axes, (1, 4)):
        s = T5[T5.eps_255 == e]
        for name, key in (("PGD", "pgd"), ("SegPGD", "segpgd")):
            r = s[s.attack == name].sort_values("steps")
            ax.plot(r.steps, 100 * r.mIoU, color=COLORS[key], marker=MARKERS[key], label=name, mec="white", mew=0.6)
        f = s[s.attack == "FGSM"]
        ax.plot(f.steps, 100 * f.mIoU, ls="none", color=COLORS["fgsm"], marker="o", ms=4.5, label="FGSM",
                mec="white", mew=0.6)
        ax.axhline(clean, color=TEXT2, lw=0.6)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 5, 10, 20, 40], ["1", "5", "10", "20", "40"])
        ax.minorticks_off()
        ax.set_ylim(0, 85)
        ax.set_title(f"$\\varepsilon$ = {e}/255", loc="left", fontweight="bold")
        ax.set_xlabel("Attack iterations")
    axes[0].set_ylabel("Headline mIoU (%)")
    axes[1].legend(loc="center right", bbox_to_anchor=(1.0, 0.55))
    fig.tight_layout(w_pad=0.5)
    fig.savefig(os.path.join(FIGDIR, "fig_ablation.pdf"))
    plt.close(fig)


def fig_qualitative():
    qd = os.path.join(RES, "qualitative")
    ref = np.load(os.path.join(qd, "reference.npz"))
    adv = np.load(os.path.join(qd, "segpgd_eps4_T20.npz"))
    rows = [0, 1, 3]
    cols = ["Clean frame", "Adversarial ($\\varepsilon$=4/255)", "Perturbation (rescaled)", "Ground truth",
            "Clean prediction", "Prediction under attack"]
    fig, axes = plt.subplots(len(rows), 6, figsize=(FULL_W, 0.105 + 1.2 * len(rows)))
    for r, i in enumerate(rows):
        img = ref["images"][i].transpose(1, 2, 0) / 255.0
        d = adv["delta"][i].astype(np.float32).transpose(1, 2, 0)
        panels = [img, np.clip(img + d, 0, 1), np.clip(d / (2 * 4 / 255.0) + 0.5, 0, 1),
                  colorize_mask(ref["gt"][i]), colorize_mask(ref["clean_pred"][i]), colorize_mask(adv["pred"][i])]
        for c, p in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(p[::2, ::2], interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(cols[c], fontsize=6.8)
    handles = [Patch(color=PALETTE[c] / 255.0, label=CLASS_NAMES[c]) for c in range(1, 13)]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=5.6, bbox_to_anchor=(0.5, -0.075),
               handlelength=1.2, columnspacing=0.9)
    fig.subplots_adjust(wspace=0.03, hspace=0.04, left=0.005, right=0.995, top=0.94, bottom=0.04)
    fig.savefig(os.path.join(FIGDIR, "fig_qualitative.png"), dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------------------------
def p2(x):
    return f"{100 * x:.2f}"


def write(name, text):
    with open(os.path.join(TABDIR, name), "w", encoding="utf-8") as f:
        f.write(text)


def tab_clean():
    pc = KEY["paper_comparison"]
    paper = {1: pc["Iris (dataset label: Cornea)"]["paper_iou"], 2: pc["Pupil"]["paper_iou"],
             3: pc["Lens (IOL)"]["paper_iou"]}
    lines = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
             r"Class & Group & Frames & IoU & Dice & Ref.~\cite{ghamsarian2024cataract} \\", r"\midrule"]
    for _, r in T1[T1.id > 0].iterrows():
        name = r["class"].replace("Phacoemulsification", "Phaco.").replace("Capsulorhexis", "Caps.") \
            .replace("Irrigation-Aspiration", "Irrigation-aspiration").replace("Slit/Incision Knife", "Slit/incision knife")
        if r["rare"]:
            name += r"$^\ast$"
        grp = "Anat." if r["group"] == "Anatomy" else "Instr."
        ref = f"{paper[r['id']]:.2f}" if r["id"] in paper else "--"
        lines.append(f"{name} & {grp} & {r['test_frames']} & {p2(r['IoU'])} & {p2(r['Dice'])} & {ref} \\\\")
    c = KEY["clean"]
    ib = np.load(os.path.join(RES, "raw", "clean.npz"))["inst_binary"]
    inst_frames = int(((ib[:, 0] + ib[:, 2]) > 0).sum())  # test frames containing any instrument
    lines += [r"\midrule",
              f"Instruments merged & -- & {inst_frames} & {pc['Instruments (merged)']['ours_iou']:.2f} & {pc['Instruments (merged)']['ours_dice']:.2f} & "
              f"{pc['Instruments (merged)']['paper_iou']:.2f} \\\\",
              r"\midrule",
              f"\\textbf{{Headline mean}} & 10 cl. & 561 & \\textbf{{{p2(c['mIoU'])}}} & \\textbf{{{p2(c['mDice'])}}} & -- \\\\",
              f"Anatomy mean & 3 cl. & 561 & {p2(c['mIoU_anatomy'])} & {p2(c['mDice_anatomy'])} & -- \\\\",
              f"Instrument mean & 7 cl. & 561 & {p2(c['mIoU_instruments'])} & {p2(c['mDice_instruments'])} & -- \\\\",
              r"\bottomrule", r"\end{tabular}"]
    write("tab_clean.tex", "\n".join(lines) + "\n")


def tab_main():
    order = ["clean"] + [f"noise_eps{e}" for e in (1, 2, 4, 8)] + [f"fgsm_eps{e}" for e in (1, 2, 4, 8)] + \
            [f"{a}_eps{e}_T20" for a in ("pgd", "segpgd") for e in (1, 2, 4, 8)]
    lab = {"clean": "Clean", "noise": "Random noise", "fgsm": "FGSM", "pgd": "PGD-20", "segpgd": "SegPGD-20"}
    lines = [r"\begin{tabular}{@{}llrrrrrrrr@{}}", r"\toprule",
             r"Attack & $\varepsilon$ & mIoU [95\% CI] & $\Delta$mIoU & Rel. fall & mDice & Anat. & Instr. & "
             r"Corr.$\to$wrong & PSNR \\",
             r" & ($\times\frac{1}{255}$) & (\%) & (pp) & (\%) & (\%) & mIoU (\%) & mIoU (\%) & (\%) & (dB) \\",
             r"\midrule"]
    prev = None
    for rid in order:
        r = T2[T2.run == rid].iloc[0]
        a = rid.split("_")[0]
        if prev is not None and a != prev:
            lines.append(r"\addlinespace[2pt]")
        prev = a
        if rid == "clean":
            lines.append(f"\\textbf{{Clean}} & -- & \\textbf{{{p2(r.mIoU)}}} [{p2(r.mIoU_lo)}, {p2(r.mIoU_hi)}] & -- & -- & "
                         f"{p2(r.mDice)} & {p2(r.mIoU_anatomy)} & {p2(r.mIoU_instruments)} & -- & -- \\\\")
            continue
        lines.append(f"{lab[a]} & {r.eps_255:g} & {p2(r.mIoU)} [{p2(r.mIoU_lo)}, {p2(r.mIoU_hi)}] & {100 * r.delta_mIoU:.2f} & "
                     f"{r.rel_fall_pct:.1f} & {p2(r.mDice)} & {p2(r.mIoU_anatomy)} & {p2(r.mIoU_instruments)} & "
                     f"{r.correct_to_wrong_pct:.1f} & {r.psnr_db:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_main.tex", "\n".join(lines) + "\n")


def tab_ablation():
    lines = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
             r"$\varepsilon$ & Attack & $T{=}1$ & 5 & 10 & 20 & 40 \\", r"\midrule"]
    for e in (1, 4):
        s = T5[T5.eps_255 == e]
        fg = p2(s[s.attack == "FGSM"].mIoU.iloc[0])
        for name in ("PGD", "SegPGD"):
            vals = [p2(s[(s.attack == name) & (s.steps == t)].mIoU.iloc[0]) for t in (5, 10, 20, 40)]
            first = f"{fg}$^\\ddagger$" if name == "PGD" else "--"
            lines.append(f"{e}/255 & {name} & {first} & " + " & ".join(vals) + r" \\")
        if e == 1:
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_ablation.tex", "\n".join(lines) + "\n")


def tab_groups():
    sel = [("Noise", 8), ("FGSM", 0.25), ("FGSM", 1), ("FGSM", 8), ("PGD-20", 0.25), ("PGD-20", 1), ("PGD-20", 8),
           ("SegPGD-20", 0.25), ("SegPGD-20", 1), ("SegPGD-20", 8)]
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"Attack & $\varepsilon$ & Anat. & Instr. & $\Delta$ [95\% CI] & $\Delta_{\mathrm{pres}}$ & Vid. \\",
             r"\midrule"]
    prev = None
    for a, e in sel:
        r = T4[(T4.attack == a) & (T4.eps_255 == e)].iloc[0]
        rc = T4c[(T4c.attack == a) & (T4c.eps_255 == e)].iloc[0]
        if prev is not None and a != prev:
            lines.append(r"\addlinespace[1.5pt]")
        prev = a
        lines.append(f"{a.replace('Noise', 'Noise')} & {eps_tex(e)} & {r.anatomy_rel_fall_pct:.1f} & "
                     f"{r.instrument_rel_fall_pct:.1f} & ${r.diff_pp:+.1f}$ [${r.diff_lo:+.1f}$, ${r.diff_hi:+.1f}$] & "
                     f"${rc.diff_present_only_pp:+.1f}$ & {int(r.videos_inst_worse)}/{int(r.videos)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_groups.tex", "\n".join(lines) + "\n")


def tab_composition():
    rows = ["Background", "Cornea", "Pupil", "Lens (IOL)", "Gauge", "Spatula", "Phacoemulsification Tip",
            "Lens Injector"]
    cols = ["Ground truth", "Clean", "FGSM 4/255", "PGD-20 4/255", "SegPGD-20 4/255"]
    lines = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r"Class & GT & Clean & FGSM & PGD-20 & SegPGD-20 \\", r"\midrule"]
    for name in rows:
        v = T7.loc[name, cols]
        nm = name.replace("Phacoemulsification Tip", "Phaco tip")
        cells = []
        for c, x in zip(cols, v):
            s = f"{x:.1f}"
            if c.startswith(("PGD", "SegPGD")) and x > 3 * max(T7.loc[name, "Ground truth"], 0.05):
                s = f"\\textbf{{{s}}}"
            cells.append(s)
        lines.append(f"{nm} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_composition.tex", "\n".join(lines) + "\n")


def tab_perclass():
    T3 = pd.read_csv(os.path.join(TAB, "T3_per_class_iou.csv"))
    cols = [("clean", "Clean"), ("noise_eps8", "Noise 8"), ("fgsm_eps1", "FGSM 1"), ("pgd_eps0.25", r"PGD $\frac{1}{4}$"),
            ("segpgd_eps0.25", r"SegPGD $\frac{1}{4}$"), ("segpgd_eps1", "SegPGD 1")]
    short = {"Phacoemulsification Tip": "Phaco tip", "Irrigation-Aspiration": "Irrigation-asp.",
             "Capsulorhexis Cystotome": "Caps. cystotome", "Capsulorhexis Forceps": "Caps. forceps",
             "Slit/Incision Knife": "Slit/inc. knife"}
    lines = [r"\begin{tabular}{@{}l" + "r" * len(cols) + r"@{}}", r"\toprule",
             "Class & " + " & ".join(h for _, h in cols) + r" \\", r"\midrule"]
    for i, (_, r) in enumerate(T3.iterrows()):
        if i == 3:
            lines.append(r"\addlinespace[2pt]")
        name = short.get(r["class"], r["class"]) + (r"$^\ast$" if r["rare"] else "")
        lines.append(name + " & " + " & ".join(f"{100 * r[c]:.1f}" for c, _ in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_perclass.tex", "\n".join(lines) + "\n")


def tab_checks():
    lines = [r"\begin{tabular}{@{}lrr@{}}", r"\toprule",
             r"\multicolumn{3}{@{}l}{\emph{(a) 8-bit quantisation of the adversarial frame}} \\",
             r"mIoU (\%) & Float & 8-bit \\", r"\midrule"]
    for a, lab in (("pgd", "PGD-20"), ("segpgd", "SegPGD-20"), ("fgsm", "FGSM")):
        for e in (1.0, 2.0):
            f = T2b[(T2b.attack == a) & (T2b.eps_255 == e) & (~T2b.quantised_8bit)].iloc[0]
            q = T2b[(T2b.attack == a) & (T2b.eps_255 == e) & (T2b.quantised_8bit)].iloc[0]
            lines.append(f"{lab}, $\\varepsilon$ = {e:g}/255 & {p2(f.mIoU)} & {p2(q.mIoU)} \\\\")
    lines += [r"\midrule",
              r"\multicolumn{3}{@{}l}{\emph{(b) Replication on an independent run}} \\",
              r"Quantity & 908 frames & Full data \\", r"\midrule"]
    a, b = T2_908[T2_908.run == "clean"].iloc[0], T2[T2.run == "clean"].iloc[0]
    lines.append(f"Clean mIoU (\\%) & {p2(a.mIoU)} & {p2(b.mIoU)} \\\\")
    for rid, lab in (("noise_eps8", "Noise 8/255"), ("fgsm_eps1", "FGSM 1/255"),
                     ("pgd_eps1_T20", "PGD-20 1/255"), ("segpgd_eps1_T20", "SegPGD-20 1/255")):
        a, b = T2_908[T2_908.run == rid].iloc[0], T2[T2.run == rid].iloc[0]
        lines.append(f"{lab}: rel.\\ fall (\\%) & {a.rel_fall_pct:.1f} & {b.rel_fall_pct:.1f} \\\\")
    for a_, e, lab in (("FGSM", 1, "FGSM 1/255"), ("PGD-20", 0.25, r"PGD-20 $\frac{1}{4}$/255")):
        ra = T4_908[(T4_908.attack == a_) & (T4_908.eps_255 == e)].iloc[0]
        rb = T4[(T4.attack == a_) & (T4.eps_255 == e)].iloc[0]
        lines.append(f"{lab}: gap, pp (videos) & ${ra.diff_pp:+.1f}$ ({int(ra.videos_inst_worse)}/6) & "
                     f"${rb.diff_pp:+.1f}$ ({int(rb.videos_inst_worse)}/6) \\\\")
    pa, pb = T7_908.loc["Pupil", "SegPGD-20 4/255"], T7.loc["Pupil", "SegPGD-20 4/255"]
    lines.append(f"SegPGD-20 4/255: pupil share (\\%) & {pa:.1f} & {pb:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_checks.tex", "\n".join(lines) + "\n")


# ----------------------------------------------------------------------------------------------
# Numbers quoted in the text
# ----------------------------------------------------------------------------------------------
def numbers():
    g = {r["run"]: r for r in KEY["main_grid"]}
    c = KEY["clean"]
    M = {}

    def put(name, val):
        M[name] = val

    put("CleanMIoU", p2(c["mIoU"])); put("CleanLo", p2(c["mIoU_ci"][0])); put("CleanHi", p2(c["mIoU_ci"][1]))
    put("CleanMDice", p2(c["mDice"])); put("CleanAnat", p2(c["mIoU_anatomy"])); put("CleanInst", p2(c["mIoU_instruments"]))
    for rid, nm in (("pgd_eps1_T20", "PGDOne"), ("segpgd_eps1_T20", "SegOne"), ("pgd_eps8_T20", "PGDEight"),
                    ("segpgd_eps8_T20", "SegEight"), ("fgsm_eps1", "FGSMOne"), ("fgsm_eps8", "FGSMEight"),
                    ("noise_eps1", "NoiseOne"), ("noise_eps8", "NoiseEight")):
        put(nm, p2(g[rid]["mIoU"])); put(nm + "Fall", f"{g[rid]['rel_fall_pct']:.1f}")
    fg = [g[f"fgsm_eps{e}"]["rel_fall_pct"] for e in (1, 2, 4, 8)]
    put("FGSMFallMin", f"{min(fg):.1f}"); put("FGSMFallMax", f"{max(fg):.1f}")
    it = [g[f"{a}_eps{e}_T20"]["rel_fall_pct"] for a in ("pgd", "segpgd") for e in (1, 2, 4, 8)]
    put("IterFallMin", f"{min(it):.1f}"); put("IterFallMax", f"{max(it):.1f}")
    put("PSNRPGDOne", f"{g['pgd_eps1_T20']['psnr_db']:.1f}"); put("PSNRFGSMOne", f"{g['fgsm_eps1']['psnr_db']:.1f}")
    cw_seg = [g[f"segpgd_eps{e}_T20"]["correct_to_wrong_pct"] for e in (1, 2, 4, 8)]
    cw_pgd = [g[f"pgd_eps{e}_T20"]["correct_to_wrong_pct"] for e in (1, 2, 4, 8)]
    put("CWSegMin", f"{min(cw_seg):.0f}"); put("CWSegMax", f"{max(cw_seg):.0f}")
    put("CWPGDMin", f"{min(cw_pgd):.0f}"); put("CWPGDMax", f"{max(cw_pgd):.0f}")
    for a, nm in (("pgd", "PGDQ"), ("segpgd", "SegQ"), ("fgsm", "FGSMQ")):
        r = run(f"{a}_eps0.25" + ("_T20" if a != "fgsm" else ""))
        put(nm, p2(r.mIoU)); put(nm + "Fall", f"{r.rel_fall_pct:.1f}")
    # quantisation
    q = T2b[T2b.quantised_8bit]
    diffs = []
    for _, r in q.iterrows():
        f = T2b[(T2b.attack == r.attack) & (T2b.eps_255 == r.eps_255) & (~T2b.quantised_8bit)].iloc[0]
        diffs.append(abs(r.mIoU - f.mIoU) * 100)
    put("QuantMax", f"{max(diffs):.2f}")
    # groups
    def gap(a, e, col="diff_pp", t=T4):
        return t[(t.attack == a) & (t.eps_255 == e)].iloc[0][col]
    put("GapNoiseEight", f"{gap('Noise', 8):.1f}")
    put("GapFGSMQ", f"{gap('FGSM', 0.25):.1f}"); put("GapFGSMEight", f"{gap('FGSM', 8):.1f}")
    put("GapFGSMOne", f"{gap('FGSM', 1):.1f}")
    put("GapPGDQ", f"{gap('PGD-20', 0.25):.1f}"); put("GapSegQ", f"{gap('SegPGD-20', 0.25):.1f}")
    put("GapPGDOne", f"{gap('PGD-20', 1):.1f}"); put("GapSegOne", f"{gap('SegPGD-20', 1):.1f}")
    put("GapPresPGDQ", f"{gap('PGD-20', 0.25, 'diff_present_only_pp', T4c):.1f}")
    put("GapPresSegQ", f"{gap('SegPGD-20', 0.25, 'diff_present_only_pp', T4c):.1f}")
    put("AnatFallPGDQ", f"{gap('PGD-20', 0.25, 'anatomy_rel_fall_pct'):.1f}")
    put("InstFallPGDQ", f"{gap('PGD-20', 0.25, 'instrument_rel_fall_pct'):.1f}")
    put("AnatFallFGSMOne", f"{gap('FGSM', 1, 'anatomy_rel_fall_pct'):.1f}")
    put("InstFallFGSMOne", f"{gap('FGSM', 1, 'instrument_rel_fall_pct'):.1f}")
    # composition
    put("PupilGT", f"{T7.loc['Pupil', 'Ground truth']:.1f}"); put("PupilSeg", f"{T7.loc['Pupil', 'SegPGD-20 4/255']:.1f}")
    put("PupilPGD", f"{T7.loc['Pupil', 'PGD-20 4/255']:.1f}")
    put("InjGT", f"{T7.loc['Lens Injector', 'Ground truth']:.1f}"); put("InjSeg", f"{T7.loc['Lens Injector', 'SegPGD-20 4/255']:.1f}")
    put("BgGT", f"{T7.loc['Background', 'Ground truth']:.1f}"); put("BgSeg", f"{T7.loc['Background', 'SegPGD-20 4/255']:.1f}")
    # ablation
    s1 = T5[T5.eps_255 == 1]
    put("PGDFiveOne", p2(s1[(s1.attack == "PGD") & (s1.steps == 5)].mIoU.iloc[0]))
    put("PGDFortyOne", p2(s1[(s1.attack == "PGD") & (s1.steps == 40)].mIoU.iloc[0]))
    put("PGDFiveOneFall", f"{100 * (1 - s1[(s1.attack == 'PGD') & (s1.steps == 5)].mIoU.iloc[0] / c['mIoU']):.0f}")
    # size correlation (non-saturated, non-trivial runs)
    sp = KEY["size_vs_fall_spearman"]
    T4b = pd.read_csv(os.path.join(TAB, "T4b_object_size_vs_fall.csv"))
    rhos, ps = [], []
    for rid, (rho, p, n) in sp.items():
        m = T4b[(T4b.run == rid) & (~T4b.rare)].rel_fall_pct.mean()
        if 1.0 < m < 95:
            rhos.append(rho); ps.append(p)
    put("RhoMin", f"{min(rhos):.2f}"); put("RhoMax", f"{max(rhos):.2f}"); put("RhoPMin", f"{min(ps):.2f}")
    # training
    best = HIST.loc[HIST.val_mIoU.idxmax()]
    put("BestEpoch", f"{int(best.epoch)}"); put("ValBest", p2(best.val_mIoU)); put("TrainMin", f"{HIST.epoch_seconds.sum() / 60:.0f}")
    # 908 replication
    put("RepPGDOne", f"{T2_908[T2_908.run == 'pgd_eps1_T20'].iloc[0].rel_fall_pct:.1f}")
    put("RepNoiseEight", f"{T2_908[T2_908.run == 'noise_eps8'].iloc[0].rel_fall_pct:.1f}")
    # paper comparison
    pc = KEY["paper_comparison"]
    put("RefIris", f"{pc['Iris (dataset label: Cornea)']['paper_iou']:.2f}"); put("OurIris", f"{pc['Iris (dataset label: Cornea)']['ours_iou']:.2f}")
    put("RefPupil", f"{pc['Pupil']['paper_iou']:.2f}"); put("OurPupil", f"{pc['Pupil']['ours_iou']:.2f}")
    put("RefLens", f"{pc['Lens (IOL)']['paper_iou']:.2f}"); put("OurLens", f"{pc['Lens (IOL)']['ours_iou']:.2f}")
    with open(os.path.join(HERE, "numbers.tex"), "w", encoding="utf-8") as f:
        f.write("% Auto-generated by build_paper_assets.py -- do not edit by hand.\n")
        for k, v in M.items():
            f.write(f"\\newcommand{{\\n{k}}}{{{v}}}\n")
    return M


if __name__ == "__main__":
    fig_miou_eps(); fig_groups(); fig_class_fall(); fig_ablation(); fig_qualitative()
    tab_clean(); tab_main(); tab_ablation(); tab_groups(); tab_composition(); tab_checks(); tab_perclass()
    M = numbers()
    print(f"figures: {sorted(os.listdir(FIGDIR))}")
    print(f"tables:  {sorted(os.listdir(TABDIR))}")
    print(f"{len(M)} number macros:", ", ".join(f"{k}={v}" for k, v in M.items()))
