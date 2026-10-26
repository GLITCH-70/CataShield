"""
build_results_pdf.py
====================
Builds the detailed Results & Verification reference document (PDF) for the Version 1 project.

All tables are generated from the experiment outputs, so the PDF always matches the data:
  results/tables/*.csv, results/summary.json, results/run_config.json, results/raw/*.npz,
  runs/baseline/history.csv, results_908frames/ (replication run), dataset cache (class counts).
Narrative text is written here and reflects the results at the time of writing (October 2026).

Usage:
    python report/build_results_pdf.py
Output:
    report/Cataract1K_Adversarial_Robustness_Results_and_Verification.pdf
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from datetime import date

import numpy as np
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether,
                                NextPageTemplate, PageBreak, PageTemplate, Paragraph, Spacer,
                                Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
RES = os.path.join(ROOT, "results")
RES908 = os.path.join(ROOT, "results_908frames")
TAB = os.path.join(RES, "tables")
FIG = os.path.join(RES, "figures")
OUT = os.path.join(HERE, "Cataract1K_Adversarial_Robustness_Results_and_Verification.pdf")

# ----------------------------------------------------------------------------------------------
# Fonts and styles (DejaVu ships with matplotlib and covers Greek letters, arrows, ≤, ±, ¼, ✔)
# ----------------------------------------------------------------------------------------------
import matplotlib

FONT_DIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
for name, file in (("DejaVu", "DejaVuSans.ttf"), ("DejaVu-Bold", "DejaVuSans-Bold.ttf"),
                   ("DejaVu-Oblique", "DejaVuSans-Oblique.ttf"), ("DejaVu-BoldOblique", "DejaVuSans-BoldOblique.ttf"),
                   ("DejaVuMono", "DejaVuSansMono.ttf")):
    pdfmetrics.registerFont(TTFont(name, os.path.join(FONT_DIR, file)))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold",
                              italic="DejaVu-Oblique", boldItalic="DejaVu-BoldOblique")

NAVY = colors.HexColor("#1F3A5F")
BLUE = colors.HexColor("#2a78d6")
INK = colors.HexColor("#1b1b1b")
INK2 = colors.HexColor("#52514e")
RULE = colors.HexColor("#C9D3E0")
HEAD_FILL = colors.HexColor("#E8EEF6")
ALT_FILL = colors.HexColor("#F7F9FC")
NOTE_FILL = colors.HexColor("#F3F6FA")

ST = {
    "body": ParagraphStyle("body", fontName="DejaVu", fontSize=9.4, leading=13.6, textColor=INK, spaceAfter=6),
    "bullet": ParagraphStyle("bullet", fontName="DejaVu", fontSize=9.4, leading=13.6, textColor=INK,
                             leftIndent=13, bulletIndent=3, spaceAfter=3.5),
    "H1": ParagraphStyle("H1", keepWithNext=1, fontName="DejaVu-Bold", fontSize=15.5, leading=20, textColor=NAVY,
                         spaceBefore=6, spaceAfter=9),
    "H2": ParagraphStyle("H2", keepWithNext=1, fontName="DejaVu-Bold", fontSize=11.6, leading=15, textColor=NAVY,
                         spaceBefore=11, spaceAfter=5),
    "H3": ParagraphStyle("H3", keepWithNext=1, fontName="DejaVu-Bold", fontSize=9.8, leading=13, textColor=INK,
                         spaceBefore=7, spaceAfter=3),
    "cap": ParagraphStyle("cap", fontName="DejaVu", fontSize=8.1, leading=11, textColor=INK2,
                          spaceBefore=3, spaceAfter=10),
    "tcap": ParagraphStyle("tcap", keepWithNext=1, fontName="DejaVu", fontSize=8.1, leading=11, textColor=INK2,
                           spaceBefore=6, spaceAfter=4),
    "toctitle": ParagraphStyle("toctitle", fontName="DejaVu-Bold", fontSize=15.5, leading=20, textColor=NAVY,
                               spaceAfter=12),
    "cell": ParagraphStyle("cell", fontName="DejaVu", fontSize=7.7, leading=9.8, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="DejaVu-Bold", fontSize=7.7, leading=9.8, textColor=INK),
    "celln": ParagraphStyle("celln", fontName="DejaVu", fontSize=7.7, leading=9.8, textColor=INK, alignment=TA_RIGHT),
    "cellh": ParagraphStyle("cellh", fontName="DejaVu-Bold", fontSize=7.5, leading=9.4, textColor=NAVY),
    "cellhn": ParagraphStyle("cellhn", fontName="DejaVu-Bold", fontSize=7.5, leading=9.4, textColor=NAVY,
                             alignment=TA_RIGHT),
    "small": ParagraphStyle("small", fontName="DejaVu", fontSize=7.8, leading=10.5, textColor=INK2),
    "code": ParagraphStyle("code", fontName="DejaVuMono", fontSize=8, leading=11, textColor=INK,
                           backColor=NOTE_FILL, borderPadding=5, leftIndent=6, rightIndent=6,
                           spaceBefore=4, spaceAfter=8),
    "toc0": ParagraphStyle("toc0", fontName="DejaVu-Bold", fontSize=9.6, leading=15, textColor=NAVY),
    "toc1": ParagraphStyle("toc1", fontName="DejaVu", fontSize=8.8, leading=12.5, leftIndent=16, textColor=INK),
    "tile_num": ParagraphStyle("tile_num", fontName="DejaVu-Bold", fontSize=19, leading=23, textColor=NAVY,
                               alignment=TA_CENTER),
    "tile_lab": ParagraphStyle("tile_lab", fontName="DejaVu", fontSize=7.6, leading=10, textColor=INK2,
                               alignment=TA_CENTER),
}

PAGE_W, PAGE_H = A4
MARGIN = 1.9 * cm
TEXT_W = PAGE_W - 2 * MARGIN
LAND_W = landscape(A4)[0] - 2 * MARGIN
LAND_H = landscape(A4)[1] - 2 * MARGIN - 1.2 * cm


# ----------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------
def P(text, style="body"):
    return Paragraph(text, ST[style])


def bullets(items):
    return [Paragraph(t, ST["bullet"], bulletText="•") for t in items]


def H1(text):
    return Paragraph(text, ST["H1"])


def H2(text):
    return Paragraph(text, ST["H2"])


def H3(text):
    return Paragraph(text, ST["H3"])


def esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def pc(x, d=2):
    """fraction -> percent string"""
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}"


def num(x, d=1, signed=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:+.{d}f}" if signed else f"{x:.{d}f}"


def eps_s(e) -> str:
    return {0.25: "¼", 0.5: "½"}.get(float(e), f"{float(e):g}") + "/255"


class Counter:
    def __init__(self):
        self.t = 0
        self.f = 0

    def table(self):
        self.t += 1
        return self.t

    def fig(self):
        self.f += 1
        return self.f


N = Counter()


def table(rows, widths, caption=None, header_rows=1, numeric_cols=None, bold_first_col=False,
          zebra=True, font_scale=1.0, keep=True):
    """rows: list of lists of str (header first). numeric_cols: indexes right-aligned."""
    numeric_cols = set(numeric_cols or [])
    data = []
    for r, row in enumerate(rows):
        cells = []
        for c, val in enumerate(row):
            if r < header_rows:
                st = ST["cellhn"] if c in numeric_cols else ST["cellh"]
            elif c in numeric_cols:
                st = ST["celln"]
            elif bold_first_col and c == 0:
                st = ST["cellb"]
            else:
                st = ST["cell"]
            if font_scale != 1.0:
                st = ParagraphStyle(st.name + "s", parent=st, fontSize=st.fontSize * font_scale,
                                    leading=st.leading * font_scale)
            cells.append(val if not isinstance(val, str) else Paragraph(val, st))
        data.append(cells)
    t = Table(data, colWidths=widths, repeatRows=header_rows)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, header_rows - 1), HEAD_FILL),
        ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.8, NAVY),
        ("LINEBELOW", (0, header_rows), (-1, -1), 0.25, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 2.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
    ]
    if zebra:
        for r in range(header_rows, len(rows)):
            if (r - header_rows) % 2 == 1:
                style.append(("BACKGROUND", (0, r), (-1, r), ALT_FILL))
    t.setStyle(TableStyle(style))
    # The caption is keep-with-next, so the document groups [heading,] caption and table and moves
    # them to the next page together when the table would otherwise be separated from them.
    if caption:
        n = N.table()
        return [P(f"<b>Table {n}.</b> {caption}", "tcap"), t, Spacer(1, 8)]
    return [t, Spacer(1, 8)]


_tmp = tempfile.mkdtemp(prefix="pdfassets_")


def figure(path, caption, max_w=TEXT_W, max_h=21 * cm, photo=False):
    """Scaled figure + caption; large photo-like panels are re-encoded to keep the PDF small."""
    im = PILImage.open(path)
    w, h = im.size
    src = path
    if photo or w > 2400:
        scale = min(1.0, 2400 / w)
        im2 = im.convert("RGB").resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
        src = os.path.join(_tmp, os.path.splitext(os.path.basename(path))[0] + ".jpg")
        im2.save(src, quality=90)
    ratio = h / w
    width = min(max_w, max_h / ratio)
    img = Image(src, width=width, height=width * ratio)
    img.keepWithNext = 1  # image stays with its caption (and a preceding heading)
    n = N.fig()
    return [img, P(f"<b>Figure {n}.</b> {caption}", "cap")]


def note_box(text):
    t = Table([[Paragraph(text, ST["body"])]], colWidths=[TEXT_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), NOTE_FILL),
                           ("LINEBEFORE", (0, 0), (0, -1), 2.5, BLUE),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return [t, Spacer(1, 8)]


def tiles(items, width=TEXT_W):
    cells = [[Paragraph(v, ST["tile_num"]) for v, _ in items], [Paragraph(l, ST["tile_lab"]) for _, l in items]]
    t = Table(cells, colWidths=[width / len(items)] * len(items))
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), NOTE_FILL),
                           ("LINEAFTER", (0, 0), (-2, -1), 1.5, colors.white),
                           ("TOPPADDING", (0, 0), (-1, 0), 9), ("BOTTOMPADDING", (0, 1), (-1, 1), 9),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return [t, Spacer(1, 10)]


def to_landscape():
    return [NextPageTemplate("landscape"), PageBreak()]


def to_portrait():
    return [NextPageTemplate("portrait"), PageBreak()]


# ----------------------------------------------------------------------------------------------
# Document template: cover / portrait / landscape, TOC + PDF outline, "Page x of y"
# ----------------------------------------------------------------------------------------------
RUN_TITLE = "Adversarial Robustness of Cataract Surgical Segmentation — Results & Verification (V1)"


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved = []

    def showPage(self):
        self._saved.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved)
        for state in self._saved:
            self.__dict__.update(state)
            if self._pageNumber > 1:
                w, _ = self._pagesize
                self.setFont("DejaVu", 7.5)
                self.setFillColor(INK2)
                self.drawRightString(w - MARGIN, 1.15 * cm, f"Page {self._pageNumber} of {total}")
            super().showPage()
        super().save()


def _decorate(c, doc):
    w, h = c._pagesize
    c.saveState()
    c.setStrokeColor(RULE)
    c.setLineWidth(0.6)
    c.line(MARGIN, h - 1.35 * cm, w - MARGIN, h - 1.35 * cm)
    c.line(MARGIN, 1.5 * cm, w - MARGIN, 1.5 * cm)
    c.setFont("DejaVu", 7.5)
    c.setFillColor(INK2)
    c.drawString(MARGIN, h - 1.15 * cm, RUN_TITLE)
    c.drawString(MARGIN, 1.15 * cm, "BVCOE (GGSIPU) · B.Tech CSE Micro-Project 2026 · for reference use")
    c.restoreState()


def _cover(c, doc):
    w, h = c._pagesize
    c.saveState()
    c.setFillColor(NAVY)
    c.rect(0, h - 9.2 * cm, w, 9.2 * cm, fill=1, stroke=0)
    c.setFillColor(BLUE)
    c.rect(0, h - 9.45 * cm, w, 0.25 * cm, fill=1, stroke=0)
    c.restoreState()


class ReportDoc(BaseDocTemplate):
    def __init__(self, filename, **kw):
        super().__init__(filename, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                         topMargin=MARGIN, bottomMargin=MARGIN, **kw)
        fp = Frame(MARGIN, MARGIN, PAGE_W - 2 * MARGIN, PAGE_H - 2 * MARGIN, id="fp")
        lw, lh = landscape(A4)
        fl = Frame(MARGIN, MARGIN, lw - 2 * MARGIN, lh - 2 * MARGIN, id="fl")
        self.addPageTemplates([
            PageTemplate(id="cover", frames=[fp], onPage=_cover, pagesize=A4),
            PageTemplate(id="portrait", frames=[fp], onPage=_decorate, pagesize=A4),
            PageTemplate(id="landscape", frames=[fl], onPage=_decorate, pagesize=landscape(A4)),
        ])

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name in ("H1", "H2"):
            text = flowable.getPlainText()
            level = 0 if flowable.style.name == "H1" else 1
            key = "k" + re.sub(r"\W+", "_", text)[:60]
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=level > 0)
            self.notify("TOCEntry", (level, text, self.page, key))


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
def load():
    D = {}
    D["key"] = json.load(open(os.path.join(RES, "summary.json")))
    D["key908"] = json.load(open(os.path.join(RES908, "summary.json")))
    D["cfg"] = json.load(open(os.path.join(RES, "run_config.json")))
    for t in ("T1_clean_baseline_per_class", "T2_main_grid", "T2b_eps_curve_and_quantisation", "T3_per_class_iou",
              "T3b_per_class_relative_fall_pct", "T4_anatomy_vs_instruments", "T4b_object_size_vs_fall",
              "T4c_present_frames_only", "T5_step_ablation"):
        D[t] = pd.read_csv(os.path.join(TAB, t + ".csv"))
    D["T7"] = pd.read_csv(os.path.join(TAB, "T7_predicted_class_composition.csv"), index_col=0)
    for t in ("T2_main_grid", "T2b_eps_curve_and_quantisation", "T4_anatomy_vs_instruments"):
        D["908_" + t] = pd.read_csv(os.path.join(RES908, "tables", t + ".csv"))
    D["T7_908"] = pd.read_csv(os.path.join(RES908, "tables", "T7_predicted_class_composition.csv"), index_col=0)
    D["hist"] = pd.read_csv(os.path.join(ROOT, "runs", "baseline", "history.csv"))
    runs = []
    for r in D["cfg"]["runs"]:
        z = np.load(os.path.join(RES, "raw", r["id"] + ".npz"))
        runs.append({**r, "seconds": float(z["seconds"]), "linf": float(z["linf_max"]) * 255,
                     "n": int(z["stats"].shape[0])})
    D["runs"] = runs
    # class frames per split, from the dataset cache
    from cataract_dataset import CLASS_NAMES, OFFICIAL_FOLDS, DEFAULT_VAL_CASES, CataractDataset, get_split_cases
    D["CLASS_NAMES"], D["FOLDS"], D["VAL"] = CLASS_NAMES, OFFICIAL_FOLDS, DEFAULT_VAL_CASES
    D["splits"] = get_split_cases()
    D["split_counts"] = {}
    for s in ("train", "val", "test"):
        ds = CataractDataset(ROOT, s)
        D["split_counts"][s] = (len(ds), ds.class_frame_counts())
    import glob
    D["fold_frames"] = {f: sum(len(glob.glob(os.path.join(ROOT, "dataset", f"{c}_*.png"))) for c in cs)
                        for f, cs in OFFICIAL_FOLDS.items()}
    return D


# ----------------------------------------------------------------------------------------------
# Content
# ----------------------------------------------------------------------------------------------
def cover(story, D):
    k = D["key"]
    story.append(Spacer(1, 1.3 * cm))
    story.append(Paragraph("RESULTS &amp; VERIFICATION REPORT · VERSION 1",
                           ParagraphStyle("kick", fontName="DejaVu-Bold", fontSize=9, leading=12,
                                          textColor=colors.HexColor("#B9CCE6"))))
    story.append(Spacer(1, 0.35 * cm))
    story.append(Paragraph("Adversarial Attack and Robustness Analysis of AI-Based Cataract Surgical "
                           "Instrument Segmentation",
                           ParagraphStyle("ct", fontName="DejaVu-Bold", fontSize=21, leading=27,
                                          textColor=colors.white)))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("DeepLabV3+ (ResNet-50) on Cataract-1K under FGSM, PGD-20 and SegPGD-20 white-box "
                           "L<font size=6>∞</font> attacks — complete experimental record, findings and "
                           "verification against the problem statement",
                           ParagraphStyle("cs", fontName="DejaVu", fontSize=10.5, leading=15,
                                          textColor=colors.HexColor("#DCE6F3"))))
    story.append(Spacer(1, 2.6 * cm))
    story += tiles([(f"{pc(k['clean']['mIoU'])}%", "clean headline mIoU<br/>(held-out Fold 1, 561 frames)"),
                    ("4.35%", "mIoU under PGD-20<br/>at ε = 1/255"),
                    ("−95%", "relative fall from one<br/>intensity level per pixel"),
                    ("6 / 6", "test videos: instruments<br/>degrade faster than anatomy")])
    story.append(Spacer(1, 0.6 * cm))
    rows = [
        ["Institution", "Bharati Vidyapeeth’s College of Engineering (BVCOE, GGSIPU), New Delhi — Department of CSE"],
        ["Programme", "B.Tech Computer Science &amp; Engineering, Semester Micro-Project 2026"],
        ["Faculty guide", "Dr. Jolly Parikh, Associate Professor, Department of CSE"],
        ["Project team", "Parikshat Singh (04611502724) · Venu Kulshreshtha (02111502724) · Sayyad Ahsan (03411502724)"],
        ["Experiments", "2–3 October 2026 · full Cataract-1K segmentation subset (2,256 annotated frames, 30 videos)"],
        ["Document date", date.today().strftime("%d %B %Y")],
        ["Status", "Version 1 complete: attack + robustness analysis (no defences; defences are Version 2)"],
    ]
    t = Table([[Paragraph(f"<b>{a}</b>", ST["cell"]), Paragraph(b, ST["cell"])] for a, b in rows],
              colWidths=[3.2 * cm, TEXT_W - 3.2 * cm])
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.25, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5)]))
    story.append(t)
    story.append(Spacer(1, 0.5 * cm))
    story.append(P("Every number in this document is generated from the experiment outputs "
                   "(<font name='DejaVuMono'>results/</font>, <font name='DejaVuMono'>runs/</font>) by "
                   "<font name='DejaVuMono'>report/build_results_pdf.py</font>; re-running the script after new "
                   "experiments regenerates all tables.", "small"))
    story += [NextPageTemplate("portrait"), PageBreak()]


def toc_page(story):
    story.append(Paragraph("Contents", ST["toctitle"]))
    toc = TableOfContents()
    toc.levelStyles = [ST["toc0"], ST["toc1"]]
    toc.dotsMinLevel = 0
    story.append(toc)
    story.append(PageBreak())


def executive_summary(story, D):
    k = D["key"]
    g = {r["run"]: r for r in k["main_grid"]}
    story.append(H1("1. Executive Summary"))
    story.append(P(
        "This project measured how vulnerable an AI model that segments cataract surgery video frames is to "
        "<i>adversarial perturbations</i>: small, deliberately computed changes to the input image that are "
        "invisible to a human. A DeepLabV3+ network with an ImageNet-pretrained ResNet-50 encoder was trained on "
        "the Cataract-1K segmentation dataset (3 anatomy and 9 instrument classes), evaluated on the official "
        "held-out patient fold, and then attacked with three white-box gradient attacks (FGSM, PGD-20, SegPGD-20) "
        "at perturbation budgets of 1–8 intensity levels per pixel, plus a random-noise control and several "
        "supplementary checks. In total 41 evaluation runs were made on 561 held-out frames."))
    story += tiles([(f"{pc(k['clean']['mIoU'])}%", "clean mIoU<br/>(95% CI "
                     f"{pc(k['clean']['mIoU_ci'][0])}–{pc(k['clean']['mIoU_ci'][1])})"),
                    (f"{pc(g['pgd_eps1_T20']['mIoU'])}%", "PGD-20, ε = 1/255"),
                    (f"{pc(g['segpgd_eps1_T20']['mIoU'])}%", "SegPGD-20, ε = 1/255"),
                    (f"{g['noise_eps8']['rel_fall_pct']:.1f}%", "loss from random noise<br/>at ε = 8/255")])
    story.append(H2("Key findings"))
    story += bullets([
        f"<b>The clean baseline is strong.</b> {pc(k['clean']['mIoU'])}% headline mIoU and "
        f"{pc(k['clean']['mDice'])}% mean Dice on the full held-out Fold 1; every anatomy class meets or exceeds "
        "the published Cataract-1K DeepLabV3+ result. The vulnerability below cannot be blamed on a weak model.",
        "<b>The model is extremely fragile.</b> Changing each pixel by at most one 8-bit intensity level "
        f"(ε = 1/255, PSNR ≈ 51 dB) drops headline mIoU from {pc(k['clean']['mIoU'])}% to "
        f"{pc(g['pgd_eps1_T20']['mIoU'])}% (PGD-20) and {pc(g['segpgd_eps1_T20']['mIoU'])}% (SegPGD-20), a 95% "
        "relative fall. Even a quarter of one intensity level removes 73%.",
        "<b>The damage comes from the adversarial direction, not from noise.</b> Random ±ε noise of the same size "
        f"costs at most {g['noise_eps8']['rel_fall_pct']:.1f}% (ε = 8/255). FGSM costs 29–31% at every budget; "
        "iterative attacks cost 95–99.6%.",
        "<b>Instruments are more vulnerable than anatomy.</b> Wherever an attack is not saturated, instruments lose "
        "a larger share of their clean IoU than the cornea, pupil and lens (+3.6 to +20.7 percentage points), the "
        "bootstrap confidence intervals exclude zero, and all six test videos agree. The gap survives a control "
        "that removes hallucinated pixels.",
        "<b>The attack survives saving to an 8-bit image</b> (mIoU changes by at most 0.26 points after rounding).",
        "<b>Under iterative attack the model hallucinates.</b> At ε = 4/255 SegPGD-20 makes the model label "
        "49.5% of all pixels as pupil (true share 6.7%) and 6.0% as lens injector (true share 0.1%).",
        "<b>More training data made the model more robust to noise but not to attacks.</b> Compared with an "
        "earlier run on a 908-frame subset, noise sensitivity halved while iterative-attack vulnerability was "
        "unchanged (95% fall at 1/255 in both).",
    ])
    story += note_box(
        "<b>Verdict on the problem statement.</b> The question “How vulnerable is an AI-based cataract surgical "
        "instrument segmentation model to adversarial perturbations, and how significantly do such attacks affect "
        "its performance?” is answered: <b>extremely vulnerable</b>. Perturbations far below visibility remove "
        "almost all segmentation quality, the effect is directional rather than noise-like, it persists in stored "
        "8-bit images, and surgical instruments are the most fragile part of the scene. All ten specific "
        "objectives of the Version 1 specification are met (Section 7).")
    story.append(PageBreak())


def project_context(story, D):
    story.append(H1("2. Project Context"))
    story.append(H2("2.1 Problem statement"))
    story.append(P(
        "Deep-learning models for surgical video analysis classify every pixel of a frame into anatomical "
        "structures or surgical instruments. They are almost always evaluated on clean frames. Neural networks are, "
        "however, known to be susceptible to adversarial examples: inputs with carefully computed perturbations "
        "that are visually subtle but change the model’s output. The central question of Version 1 is:"))
    story += note_box("<b>How vulnerable is an AI-based cataract surgical instrument segmentation model to "
                      "adversarial perturbations, and how significantly do such attacks affect its performance?</b>")
    story.append(H2("2.2 Research gap"))
    story.append(P(
        "The project’s literature review (~65 papers; 22-paper gap matrix in "
        "<font name='DejaVuMono'>Paper_Gap_Matrix.md</font>) found that every neighbouring area is occupied but "
        "the intersection is empty:"))
    story += bullets([
        "Adversarial robustness of medical segmentation is studied on radiology volumes, dental X-rays and "
        "ultrasound (e.g. Malik et al., BMVC 2024), not on intra-operative video.",
        "Surgical tool-segmentation robustness benchmarks are explicitly non-adversarial (SegSTRONG-C, "
        "EndoVis 2024: smoke, bleeding, low brightness).",
        "Adversarial work inside surgery targets detection, action triplets or robot policies on laparoscopic data "
        "(GUARDIAN, MICCAI 2024 workshop), not cataract segmentation.",
        "Ophthalmic adversarial studies address diagnostic imaging (fundus, OCT), never the operating-room video.",
        "All published Cataract-1K models, including the 2025 state of the art, are evaluated on clean frames only.",
    ])
    story.append(P("The cell “gradient-based adversarial attack × cataract surgical instrument segmentation” "
                   "therefore had no matching work in the searches performed; this report fills it with a "
                   "quantified vulnerability profile, including the first anatomy-versus-instrument comparison "
                   "within the same surgical frame."))
    story.append(H2("2.3 Research questions"))
    story += table([
        ["#", "Question (Version 1 specification §6)"],
        ["Primary", "How do adversarial attacks affect the performance and robustness of AI-based cataract surgical "
                    "instrument segmentation models?"],
        ["RQ1", "How much does model performance decrease when adversarial perturbations are introduced?"],
        ["RQ2", "Which attack settings produce greater degradation?"],
        ["RQ3", "Are some surgical instruments or scene conditions more vulnerable than others?"],
        ["RQ4", "How does performance on clean inputs compare with performance on adversarial inputs?"],
        ["RQ5", "How does attack strength affect the trade-off between perturbation magnitude and degradation?"],
    ], [1.6 * cm, TEXT_W - 1.6 * cm], caption="Research questions answered in Section 6.", bold_first_col=True)
    story.append(H2("2.4 Scope"))
    story += table([
        ["In scope (Version 1)", "Out of scope (reserved for Version 2)"],
        ["Clean baseline training and evaluation; white-box untargeted L∞ attacks (FGSM, PGD, SegPGD); budget "
         "and iteration sweeps; per-class and anatomy-vs-instrument degradation analysis; qualitative panels",
         "Adversarial defences, purification, adversarial training, black-box / transfer attacks, physical or "
         "patch attacks, temporal (video) attacks, clinical safety claims"],
    ], [TEXT_W / 2] * 2, caption="Version 1 scope (specification §8 and §28). No defence was added.")


def data_section(story, D):
    CN = D["CLASS_NAMES"]
    story.append(PageBreak())
    story.append(H1("3. Data"))
    story.append(H2("3.1 Dataset"))
    story.append(P(
        "Cataract-1K (Ghamsarian et al., <i>Scientific Data</i>, 2024) contains 1,000 cataract surgery videos "
        "recorded at Klinikum Klagenfurt. Its segmentation subset (Synapse <font name='DejaVuMono'>syn53395479</font>) "
        "provides <b>2,256 annotated frames from 30 videos</b> at 1024×768, sampled one frame every five seconds, "
        "with Supervisely polygon annotations of 3 anatomy and 9 instrument classes and five published "
        "patient-disjoint folds. Licence: CC BY 4.0."))
    story.append(H2("3.2 Acquisition and integrity checks"))
    story.append(P(
        "The first web download contained images for only 908 of the 2,256 frames: Synapse packages web downloads "
        "in 1 GB parts and only the first part had been fetched. The missing 1,348 frames were downloaded directly "
        "from the Synapse folder <font name='DejaVuMono'>Annotations/Images-and-Supervisely-Annotations/&lt;case&gt;/img</font> "
        "with <font name='DejaVuMono'>download_synapse_dataset.py</font> (view+download access token; synapseclient "
        "verifies each file’s MD5). The 30 full surgery videos in the same Synapse folder are not used. "
        "Checks performed before training:"))
    story += bullets([
        "2,256 images, 2,256 annotation JSONs and 2,256 rasterised masks, pairing one-to-one by file name.",
        "All images 1024×768 and decodable; every mask has the same size as its image.",
        "All 7,585 annotation objects are hole-free polygons; 0 unknown class labels.",
        "Annotated frames per fold reproduce the paper exactly: 561 / 459 / 420 / 385 / 431.",
        "Visual overlay checks of masks on newly downloaded and original frames.",
    ])
    story.append(P("The earlier 908-frame run is retained as an independent replication (Section 5.9)."))
    story.append(H2("3.3 Classes"))
    labels = {0: "—", 1: "Cornea, cornea1", 2: "Pupil, pupil1", 3: "Lens", 4: "Slit Knife, Incision Knife",
              5: "Gauge", 6: "Capsulorhexis Cystotome", 7: "Capsulorhexis Forceps", 8: "Katena Forceps",
              9: "Spatula", 10: "Phacoemulsification Tip", 11: "Irrigation-Aspiration", 12: "Lens Injector"}
    rows = [["ID", "Class", "Group", "Supervisely label(s)", "Note"]]
    for c in range(13):
        grp = "Background" if c == 0 else ("Anatomy" if c <= 3 else "Instrument")
        note = {1: "called “iris” in the paper", 4: "merged as in the paper (9 instrument classes); rare",
                8: "rare (spec §25)"}.get(c, "")
        rows.append([str(c), CN[c], grp, labels[c], note])
    story += table(rows, [1.0 * cm, 4.2 * cm, 2.2 * cm, 4.5 * cm, TEXT_W - 11.9 * cm],
                   caption="The 13-class label map (1 background + 3 anatomy + 9 instrument classes). Polygons are "
                           "rasterised in z-order: anatomy first, instruments last, so an occluding instrument keeps "
                           "its own label.", numeric_cols=[0])
    story.append(H2("3.4 Patient-disjoint split"))
    rows = [["Fold", "Videos", "Annotated frames", "Role"]]
    for f, cases in D["FOLDS"].items():
        vids = ", ".join((f"<b>{c}</b>" if c in D["VAL"] else c) for c in cases)
        role = "<b>Test</b> (held out)" if f == 1 else "Train + 1 validation video (bold)"
        rows.append([str(f), vids, f"{D['fold_frames'][f]:,}", role])
    story += table(rows, [1.1 * cm, 9.6 * cm, 2.6 * cm, TEXT_W - 13.3 * cm],
                   caption="Official Cataract-1K folds. Fold 1 is the test set; one video from each other fold is "
                           "used for validation (model selection), the remaining 20 videos for training. No patient "
                           "appears in more than one split (asserted in code).", numeric_cols=[2])
    sc = D["split_counts"]
    rows = [["Class"] + [f"{s.capitalize()} ({sc[s][0]:,} frames)" for s in ("train", "val", "test")]]
    for c in range(1, 13):
        rows.append([CN[c]] + [f"{int(sc[s][1][c]):,}" for s in ("train", "val", "test")])
    story += table(rows, [5.5 * cm] + [(TEXT_W - 5.5 * cm) / 3] * 3,
                   caption="Number of frames in which each class appears, per split. Every foreground class occurs "
                           "in all three splits; Katena Forceps and Slit/Incision Knife are rare.",
                   numeric_cols=[1, 2, 3])


def method_section(story, D):
    h = D["hist"]
    best = h.loc[h.val_mIoU.idxmax()]
    story.append(PageBreak())
    story.append(H1("4. Methodology"))
    story.append(H2("4.1 Pipeline"))
    story += table([
        ["Stage", "File", "What it does"],
        ["Download", "download_synapse_dataset.py", "Fetches missing frames from Synapse (MD5-verified)"],
        ["Masks", "convert_supervisely_to_masks.py", "Supervisely polygons → 13-class PNG label masks"],
        ["Data / split", "cataract_dataset.py", "Official folds, patient-disjoint split, 512×512 memory-mapped cache"],
        ["Model", "model_builder.py", "DeepLabV3+ / ResNet-50 (OS16) with ImageNet normalisation inside the model"],
        ["Loss, metrics", "loss.py, metrics.py", "CE + 0.8·Dice; per-frame per-class TP/FP/FN; class-set means"],
        ["Training", "train_baseline.py", "GPU augmentation, bf16 training, best-epoch selection on validation"],
        ["Attacks", "attacks.py", "FGSM, PGD, SegPGD, random-sign noise; L∞ budget check"],
        ["Experiments", "run_experiments.py", "41 runs on the test fold; resumable; raw per-frame counts saved"],
        ["Analysis", "analyze_results.py", "Tables, bootstrap CIs, sign tests, per-video counts, figures"],
        ["This document", "report/build_results_pdf.py", "Builds this PDF from the outputs above"],
    ], [2.6 * cm, 5.2 * cm, TEXT_W - 7.8 * cm], caption="Code modules (project root).")
    story.append(H2("4.2 Model"))
    story.append(P(
        "DeepLabV3+ (Chen et al., ECCV 2018) with a torchvision ResNet-50 encoder initialised from ImageNet "
        "(IMAGENET1K_V2). Output stride 16 is obtained by dilating the last residual stage. An atrous spatial pyramid "
        "pooling module (rates 6, 12, 18 plus image pooling, 256 channels) feeds a decoder that fuses the upsampled "
        "features with 48-channel projected low-level features from the first residual stage, followed by two 3×3 "
        "convolution blocks, a 1×1 classifier with 13 outputs and bilinear upsampling to 512×512 — 40.35 M "
        "parameters. ImageNet mean/std normalisation is part of the model, so the model input — and therefore the "
        "attack — lives in [0, 1] RGB pixel space. DeepLabV3+ is one of the eleven networks benchmarked in the "
        "Cataract-1K paper, and a conventional encoder–decoder avoids the known weakness of FGSM/PGD against "
        "SAM-family models, which would make a small measured degradation uninterpretable."))
    story.append(H2("4.3 Training"))
    story += table([
        ["Setting", "Value"],
        ["Input / batch", "512×512 (bilinear for frames, nearest for masks), batch 8"],
        ["Loss", "Cross-entropy + 0.8 × soft Dice (Cataract-1K benchmark convention); padding pixels ignored"],
        ["Optimiser", "AdamW, learning rate 1e-4 (encoder) / 5e-4 (decoder), weight decay 1e-4"],
        ["Schedule", "1-epoch linear warm-up, cosine decay per iteration; 40 epochs (as in the project plan)"],
        ["Augmentation (GPU)", "Horizontal (p = 0.5) and vertical (p = 0.25) flips; affine scale 0.75–1.25, rotation "
                               "±15°, shift ±10%; brightness/contrast 0.75–1.25, saturation 0.8–1.2; Gaussian blur "
                               "(p = 0.2)"],
        ["Precision", "bf16 autocast for training; all evaluation and all attacks in fp32"],
        ["Model selection", f"Best validation headline mIoU: epoch {int(best.epoch)} of {len(h)} "
                            f"(validation mIoU {pc(best.val_mIoU)}%)"],
        ["Training time", f"{h.epoch_seconds.sum() / 60:.1f} min on an NVIDIA RTX 3060 Laptop GPU (6 GB)"],
        ["Seed", "42 (Python, NumPy, PyTorch); cuDNN benchmark mode on"],
    ], [3.6 * cm, TEXT_W - 3.6 * cm], caption="Baseline training configuration.", bold_first_col=True)
    story.extend(figure(os.path.join(FIG, "fig0_training_curves.png"),
                        "Training and validation loss (left) and validation mIoU per class group (right). The test "
                        "fold was never used for training or model selection.", max_h=7 * cm))
    story.append(H2("4.4 Threat model"))
    story += bullets([
        "<b>White-box:</b> the attacker knows the architecture and weights and can compute input gradients.",
        "<b>Untargeted:</b> the goal is to make as many pixels as possible wrong; the attacker uses the ground-truth "
        "mask as the reference label (the standard, worst-case convention).",
        "<b>L∞-bounded in pixel space:</b> the perturbation δ satisfies ‖δ‖<sub>∞</sub> ≤ ε and x + δ ∈ [0, 1], "
        "applied before ImageNet normalisation. ε is quoted in 8-bit units: ε = 1/255 means at most one "
        "intensity level per pixel and channel.",
        "<b>Digital:</b> perturbations are added to the decoded frame; physical, patch and temporal attacks are out "
        "of scope.",
    ])
    story.append(H2("4.5 Attacks"))
    story.append(P("Let L(x) = Σ<sub>pixels</sub> CE(f(x), y) be the summed pixel-wise cross-entropy and Π the "
                   "projection onto the ε-ball intersected with [0, 1]. Summing instead of averaging does not change "
                   "the sign of the gradient but keeps per-pixel gradients away from floating-point underflow."))
    story += table([
        ["Attack", "Update", "Settings"],
        ["FGSM (Goodfellow et al., 2015)", "x<sub>adv</sub> = Π(x + ε · sign(∇<sub>x</sub>L(x)))", "one step"],
        ["PGD-T (Madry et al., 2018)", "x<sup>0</sup> = Π(x + U(−ε, ε)); x<sup>t+1</sup> = Π(x<sup>t</sup> + α · "
                                       "sign(∇L(x<sup>t</sup>)))", "T = 20 (ablation 5–40), α = ε/4, random start"],
        ["SegPGD-T (Gu et al., 2022)", "same, with L<sub>t</sub> = (1 − λ<sub>t</sub>) Σ<sub>correct</sub> CE + "
                                       "λ<sub>t</sub> Σ<sub>wrong</sub> CE, λ<sub>t</sub> = (t − 1)/(2T)",
         "T = 20 (ablation 5–40), α = ε/4, random start"],
        ["Random noise (control)", "x<sub>adv</sub> = Π(x + ε · s), s ∈ {−1, +1} uniformly at random",
         "no gradient information"],
    ], [4.0 * cm, 8.2 * cm, TEXT_W - 12.2 * cm],
        caption="Attack definitions. SegPGD first pushes still-correct pixels over the decision boundary and "
                "progressively includes already-wrong pixels, so it does not waste gradient on pixels that are "
                "already misclassified, which plain pixel-averaged PGD does.")
    story.append(P("Every attacked batch is checked at run time: the maximum measured ‖δ‖<sub>∞</sub> must not "
                   "exceed ε and every pixel must stay in [0, 1]. In all 40 attacked runs the measured maximum "
                   "equals ε exactly (Appendix D)."))
    story.append(H2("4.6 Experiment grid"))
    story += table([
        ["Group", "Runs", "Budgets ε (/255)", "Purpose"],
        ["Clean", "1", "—", "Reference performance"],
        ["Main grid (specification)", "12", "1, 2, 4, 8 × {FGSM, PGD-20, SegPGD-20}", "Core measurement"],
        ["Noise control", "4", "1, 2, 4, 8", "Is the damage adversarial or just noise?"],
        ["Step ablation (specification)", "6", "4 (T = 5, 10, 40; T = 20 reused)", "Effect of iterations"],
        ["Step ablation (supplementary)", "6", "1 (T = 5, 10, 40)", "Same, before saturation"],
        ["Sub-8-bit budgets (supplementary)", "6", "¼, ½ × {FGSM, PGD-20, SegPGD-20}", "Where does the collapse start?"],
        ["8-bit quantised (supplementary)", "6", "1, 2 × {FGSM, PGD-20, SegPGD-20}", "Does it survive saving as PNG?"],
        ["<b>Total</b>", "<b>41</b>", "", "561 test frames each"],
    ], [5.0 * cm, 1.3 * cm, 5.6 * cm, TEXT_W - 11.9 * cm], caption="The 41 evaluation runs.", numeric_cols=[1])
    story.append(H2("4.7 Metrics and statistics"))
    story += bullets([
        "<b>Per-class IoU and Dice</b> from pixel counts pooled over all test frames: IoU = TP/(TP+FP+FN), "
        "Dice = 2TP/(2TP+FP+FN).",
        "<b>Headline mIoU / mean Dice:</b> mean over the 10 non-rare foreground classes present in the test ground "
        "truth. Rare classes (Slit/Incision Knife, Katena Forceps) are reported separately (spec §25). Because ground "
        "truth does not change under attack, clean and attacked runs average over exactly the same classes.",
        "<b>Anatomy / instrument mIoU:</b> the same, restricted to the 3 anatomy or the 7 non-rare instrument classes.",
        "<b>Degradation:</b> ΔmIoU = mIoU<sub>clean</sub> − mIoU<sub>adv</sub>; relative fall = ΔmIoU / "
        "mIoU<sub>clean</sub> × 100; robustness ratio = mIoU<sub>adv</sub> / mIoU<sub>clean</sub>.",
        "<b>Correct→wrong:</b> share of pixels the clean model classified correctly that the attack made wrong.",
        "<b>PSNR</b> of the adversarial frame relative to the clean frame (perceptibility).",
        "<b>Confidence intervals:</b> paired frame bootstrap, 2,000 resamples, the same resampled frames for clean "
        "and attacked runs, so CIs on differences are paired.",
        "<b>Consistency:</b> per-frame one-sided sign test on robustness ratios, and the number of the 6 test videos "
        "in which the effect has the same direction (the conservative evidence, as frames within a video are "
        "correlated).",
        "<b>Object size:</b> Spearman ρ across classes between mean ground-truth area per frame and relative fall.",
    ])
    story.append(H2("4.8 Corrections made to the pre-existing code"))
    story += table([
        ["Issue found during the audit", "Consequence if left", "Fix"],
        ["test_fold=1 indexed a 0-based dict, so the test set was the paper’s Fold 2 (459 frames)",
         "Not comparable to the benchmark; spec violated", "Folds renumbered 1–5; sizes asserted against the paper"],
        ["model_builder.py built torchvision DeepLabV3, not DeepLabV3+", "Specified architecture not implemented",
         "DeepLabV3+ built from ResNet-50 + ASPP + low-level decoder"],
        ["ImageNet normalisation applied in the dataset", "ε would be defined in normalised space",
         "Normalisation moved inside the model"],
        ["Dice loss clamped labels, mapping the ignore label 255 to class 12", "Padding trained as Lens Injector",
         "Ignore pixels masked from both loss terms (unit-tested)"],
        ["Validation used a whole fold (6 videos)", "Spec requires 20 train / 4 val videos",
         "One validation video per non-test fold"],
        ["mIoU over every class with any prediction, incl. background and absent classes",
         "Class set would change under attack", "Class set fixed from test ground truth; rare classes separate"],
        ["Whole dataset held in RAM", "Swapping on a 16 GB laptop", "Memory-mapped 512×512 cache, verified "
                                                                     "pixel-identical"],
    ], [6.4 * cm, 4.6 * cm, TEXT_W - 11.0 * cm], caption="Corrections to the code base made before the reported "
                                                         "experiments.")


def results_section(story, D):
    k, CN = D["key"], D["CLASS_NAMES"]
    T1, T2 = D["T1_clean_baseline_per_class"], D["T2_main_grid"]
    g = {r["run"]: r for r in k["main_grid"]}
    story.append(PageBreak())
    story.append(H1("5. Results"))
    # 5.1 clean
    story.append(H2("5.1 Clean baseline"))
    c = k["clean"]
    story += table([
        ["Metric", "Value"],
        ["Headline mIoU (95% CI)", f"{pc(c['mIoU'])} ({pc(c['mIoU_ci'][0])}–{pc(c['mIoU_ci'][1])})"],
        ["Headline mean Dice", pc(c["mDice"])],
        ["Anatomy mIoU / Dice", f"{pc(c['mIoU_anatomy'])} / {pc(c['mDice_anatomy'])}"],
        ["Instrument mIoU / Dice (non-rare)", f"{pc(c['mIoU_instruments'])} / {pc(c['mDice_instruments'])}"],
        ["Instrument mIoU incl. rare classes", pc(c["mIoU_instruments_all"])],
        ["mIoU over all 12 foreground classes", pc(c["mIoU_eval"])],
        ["Pixel accuracy", pc(c["pixel_accuracy"])],
    ], [8 * cm, TEXT_W - 8 * cm], caption="Clean performance on the held-out Fold 1 (561 frames, 6 patients).",
        numeric_cols=[1])
    rows = [["ID", "Class", "Group", "Test frames", "IoU %", "Dice %"]]
    for _, r in T1.iterrows():
        rows.append([str(r["id"]), esc(r["class"]) + (" <i>(rare)</i>" if r["rare"] else ""), r["group"],
                     f"{r['test_frames']}", pc(r["IoU"]), pc(r["Dice"])])
    story += table(rows, [1.0 * cm, 5.4 * cm, 2.6 * cm, 2.4 * cm, 2.4 * cm, TEXT_W - 13.8 * cm],
                   caption="Clean per-class results. Katena Forceps is largely missed even without attack "
                           "(8 test frames); see Figure 8, row 3.", numeric_cols=[0, 3, 4, 5])
    rows = [["Structure", "Paper IoU", "Ours IoU", "Paper Dice", "Ours Dice"]]
    for name, v in k["paper_comparison"].items():
        rows.append([esc(name), f"{v['paper_iou']:.2f}", f"{v['ours_iou']:.2f}", f"{v['paper_dice']:.2f}",
                     f"{v['ours_dice']:.2f}"])
    story += table(rows, [6.0 * cm] + [(TEXT_W - 6.0 * cm) / 4] * 4,
                   caption="Comparison with the published benchmark: Cataract-1K paper (Sci. Data 2024), Table 7, "
                           "DeepLabV3+ with ResNet-50, anatomy-plus-instrument task, where all instruments form one "
                           "class (ours merged the same way).", numeric_cols=[1, 2, 3, 4])
    story.append(P(
        "The baseline meets or exceeds the published DeepLabV3+ on every structure. The comparison is indicative, "
        "not like-for-like: the paper reports a 5-fold cross-validation average with a different training recipe, "
        "whereas this project evaluates the single official Fold 1. The large merged-instrument gap is therefore "
        "<i>not</i> claimed as an improvement over the benchmark; the conclusion drawn is only that the baseline is "
        "well trained, so the vulnerability measured below is not an artefact of a weak model."))
    # 5.2 main grid
    story.append(CondPageBreak(9 * cm))
    story.append(H2("5.2 Main grid: clean versus attacked"))
    order = ["clean"] + [f"{a}_eps{e}" for a in ("noise", "fgsm") for e in (1, 2, 4, 8)] + \
            [f"{a}_eps{e}_T20" for a in ("pgd", "segpgd") for e in (1, 2, 4, 8)]
    label = {"clean": "Clean", "noise": "Noise (control)", "fgsm": "FGSM", "pgd": "PGD-20", "segpgd": "SegPGD-20"}
    rows = [["Attack", "ε", "mIoU % (95% CI)", "ΔmIoU<br/>pp", "Rel.<br/>fall %", "Robust.<br/>ratio", "mDice<br/>%",
             "Anat.<br/>mIoU", "Instr.<br/>mIoU", "Correct→<br/>wrong %", "PSNR<br/>dB"]]
    for rid in order:
        r = T2[T2.run == rid].iloc[0]
        a = rid.split("_")[0]
        if rid == "clean":
            rows.append(["<b>Clean</b>", "—", f"{pc(r.mIoU)} ({pc(r.mIoU_lo)}–{pc(r.mIoU_hi)})", "—", "—", "1.000",
                         pc(r.mDice), pc(r.mIoU_anatomy), pc(r.mIoU_instruments), "—", "—"])
        else:
            rows.append([label[a], eps_s(r.eps_255), f"{pc(r.mIoU)} ({pc(r.mIoU_lo)}–{pc(r.mIoU_hi)})",
                         f"{100 * r.delta_mIoU:.2f}", f"{r.rel_fall_pct:.1f}", f"{r.robustness_ratio:.3f}",
                         pc(r.mDice), pc(r.mIoU_anatomy), pc(r.mIoU_instruments), f"{r.correct_to_wrong_pct:.1f}",
                         f"{r.psnr_db:.1f}"])
    w = [2.25, 1.05, 2.85, 1.35, 1.35, 1.35, 1.35, 1.35, 1.25, 1.6, 1.25]
    scale = TEXT_W / (sum(w) * cm)
    story += table(rows, [x * cm * scale for x in w],
                   caption="Main grid on the test fold. PGD/SegPGD: 20 steps, α = ε/4, random start. 95% CIs from a "
                           "paired frame bootstrap. Anatomy / Instr. = group mIoU %.",
                   numeric_cols=list(range(2, 11)), font_scale=0.93)
    story.extend(figure(os.path.join(FIG, "fig1_miou_vs_eps.png"),
                        "Headline mIoU versus perturbation budget for the three attacks and the noise control (shaded: "
                        "95% bootstrap CI). The grey band marks the supplementary budgets below one 8-bit step.",
                        max_w=TEXT_W * 0.86, max_h=10.5 * cm))
    story.append(H3("Reading the grid"))
    story += bullets([
        "<b>Noise versus gradient attacks.</b> At ε = 1/255 the noise control and FGSM have identical PSNR (48.2 dB, "
        f"every pixel moved by exactly ±1/255), yet noise costs {g['noise_eps1']['rel_fall_pct']:.1f}% of mIoU and "
        f"FGSM {g['fgsm_eps1']['rel_fall_pct']:.1f}%. The vulnerability is directional, not a general sensitivity "
        "to small perturbations.",
        "<b>FGSM plateaus.</b> Its damage barely changes with ε (" + " → ".join(
            f"{g[f'fgsm_eps{e}']['rel_fall_pct']:.1f}%" for e in (1, 2, 4, 8)) +
        "). A single linear step stops gaining once the loss surface curves; a larger budget does not substitute "
        "for iterations.",
        "<b>Iterative attacks saturate at the smallest specified budget.</b> PGD-20 and SegPGD-20 already remove "
        "95% of mIoU at ε = 1/255, which motivated the supplementary sub-8-bit budgets (Section 5.6).",
        "<b>PGD perturbations are less visible than FGSM’s</b> at the same ε (51.2 vs 48.2 dB at 1/255) — PGD does "
        "not push every pixel to ±ε — yet they do about three times the damage.",
    ])
    # 5.3 ablation
    story.append(CondPageBreak(9 * cm))
    story.append(H2("5.3 Attack strength and number of iterations"))
    T5 = D["T5_step_ablation"]
    T2b = D["T2b_eps_curve_and_quantisation"]
    for e, tag in ((4, "specification"), (1, "supplementary")):
        sub = T5[T5.eps_255 == e]
        rows = [["Steps", "1 (FGSM)", "5", "10", "20", "40"]]
        for att in ("PGD", "SegPGD"):
            vals = []
            for t in (1, 5, 10, 20, 40):
                if t == 1:
                    s = sub[(sub.attack == "FGSM")]
                    vals.append(pc(s.mIoU.iloc[0]) if att == "PGD" and len(s) else "—")
                else:
                    s = sub[(sub.attack == att) & (sub.steps == t)]
                    vals.append(pc(s.mIoU.iloc[0]) if len(s) else "—")
            rows.append([f"{att} mIoU %"] + vals)
        for att, key in (("PGD", "pgd"), ("SegPGD", "segpgd")):
            vals = []
            for t in (1, 5, 10, 20, 40):
                if t == 1:
                    rid = f"fgsm_eps{e}"
                else:
                    rid = f"{key}_eps{e}_T{t}"
                rr = next((r for r in D["runs"] if r["id"] == rid), None)
                if t == 1 and att == "SegPGD":
                    vals.append("—")
                    continue
                z = np.load(os.path.join(RES, "raw", rid + ".npz"))
                vals.append(f"{100 * z['flipped_correct'].sum() / z['clean_correct'].sum():.1f}")
            rows.append([f"{att} correct→wrong %"] + vals)
        story += table(rows, [4.6 * cm] + [(TEXT_W - 4.6 * cm) / 5] * 5,
                       caption=f"Step-count ablation at ε = {e}/255 ({tag}); α = ε/4 for every step count.",
                       numeric_cols=[1, 2, 3, 4, 5], bold_first_col=True)
    story.extend(figure(os.path.join(FIG, "fig5_step_ablation.png"),
                        "Headline mIoU versus number of attack iterations at ε = 4/255 (left, specification) and "
                        "1/255 (right, supplementary).", max_h=7.5 * cm))
    story += bullets([
        "Five PGD steps at one intensity level already do more damage (−72% relative) than FGSM at eight levels "
        "(−31%): iterations matter far more than budget.",
        "Damage keeps increasing up to 40 steps, so the 20-step results are conservative. Measured mIoU under attack "
        "is an upper bound on the model’s robustness, never proof of it (Croce et al., SEA).",
        "SegPGD and PGD reach statistically indistinguishable mIoU at every step count, but SegPGD consistently "
        "converts 3–7 percentage points more of the correctly classified pixels — exactly what its reweighted loss "
        "is designed to do. On this model plain PGD is already strong enough to collapse mIoU, so the "
        "segmentation-aware reweighting shows in the pixel-flip rate rather than in mIoU.",
    ])
    # 5.4 anatomy vs instruments
    story.append(CondPageBreak(10 * cm))
    story.append(H2("5.4 Anatomy versus instruments"))
    story.append(P("This comparison — whether delicate metallic instruments degrade faster than the surrounding "
                   "biological structures within the same frame — is the project’s main novelty claim (RQ3)."))
    T4, T4c = D["T4_anatomy_vs_instruments"], D["T4c_present_frames_only"]
    sel = [("Noise", 8), ("FGSM", 0.25), ("FGSM", 1), ("FGSM", 8), ("PGD-20", 0.25), ("PGD-20", 0.5),
           ("PGD-20", 1), ("PGD-20", 8), ("SegPGD-20", 0.25), ("SegPGD-20", 0.5), ("SegPGD-20", 1), ("SegPGD-20", 8)]
    rows = [["Attack", "ε", "Anatomy rel. fall %", "Instrument rel. fall %", "Difference pp (95% CI)",
             "Present-frames-only diff. pp", "Videos (of 6)"]]
    for a, e in sel:
        r = T4[(T4.attack == a) & (T4.eps_255 == e)].iloc[0]
        rc = T4c[(T4c.attack == a) & (T4c.eps_255 == e)].iloc[0]
        rows.append([a, eps_s(e), f"{r.anatomy_rel_fall_pct:.1f}", f"{r.instrument_rel_fall_pct:.1f}",
                     f"{r.diff_pp:+.1f} ({r.diff_lo:+.1f} to {r.diff_hi:+.1f})", f"{rc.diff_present_only_pp:+.1f}",
                     f"{int(r.videos_inst_worse)}/{int(r.videos)}"])
    story += table(rows, [2.2 * cm, 1.4 * cm, 2.3 * cm, 2.4 * cm, 3.6 * cm, 2.6 * cm, TEXT_W - 14.5 * cm],
                   caption="Relative fall of the anatomy and instrument group mIoU, their difference (instrument − "
                           "anatomy) with a paired bootstrap CI, the same difference when each class is scored only "
                           "on frames where it is present (removes hallucinated pixels), and the number of test "
                           "videos where instruments fall further. ¼ and ½ /255 are supplementary budgets. All "
                           "budgets: Appendix B.", numeric_cols=[1, 2, 3, 4, 5, 6])
    story += bullets([
        "<b>Consistent across attack types.</b> Wherever the attack is not saturated (noise at 8/255, FGSM at every "
        "budget, PGD/SegPGD up to 2/255), instruments lose a larger share of their clean IoU than anatomy: the CIs "
        "exclude zero, the per-frame sign tests give p ≤ 0.025 (p &lt; 0.001 in all but one setting), and all six "
        "test videos agree.",
        "<b>Mostly genuine segmentation failure.</b> Excluding hallucinated pixels shrinks the gap at ¼/255 "
        "(+20.7 → +14.6 pp for PGD-20) — hallucination contributes — but a large gap remains: the attacked model "
        "segments the <i>real</i> instruments worse than the real anatomy.",
        "<b>At saturation the distinction disappears.</b> At ε ≥ 4/255 both groups are near zero and the gap is "
        "≤ 1.6 pp.",
        "<b>Relative versus absolute.</b> Under strong iterative attacks anatomy loses more <i>absolute</i> IoU "
        "(≈ 86 vs ≈ 75 points) simply because it starts higher; the hypothesis concerns how fast each group "
        "degrades, so relative fall is the primary measure. Both are tabulated in Appendix B.",
    ])
    story += to_landscape()
    story.extend(figure(os.path.join(FIG, "fig2_anatomy_vs_instruments.png"),
                        "Share of clean group mIoU retained under each attack. The instrument line (orange) lies below "
                        "the anatomy line (blue) under every attack; the gap is largest at small budgets. Grey band: "
                        "supplementary budgets below one 8-bit step.", max_w=LAND_W, max_h=7.5 * cm))
    story.extend(figure(os.path.join(FIG, "fig4_per_class_relative_fall.png"),
                        "Per-class relative IoU fall in three unsaturated regimes (headline classes; blue = anatomy, "
                        "orange = instrument). The anatomy classes are among the least affected in every regime; "
                        "Capsulorhexis Forceps is the exception among instruments.", max_w=LAND_W, max_h=8.6 * cm))
    story += to_portrait()
    # 5.5 per class
    story.append(H2("5.5 Per-class vulnerability and the “thin metallic tools” hypothesis"))
    T3b = D["T3b_per_class_relative_fall_pct"]
    rows = [["Class", "Group", "Test frames", "Noise 8/255", "FGSM 1/255", "PGD-20 ¼/255", "SegPGD-20 ¼/255",
             "SegPGD-20 1/255"]]
    for _, r in T3b.iterrows():
        rows.append([esc(r["class"]) + (" *" if r["rare"] else ""), r["group"], str(r["test_frames"])] +
                    [num(r[col], 1) for col in ("noise_eps8", "fgsm_eps1", "pgd_eps0.25", "segpgd_eps0.25",
                                                "segpgd_eps1")])
    story += table(rows, [4.4 * cm, 1.9 * cm, 1.5 * cm] + [(TEXT_W - 7.8 * cm) / 5] * 5,
                   caption="Per-class relative IoU fall (%) in the unsaturated regimes. * rare class (excluded from "
                           "headline means; Katena Forceps is near 0% IoU even when clean, so its relative fall is not "
                           "meaningful). Absolute IoU for every class and run: Appendix A.",
                   numeric_cols=[2, 3, 4, 5, 6, 7])
    sp = k["size_vs_fall_spearman"]
    sel = ["noise_eps8", "fgsm_eps1", "fgsm_eps8", "pgd_eps0.25_T20", "pgd_eps1_T20", "segpgd_eps0.25_T20"]
    rho = ", ".join(f"{s.split('_')[0].replace('segpgd', 'SegPGD').replace('pgd', 'PGD').replace('fgsm', 'FGSM').replace('noise', 'noise')} "
                    f"{eps_s(float(s.split('_eps')[1].split('_')[0]))}: ρ = {sp[s][0]:+.2f}" for s in sel if s in sp)
    story += bullets([
        "<b>The three anatomy classes are the least affected in every regime.</b> Under FGSM at 1/255 the pupil, "
        "lens and cornea lose 13–27% while instruments lose 28–45%; under SegPGD-20 at ¼/255 anatomy loses 56–69% "
        "and instruments 64–99% (Lens Injector −99%, Irrigation-Aspiration −87%).",
        "<b>One instrument behaves like anatomy.</b> Capsulorhexis Forceps is consistently among the most robust "
        "classes despite being thin and metallic.",
        f"<b>Object size.</b> Across the 10 headline classes the Spearman correlation between mean object area and "
        f"relative fall is negative in every non-trivial run ({rho}); smaller structures tend to fall further. "
        "With 10 classes none of these correlations is statistically significant (smallest p = 0.09; Appendix C).",
    ])
    story += note_box(
        "<b>Verdict on the hypothesis “thin, specular metallic tools degrade faster under small noise caps.”</b> "
        "<i>Supported at the group level:</i> instruments degrade faster than anatomy under every attack type at "
        "small budgets, with CIs excluding zero, unanimous across videos, and not explained by hallucination. "
        "<i>Size is a plausible but unproven explanation:</i> the size–vulnerability correlation has a consistent "
        "negative sign but is not significant, and Capsulorhexis Forceps — a thin tool that stays robust — shows that "
        "thinness alone does not determine vulnerability.")
    story += to_landscape()
    story.extend(figure(os.path.join(FIG, "fig6_object_size_vs_fall.png"),
                        "Mean ground-truth area per frame versus relative IoU fall, per headline class, in three "
                        "unsaturated regimes (Spearman ρ across classes in the panel titles).",
                        max_w=LAND_W, max_h=8 * cm))
    story.extend(figure(os.path.join(FIG, "fig3_per_class_iou_heatmap.png"),
                        "Per-class IoU (%) on the test fold, clean and under every attack and budget (¼ and ½ /255 are "
                        "supplementary; * rare class).", max_w=LAND_W, max_h=9.2 * cm))
    story += to_portrait()
    # 5.6 supplementary
    story.append(H2("5.6 Below one 8-bit step, and 8-bit quantisation"))
    story.append(P("The specification’s smallest budget (1/255) already saturates PGD and SegPGD. Two questions "
                   "follow: where does the collapse begin, and does the attack still work once the adversarial frame "
                   "is stored as an ordinary 8-bit image?"))
    rows = [["Attack", "ε", "mIoU % (95% CI)", "Rel. fall %", "Anatomy mIoU %", "Instrument mIoU %",
             "Correct→wrong %"]]
    for a, lab in (("fgsm", "FGSM"), ("pgd", "PGD-20"), ("segpgd", "SegPGD-20")):
        for e in (0.25, 0.5, 1.0):
            r = T2b[(T2b.attack == a) & (T2b.eps_255 == e) & (~T2b.quantised_8bit)].iloc[0]
            rows.append([lab, eps_s(e), f"{pc(r.mIoU)} ({pc(r.mIoU_lo)}–{pc(r.mIoU_hi)})", f"{r.rel_fall_pct:.1f}",
                         pc(r.mIoU_anatomy), pc(r.mIoU_instruments), f"{r.correct_to_wrong_pct:.1f}"])
    story += table(rows, [2.4 * cm, 1.4 * cm, 3.6 * cm, 2.0 * cm, 2.6 * cm, 2.8 * cm, TEXT_W - 14.8 * cm],
                   caption="Budgets below one 8-bit step (¼, ½ /255; supplementary) with 1/255 for reference. These "
                           "perturbations exist only for an attacker who can modify decoded frames in memory.",
                   numeric_cols=[1, 2, 3, 4, 5, 6])
    rows = [["Attack", "ε", "mIoU % (float)", "mIoU % (8-bit)", "Change pp", "max ‖δ‖∞ after rounding"]]
    for a, lab in (("fgsm", "FGSM"), ("pgd", "PGD-20"), ("segpgd", "SegPGD-20")):
        for e in (1.0, 2.0):
            f = T2b[(T2b.attack == a) & (T2b.eps_255 == e) & (~T2b.quantised_8bit)].iloc[0]
            q = T2b[(T2b.attack == a) & (T2b.eps_255 == e) & (T2b.quantised_8bit)].iloc[0]
            rows.append([lab, eps_s(e), pc(f.mIoU), pc(q.mIoU), f"{100 * (q.mIoU - f.mIoU):+.2f}",
                         f"{q.linf_max_255:.2f}/255"])
    story += table(rows, [2.6 * cm, 1.6 * cm, 3.0 * cm, 3.0 * cm, 2.4 * cm, TEXT_W - 12.6 * cm],
                   caption="8-bit quantisation check: the adversarial frame is rounded to the 8-bit grid (as when "
                           "saved to PNG) before inference. Clean frames are already on that grid, so the budget still "
                           "holds exactly.", numeric_cols=[1, 2, 3, 4, 5])
    story += bullets([
        "A quarter of one intensity level removes 73% of mIoU with 20 iterations; the ε curve (Figure 2) is steepest "
        "<i>below</i> the specified range, so the model has no meaningful robustness margin at any budget an 8-bit "
        "attacker can use.",
        "Rounding changes mIoU by at most 0.26 points: the headline result holds for adversarial frames stored as "
        "standard 8-bit images and is not a floating-point artefact.",
    ])
    # 5.7 composition
    story.append(CondPageBreak(8 * cm))
    story.append(H2("5.7 What the model predicts under attack"))
    T7 = D["T7"]
    rows = [["Class"] + [esc(c) for c in T7.columns]]
    for name, row in T7.iterrows():
        rows.append([esc(name)] + [f"{v:.1f}" for v in row])
    story += table(rows, [4.6 * cm] + [(TEXT_W - 4.6 * cm) / len(T7.columns)] * len(T7.columns),
                   caption="Share of all test pixels (%) assigned to each class: ground truth and predictions.",
                   numeric_cols=list(range(1, len(T7.columns) + 1)), font_scale=0.95)
    story.append(P(
        "FGSM leaves the class proportions unchanged while removing about 30% of the mIoU: it <i>mislocalises and "
        "confuses</i> structures. PGD and SegPGD make the model <i>hallucinate</i>: almost half of the frame becomes "
        "“pupil” and about 6% becomes “lens injector”, an instrument present in only 14 test frames. For any "
        "downstream use — tool tracking, phase recognition, skill assessment — confident false structures are a "
        "worse failure than an empty mask."))
    # 5.8 qualitative
    story += to_landscape()
    story.append(H2("5.8 Qualitative examples"))
    story.extend(figure(os.path.join(FIG, "fig7_qualitative_segpgd_eps4.png"),
                        "SegPGD-20 at ε = 4/255 on four test frames. Columns: clean frame, adversarial frame, "
                        "perturbation (rescaled from ±ε to the full range so it is visible), ground truth, clean "
                        "prediction, prediction under attack. The adversarial frames are visually indistinguishable "
                        "from the clean ones, yet the predictions become large hallucinated regions. Row 3: the Katena "
                        "forceps (green, top left of the ground truth) is missed even by the clean model.",
                        max_w=LAND_W, max_h=LAND_H - 2.2 * cm, photo=True))
    story.append(PageBreak())
    story.extend(figure(os.path.join(FIG, "fig8_qualitative_attack_comparison_eps4.png"),
                        "The same four frames under each attack at ε = 4/255: random noise and FGSM preserve most "
                        "structure, PGD-20 and SegPGD-20 destroy it.", max_w=LAND_W, max_h=LAND_H - 2.2 * cm,
                        photo=True))
    story.append(PageBreak())
    story.extend(figure(os.path.join(FIG, "fig9_qualitative_budget_progression.png"),
                        "One frame under SegPGD-20 at increasing budgets (top: attacked frames; bottom: predictions). "
                        "The input barely changes while the output collapses already at ¼/255.",
                        max_w=LAND_W, max_h=LAND_H - 2.2 * cm, photo=True))
    story += to_portrait()
    # 5.9 replication
    story.append(H2("5.9 Replication on the first 908-frame run"))
    story.append(P("The full pipeline was first run on a partial download of 908 frames (524 train / 259 test, "
                   "80 epochs); its complete write-up is archived in <font name='DejaVuMono'>results_908frames/</font>. "
                   "Comparing the two independent runs shows which conclusions are robust."))

    def row_of(table_, run):
        return table_[table_.run == run].iloc[0]

    a2, b2 = D["908_T2_main_grid"], D["T2_main_grid"]
    a2b, b2b = D["908_T2b_eps_curve_and_quantisation"], D["T2b_eps_curve_and_quantisation"]
    a4, b4 = D["908_T4_anatomy_vs_instruments"], D["T4_anatomy_vs_instruments"]

    def quant_max(t):
        q = t[t.quantised_8bit & t.attack.isin(["pgd", "segpgd"])]
        m = 0.0
        for _, r in q.iterrows():
            f = t[(t.attack == r.attack) & (t.eps_255 == r.eps_255) & (~t.quantised_8bit)].iloc[0]
            m = max(m, abs(r.mIoU - f.mIoU) * 100)
        return m

    def gap(t, a, e):
        r = t[(t.attack == a) & (t.eps_255 == e)].iloc[0]
        p = r.sign_test_p
        return (f"{r.diff_pp:+.1f} pp, {int(r.videos_inst_worse)}/{int(r.videos)} videos, "
                f"p {'&lt; 0.001' if p < 0.001 else '= ' + format(p, '.2f')}")

    def sizerho(key, rid):
        v = key["size_vs_fall_spearman"].get(rid)
        return f"{v[0]:+.2f}" if v else "—"

    rows = [["Measure", "908-frame run", "Full data", "Same conclusion?"],
            ["Clean headline mIoU", pc(row_of(a2, "clean").mIoU), pc(row_of(b2, "clean").mIoU), "✔ competent baseline"],
            ["Noise 8/255, relative fall", f"{row_of(a2, 'noise_eps8').rel_fall_pct:.1f}%",
             f"{row_of(b2, 'noise_eps8').rel_fall_pct:.1f}%", "— more data halved noise sensitivity"],
            ["FGSM 1/255, relative fall", f"{row_of(a2, 'fgsm_eps1').rel_fall_pct:.1f}%",
             f"{row_of(b2, 'fgsm_eps1').rel_fall_pct:.1f}%", "✔ about one third, flat in ε"],
            ["PGD-20 1/255, mIoU (rel. fall)",
             f"{pc(row_of(a2, 'pgd_eps1_T20').mIoU)} ({row_of(a2, 'pgd_eps1_T20').rel_fall_pct:.1f}%)",
             f"{pc(row_of(b2, 'pgd_eps1_T20').mIoU)} ({row_of(b2, 'pgd_eps1_T20').rel_fall_pct:.1f}%)",
             "✔ collapse at one intensity level"],
            ["SegPGD-20 1/255, mIoU (rel. fall)",
             f"{pc(row_of(a2, 'segpgd_eps1_T20').mIoU)} ({row_of(a2, 'segpgd_eps1_T20').rel_fall_pct:.1f}%)",
             f"{pc(row_of(b2, 'segpgd_eps1_T20').mIoU)} ({row_of(b2, 'segpgd_eps1_T20').rel_fall_pct:.1f}%)", "✔"],
            ["8-bit rounding, max change", f"{quant_max(a2b):.2f} pp", f"{quant_max(b2b):.2f} pp",
             "✔ survives quantisation"],
            ["Pupil share under SegPGD-20 4/255", f"{D['T7_908'].loc['Pupil', 'SegPGD-20 4/255']:.1f}%",
             f"{D['T7'].loc['Pupil', 'SegPGD-20 4/255']:.1f}%", "✔ hallucination"],
            ["Instrument − anatomy gap, FGSM 1/255", gap(a4, "FGSM", 1), gap(b4, "FGSM", 1), "✔"],
            ["Instrument − anatomy gap, PGD-20 ¼/255", gap(a4, "PGD-20", 0.25), gap(b4, "PGD-20", 0.25),
             "✘ inconclusive → clear"],
            ["Size–vulnerability ρ, PGD-20 / SegPGD-20 ¼/255",
             f"{sizerho(D['key908'], 'pgd_eps0.25_T20')} / {sizerho(D['key908'], 'segpgd_eps0.25_T20')}",
             f"{sizerho(k, 'pgd_eps0.25_T20')} / {sizerho(k, 'segpgd_eps0.25_T20')}", "✘ sign flipped"]]
    story += table(rows, [5.3 * cm, 4.2 * cm, 4.2 * cm, TEXT_W - 13.7 * cm],
                   caption="Replication: the 908-frame run (2.7× less training data, 2.2× fewer test frames) versus "
                           "the full-data run.", bold_first_col=False)
    story.append(P(
        "Every headline vulnerability result replicates. Two secondary findings changed, and in both cases the "
        "full-data result — larger training and test sets, all classes evaluable — supersedes the 908-frame one: "
        "the anatomy/instrument asymmetry under iterative attacks, inconclusive on 908 frames, is clear on the full "
        "data; and the apparent reversal of the size effect under iterative attacks did not replicate. More training "
        "data made the model more robust to random noise but not to adversarial perturbation."))


def answers_section(story, D):
    k = D["key"]
    g = {r["run"]: r for r in k["main_grid"]}
    story.append(PageBreak())
    story.append(H1("6. Answers to the Research Questions"))
    story += note_box(
        "<b>Primary — How do adversarial attacks affect the performance and robustness of the model?</b><br/>"
        "Catastrophically, and at very small budgets. A white-box iterative attack with ε = 1/255 (one intensity "
        f"level, ≈ 51 dB PSNR) removes 95% of the headline mIoU ({pc(k['clean']['mIoU'])}% → "
        f"{pc(g['segpgd_eps1_T20']['mIoU'])}–{pc(g['pgd_eps1_T20']['mIoU'])}%), and the attack survives being "
        "saved as an 8-bit image. At 8/255 the model is at 0.3–0.4%. A single-step attack removes about 30%; random "
        f"noise of the same budget at most {g['noise_eps8']['rel_fall_pct']:.1f}%.")
    story += table([
        ["#", "Answer", "Evidence"],
        ["RQ1", "ΔmIoU 23–24 points under FGSM and 75–79 points under PGD-20 / SegPGD-20; mean Dice falls from 88.3% "
                "to 70–71% (FGSM) and 0.7–8% (iterative).", "§5.2"],
        ["RQ2", "Iterations matter far more than budget: SegPGD ≈ PGD ≫ FGSM ≫ noise. More steps keep helping up to "
                "40. FGSM barely improves with ε. SegPGD flips 3–7 points more correct pixels than PGD.", "§5.2, §5.3"],
        ["RQ3", "Yes. Instruments lose a larger share of IoU than anatomy at every unsaturated budget of every attack "
                "(+3.6 to +20.7 points; all six videos), mostly from genuine segmentation failure. Lens Injector and "
                "Irrigation-Aspiration are the most fragile classes; the anatomy classes and Capsulorhexis Forceps "
                "the most robust. Smaller structures tend to fall further (consistent sign, not significant).",
         "§5.4, §5.5"],
        ["RQ4", f"{pc(k['clean']['mIoU'])}% clean → 55.1–56.3% (FGSM) and 0.3–4.4% (iterative) headline mIoU; full "
                "per-class tables in Appendix A.", "§5.2, App. A"],
        ["RQ5", "Strongly non-linear. Iterative attacks remove 73% of mIoU at ¼/255 and 95% at 1/255, then flatten; "
                "FGSM reaches its ~30% plateau by 1/255; noise grows slowly (0.2% → 5.2% from 1 to 8/255). Results "
                "are unchanged after 8-bit rounding.", "§5.2, §5.6"],
    ], [1.3 * cm, TEXT_W - 4.0 * cm, 2.7 * cm], caption="Answers to the supporting research questions.",
        bold_first_col=True)


def verification_section(story, D):
    story.append(PageBreak())
    story.append(H1("7. Verification Against the Problem Statement and Plan"))
    story.append(H2("7.1 Problem statement"))
    story.append(P("<b>Status: answered.</b> The vulnerability of a cataract surgical segmentation model to "
                   "adversarial perturbation has been quantified on the complete official held-out fold, with "
                   "confidence intervals, a no-gradient control, an 8-bit storage check and an independent "
                   "replication (Sections 5 and 6)."))
    story.append(H2("7.2 Specific objectives (specification §7)"))
    story += table([
        ["#", "Objective", "Status", "Evidence"],
        ["1", "Study existing research on cataract surgical AI and adversarial attacks", "Done (before this phase)",
         "Literature pool, gap matrix"],
        ["2", "Select an appropriate dataset and AI model", "Done", "Cataract-1K (2,256 frames); DeepLabV3+ / ResNet-50"],
        ["3", "Establish baseline performance on clean inputs", "Done", "79.35% mIoU, 88.27% mDice (§5.1)"],
        ["4", "Implement adversarial attack methods", "Done", "FGSM, PGD, SegPGD + noise control (§4.5)"],
        ["5", "Generate adversarial examples", "Done", "40 attacked runs × 561 frames"],
        ["6", "Evaluate performance on adversarial inputs", "Done", "Same model and metrics, fp32 (§5.2)"],
        ["7", "Quantify degradation relative to the clean baseline", "Done",
         "ΔmIoU, relative fall, robustness ratio, paired CIs"],
        ["8", "Analyse attack strength versus degradation", "Done", "6 budgets, two step ablations (§5.2–5.3, §5.6)"],
        ["9", "Identify vulnerability patterns", "Done", "Anatomy vs instruments, per class, size, hallucination"],
        ["10", "Document findings and limitations", "Done", "This document"],
    ], [0.8 * cm, 7.2 * cm, 3.0 * cm, TEXT_W - 11.0 * cm], caption="Objectives of the Version 1 specification.",
        numeric_cols=[0])
    story.append(H2("7.3 Locked decisions (specification §25) and plan milestones"))
    story += table([
        ["Requirement", "Status"],
        ["DeepLabV3+, ImageNet ResNet-50, OS16, 13 channels, 512×512", "✔ (earlier code used DeepLabV3; corrected)"],
        ["Official patient-disjoint folds, Fold 1 test, 20 train / 4 validation videos",
         "✔ (earlier code tested on Fold 2; corrected)"],
        ["White-box, untargeted, L∞, perturbation in [0,1] RGB before normalisation, clipped",
         "✔ checked on every batch"],
        ["FGSM; PGD-20 (α = ε/4, random start); SegPGD-20 as the primary attack", "✔"],
        ["ε ∈ {1, 2, 4, 8}/255 → 12 attacked runs + 1 clean", "✔ (+ 4 noise controls)"],
        ["Step-count ablation {5, 10, 20, 40} at ε = 4/255 for PGD and SegPGD", "✔ (+ the same at 1/255)"],
        ["mIoU and mean Dice headline; per-class IoU always reported; rare classes separate", "✔"],
        ["Absolute and relative degradation, robustness ratio, mIoU-vs-ε curves", "✔"],
        ["Anatomy versus instrument comparison", "✔ with CIs, per-video consistency, hallucination control"],
        ["Clean baseline within the published benchmark range (plan, Phase 2)", "✔ meets or exceeds every value"],
        ["40 training epochs (plan, Phase 2)", "✔"],
        ["Unit-tested perturbation clipping (plan, Phase 3)", "✔ self-test + runtime assertion"],
        ["Figure panels: clean, perturbed, difference map, clean and attacked prediction (plan, Phase 4)", "✔ §5.8"],
        ["Documentation, requirements, fixed seeds, README (plan, Phase 5)", "✔"],
        ["No defences or adversarial training in Version 1 (specification §28)", "✔ none added"],
        ["Thesis-format final report and presentation slides (plan, Phases 5–6)", "Open — this document supplies "
                                                                                  "the material"],
    ], [TEXT_W - 6.2 * cm, 6.2 * cm], caption="Compliance with the locked Version 1 decisions and the "
                                              "implementation plan.")


def limitations_section(story, D):
    story.append(PageBreak())
    story.append(H1("8. Limitations and Threats to Validity"))
    story += bullets([
        "<b>One fold, one seed, one architecture.</b> Fold 1 only, as the specification requires; no cross-fold "
        "variance and no second architecture (the optional UNet++ arm was not run). The 908-frame replication "
        "(§5.9) partly offsets this.",
        "<b>Correlated frames.</b> Bootstrap CIs and per-frame sign tests treat frames as independent; frames from "
        "the same video are not. The six-video consistency counts are the conservative evidence.",
        "<b>Rare classes.</b> Slit/Incision Knife (12 test frames) and Katena Forceps (8) are too sparse for firm "
        "per-class conclusions; Katena is barely segmented even on clean frames (4.1% IoU).",
        "<b>Attack strength is a lower bound.</b> Damage was still rising from 20 to 40 iterations; measured "
        "mIoU under attack is an upper bound on robustness, never proof of it.",
        "<b>Threat model.</b> White-box, digital, untargeted, with ground-truth labels available to the attacker — "
        "the worst case by design, not a realistic operating-room attack. Black-box, transfer, physical and temporal "
        "attacks are Version 2.",
        "<b>Sub-8-bit budgets</b> (¼, ½ /255) cannot be stored in an 8-bit image and apply only to in-memory "
        "manipulation of decoded frames; the 8-bit-rounded runs cover the stored-image case.",
        "<b>Benchmark comparison</b> with the Cataract-1K paper is not like-for-like (single fold versus 5-fold mean, "
        "different training recipe).",
        "<b>Timing figures are wall-clock</b> and were affected by GPU sharing with other applications and, for one "
        "run, by a system sleep (marked ‡ in Appendix D). No result depends on timing.",
        "<b>No clinical claims.</b> These are laboratory measurements on recorded frames; they say nothing directly "
        "about patient safety.",
    ])
    story.append(PageBreak())
    story.append(H1("9. Deviations from the Plan"))
    story += table([
        ["Deviation", "Reason"],
        ["Random ±ε noise control added", "Separates “any small perturbation” from “adversarial direction”; without "
                                          "it the degradation is uninterpretable"],
        ["Sub-8-bit budgets (¼, ½ /255) and 8-bit-rounded runs added", "Iterative attacks saturated at the smallest "
                                                                       "specified budget, leaving no unsaturated "
                                                                       "iterative points"],
        ["Step ablation also at 1/255", "To see the effect of iterations before saturation"],
        ["Prediction-composition table and present-frames-only control", "Explain how the model fails "
                                                                          "(hallucination versus mislocalisation) and "
                                                                          "test whether the instrument gap is real"],
        ["First run on a 908-frame partial download, kept as replication", "The first download was truncated by "
                                                                           "Synapse’s 1 GB package limit; the study "
                                                                           "was repeated on all 2,256 frames"],
    ], [6.6 * cm, TEXT_W - 6.6 * cm], caption="Deviations from the original plan. All additions are measurements; "
                                               "none is a defence, so the Version 1 scope is preserved.")


def reproducibility_section(story, D):
    runs = D["runs"]
    story.append(PageBreak())
    story.append(H1("10. Reproducibility"))
    story.append(H2("10.1 Environment"))
    story += table([
        ["Component", "Version / specification"],
        ["Hardware", "NVIDIA GeForce RTX 3060 Laptop GPU (6 GB), 16 GB RAM, Windows 11"],
        ["Python", "3.12"],
        ["Deep learning", "PyTorch 2.5.1 + CUDA 12.1, torchvision 0.20.1"],
        ["Scientific stack", "NumPy 1.26.2, SciPy 1.13.1, pandas 2.1.3, matplotlib 3.8.2, Pillow 10.1.0"],
        ["Data download", "synapseclient 4.14.0 (separate virtual environment)"],
        ["This document", "ReportLab 5.0.1, DejaVu fonts"],
    ], [3.6 * cm, TEXT_W - 3.6 * cm], caption="Software and hardware used for all reported results.",
        bold_first_col=True)
    story.append(H2("10.2 Commands"))
    for cmd, note in (("python download_synapse_dataset.py", "fetch frames (token in ~/.synapseConfig)"),
                      ("python convert_supervisely_to_masks.py", "build the 13-class masks"),
                      ("python cataract_dataset.py", "verify folds, split and class counts"),
                      ("python train_baseline.py --epochs 40", "train the clean baseline"),
                      ("python run_experiments.py", "41-run grid (resumable)"),
                      ("python analyze_results.py", "tables and figures"),
                      ("python report/build_results_pdf.py", "this document")):
        story.append(Paragraph(f"{cmd}   <font color='#52514e'># {note}</font>", ST["code"]))
    story.append(P("Self-tests: <font name='DejaVuMono'>python loss.py</font>, "
                   "<font name='DejaVuMono'>python metrics.py</font>, <font name='DejaVuMono'>python "
                   "model_builder.py</font>, <font name='DejaVuMono'>python attacks.py</font>."))
    story.append(H2("10.3 Artefacts"))
    story += table([
        ["Path", "Contents"],
        ["runs/baseline/", "best_model.pth (epoch 36), history.csv, training_curves.png, clean_report.txt"],
        ["results/raw/", "Per-run, per-frame, per-class TP/FP/FN counts and attack statistics (41 .npz files) — the "
                         "source of every number"],
        ["results/tables/, results/figures/", "Generated tables (CSV) and figures (PNG)"],
        ["results/RESULTS_TABLES.md, results/summary.json", "Auto-generated tables and key numbers"],
        ["results/qualitative/", "Perturbations and predictions for the four qualitative frames"],
        ["results_908frames/, runs/baseline_908frames/", "The replication run on 908 frames"],
        ["Results_and_Verification.md", "Markdown version of the findings"],
        ["report/", "This PDF and its build script"],
    ], [5.6 * cm, TEXT_W - 5.6 * cm], caption="Where the outputs live (project root).")
    story.append(H2("10.4 Run times"))
    att = sum(r["seconds"] for r in runs if r["id"] != "segpgd_eps4_T10")
    h = D["hist"]
    story.append(P(f"Training took {h.epoch_seconds.sum() / 60:.1f} minutes. The 41 evaluation runs took "
                   f"{att / 3600:.1f} hours of attack compute (excluding one run interrupted by a system sleep), "
                   "with the GPU partly shared with other applications. Per-run times are listed in Appendix D."))


def appendices(story, D):
    runs = D["runs"]
    story += to_landscape()
    # A. per-class IoU, all runs (transposed: runs as rows)
    story.append(H1("Appendix A. Per-class IoU for Every Run"))
    T3 = D["T3_per_class_iou"]
    classes = list(T3["class"])
    cols = ["clean"] + [c for c in T3.columns if "_eps" in c]
    short = {"Cornea": "Cornea", "Pupil": "Pupil", "Lens (IOL)": "Lens", "Slit/Incision Knife": "Knife*",
             "Gauge": "Gauge", "Capsulorhexis Cystotome": "Cyst.", "Capsulorhexis Forceps": "Caps. F.",
             "Katena Forceps": "Katena*", "Spatula": "Spatula", "Phacoemulsification Tip": "Phaco",
             "Irrigation-Aspiration": "Irr.-Asp.", "Lens Injector": "Injector"}
    T2b = D["T2b_eps_curve_and_quantisation"]
    T2 = D["T2_main_grid"]
    rows = [["Run", "ε"] + [short[c] for c in classes] + ["Headline mIoU"]]
    lab = {"noise": "Noise", "fgsm": "FGSM", "pgd": "PGD-20", "segpgd": "SegPGD-20"}
    for col in cols:
        if col == "clean":
            head = pc(T2[T2.run == "clean"].mIoU.iloc[0])
            rows.append(["<b>Clean</b>", "—"] + [pc(v, 1) for v in T3[col]] + [head])
            continue
        a, e = col.split("_eps")
        rid = f"{a}_eps{e}" + ("_T20" if a in ("pgd", "segpgd") else "")
        m = T2b[(T2b.run == rid)]
        m = m.mIoU.iloc[0] if len(m) else T2[T2.run == rid].mIoU.iloc[0]
        rows.append([lab[a], eps_s(float(e))] + [pc(v, 1) for v in T3[col]] + [pc(m)])
    w = [2.1, 1.2] + [1.62] * len(classes) + [2.0]
    s = LAND_W / (sum(w) * cm)
    story += table(rows, [x * cm * s for x in w],
                   caption="IoU (%) per class (columns) for every clean and attacked run (rows). * rare class. "
                           "PGD/SegPGD: 20 steps. Step-ablation and quantised runs: Appendix D.",
                   numeric_cols=list(range(1, len(classes) + 3)), font_scale=0.92, keep=False)
    # B. anatomy vs instruments, all budgets
    story.append(PageBreak())
    story.append(H1("Appendix B. Anatomy versus Instruments, All Budgets"))
    T4, T4c = D["T4_anatomy_vs_instruments"], D["T4c_present_frames_only"]
    rows = [["Attack", "ε", "Anat. rel. fall %", "Instr. rel. fall %", "Diff. pp (95% CI)", "Anat. Δ pp",
             "Instr. Δ pp", "Frames instr. worse", "Sign-test p", "Videos", "Present-only diff. pp"]]
    for _, r in T4.iterrows():
        rc = T4c[(T4c.attack == r.attack) & (T4c.eps_255 == r.eps_255)].iloc[0]
        p = r.sign_test_p
        rows.append([r.attack, eps_s(r.eps_255), f"{r.anatomy_rel_fall_pct:.1f}", f"{r.instrument_rel_fall_pct:.1f}",
                     f"{r.diff_pp:+.1f} ({r.diff_lo:+.1f} to {r.diff_hi:+.1f})", f"{r.anatomy_abs_drop_pp:.1f}",
                     f"{r.instrument_abs_drop_pp:.1f}", f"{int(r.frames_inst_worse)}/{int(r.frames_compared)}",
                     "&lt;0.001" if p < 0.001 else f"{p:.3f}", f"{int(r.videos_inst_worse)}/{int(r.videos)}",
                     f"{rc.diff_present_only_pp:+.1f}"])
    w = [2.2, 1.3, 2.0, 2.0, 3.4, 1.8, 1.8, 2.4, 1.8, 1.5, 2.3]
    s = LAND_W / (sum(w) * cm)
    story += table(rows, [x * cm * s for x in w],
                   caption="Group-level comparison for every attack and budget. Relative fall = (clean − attacked) / "
                           "clean of the group mIoU; Δ = absolute drop in points. Frames: per-frame sign test on "
                           "robustness ratios (frames containing a headline instrument; one-sided). Videos: test videos "
                           "in which instruments fall further. Present-only: each class scored only on frames where it "
                           "occurs.", numeric_cols=list(range(1, 11)), font_scale=0.92, keep=False)
    # C. size
    story.append(PageBreak())
    story.append(H1("Appendix C. Object Size versus Vulnerability"))
    sp = D["key"]["size_vs_fall_spearman"]
    T4b = D["T4b_object_size_vs_fall"]
    rows = [["Run", "ε", "Spearman ρ", "p", "Classes", "Mean rel. fall %"]]
    for rid, (rho, p, n) in sp.items():
        sub = T4b[(T4b.run == rid) & (~T4b.rare)]
        a = rid.split("_")[0]
        e = float(rid.split("_eps")[1].split("_")[0])
        rows.append([lab[a], eps_s(e), f"{rho:+.2f}", f"{p:.3f}", str(n), f"{sub.rel_fall_pct.mean():.1f}"])
    story += table(rows, [3.0 * cm, 2.0 * cm, 2.6 * cm, 2.4 * cm, 2.0 * cm, 3.0 * cm],
                   caption="Spearman correlation across the 10 headline classes between mean ground-truth area per "
                           "frame and relative IoU fall (negative ρ: smaller structures fall further). Runs where every "
                           "class is near 100% fall carry no size information.", numeric_cols=[1, 2, 3, 4, 5],
                   keep=False)
    areas = T4b[(T4b.run == "fgsm_eps1")][["class", "group", "mean_area_px"]].sort_values("mean_area_px")
    rows = [["Class", "Group", "Mean GT area per frame (px at 512×512)"]] + \
           [[esc(r["class"]), r["group"], f"{r.mean_area_px:,.0f}"] for _, r in areas.iterrows()]
    story += table(rows, [5.5 * cm, 2.6 * cm, 5.0 * cm], caption="Mean ground-truth area per frame (frames where the "
                                                                 "class is present).", numeric_cols=[2])
    # D. run list
    story.append(PageBreak())
    story.append(H1("Appendix D. Complete Run List"))
    T2b_all = D["T2b_eps_curve_and_quantisation"]
    T5 = D["T5_step_ablation"]
    per_step = [r["seconds"] / max(r["steps"], 1) for r in runs if r["steps"] > 1]
    limit = 5 * float(np.median(per_step))
    rows = [["#", "Run id", "Attack", "ε", "Steps", "8-bit", "Frames", "max ‖δ‖∞ (/255)", "Headline mIoU %",
             "Time (s)"]]
    for i, r in enumerate(runs, 1):
        rid = r["id"]
        m = T2b_all[T2b_all.run == rid]
        if len(m):
            miou = m.mIoU.iloc[0]
        elif (T2.run == rid).any():
            miou = T2[T2.run == rid].mIoU.iloc[0]
        else:
            e, t = r["eps"], r["steps"]
            a = {"pgd": "PGD", "segpgd": "SegPGD"}[r["attack"]]
            miou = T5[(T5.eps_255 == e) & (T5.attack == a) & (T5.steps == t)].mIoU.iloc[0]
        slow = r["steps"] > 1 and r["seconds"] / r["steps"] > limit
        rows.append([str(i), f"<font name='DejaVuMono'>{rid}</font>", r["attack"], eps_s(r["eps"]) if r["eps"] else "—",
                     str(r["steps"]), "yes" if r.get("quantise") else "", str(r["n"]), f"{r['linf']:.3f}", pc(miou),
                     f"{r['seconds']:,.0f}" + (" ‡" if slow else "")])
    w = [0.8, 4.6, 1.8, 1.4, 1.2, 1.2, 1.4, 2.6, 2.6, 2.0]
    s = LAND_W / (sum(w) * cm)
    story += table(rows, [x * cm * s for x in w],
                   caption="All 41 runs in execution order. ‡ wall-clock time inflated by a system sleep during the "
                           "run; the result is unaffected. The measured maximum perturbation equals ε in every "
                           "attacked run.", numeric_cols=[0, 3, 4, 6, 7, 8, 9], font_scale=0.92, keep=False)
    story += to_portrait()
    # E. glossary
    story.append(H1("Appendix E. Glossary"))
    story += table([
        ["Term", "Meaning"],
        ["Adversarial example", "An input with a small, deliberately computed perturbation that changes a model’s output"],
        ["White-box", "The attacker has full access to the model, its weights and gradients"],
        ["Untargeted", "The attacker wants any wrong output, not a specific one"],
        ["L∞ budget ε", "Maximum change of any single pixel value; ε = 1/255 is one 8-bit intensity level"],
        ["FGSM", "Fast Gradient Sign Method: one step of size ε along the sign of the loss gradient"],
        ["PGD-T", "Projected Gradient Descent: T small steps, each projected back into the ε-ball"],
        ["SegPGD", "Segmentation-specific PGD that reweights correctly and wrongly classified pixels over the steps"],
        ["IoU / mIoU", "Intersection over Union per class; its mean over a set of classes"],
        ["Dice", "2·TP / (2·TP + FP + FN); closely related to IoU"],
        ["Relative fall", "(clean − attacked) / clean × 100"],
        ["Robustness ratio", "attacked / clean"],
        ["Correct→wrong", "Share of pixels correct on the clean frame that the attack made wrong"],
        ["PSNR", "Peak signal-to-noise ratio between clean and adversarial frame; higher = less visible"],
        ["Bootstrap CI", "Confidence interval from resampling the test frames with replacement"],
        ["Saturation", "An attack so strong that the metric is already near its floor, hiding further differences"],
        ["Hallucination", "The model predicting a structure where none exists"],
        ["OS16", "Output stride 16: the encoder’s deepest features are 1/16 of the input resolution"],
    ], [3.8 * cm, TEXT_W - 3.8 * cm], caption="Terms used in this document.", bold_first_col=True, keep=False)
    # F. references
    story.append(PageBreak())
    story.append(H1("Appendix F. Key References"))
    refs = [
        "N. Ghamsarian et al. Cataract-1K Dataset for Deep-Learning-Assisted Analysis of Cataract Surgery Videos. "
        "<i>Scientific Data</i>, 2024. arXiv:2312.06295.",
        "I. J. Goodfellow, J. Shlens, C. Szegedy. Explaining and Harnessing Adversarial Examples. ICLR 2015. "
        "arXiv:1412.6572.",
        "A. Madry et al. Towards Deep Learning Models Resistant to Adversarial Attacks. ICLR 2018. arXiv:1706.06083.",
        "J. Gu et al. SegPGD: An Effective and Efficient Adversarial Attack for Evaluating and Boosting Segmentation "
        "Robustness. ECCV 2022. arXiv:2207.12391.",
        "F. Croce, N. D. Singh, M. Hein. Robust Semantic Segmentation: Strong Adversarial Attacks and Fast Training of "
        "Robust Models (SEA). arXiv:2306.12941, 2023.",
        "L.-C. Chen et al. Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation "
        "(DeepLabV3+). ECCV 2018.",
        "K. He et al. Deep Residual Learning for Image Recognition. CVPR 2016.",
        "C. Xie et al. Adversarial Examples for Semantic Segmentation and Object Detection. ICCV 2017. arXiv:1703.08603.",
        "A. Arnab, O. Miksik, P. H. S. Torr. On the Robustness of Semantic Segmentation Models to Adversarial "
        "Attacks. CVPR 2018. arXiv:1711.09856.",
        "X. Ma et al. Understanding Adversarial Attacks on Deep Learning Based Medical Image Analysis Systems. "
        "<i>Pattern Recognition</i>, 2021. arXiv:1907.10456.",
        "A. Malik et al. On Evaluating Adversarial Robustness of Volumetric Medical Segmentation Models. BMVC 2024. "
        "arXiv:2406.08486.",
        "SegSTRONG-C: Segmenting Surgical Tools Robustly On Non-adversarial Generated Corruptions. EndoVis 2024 / "
        "MICCAI. arXiv:2407.11906.",
        "M. Khan et al. GUARDIAN: Guarding Against Uncertainty and Adversarial Risks in Robot-Assisted Surgeries. "
        "MICCAI 2024 workshop.",
    ]
    for i, r in enumerate(refs, 1):
        story.append(Paragraph(r, ParagraphStyle(f"ref{i}", parent=ST["bullet"], fontSize=8.6, leading=12),
                               bulletText=f"[{i}]"))
    story.append(Spacer(1, 6))
    story.append(P("The full literature record (~65 papers) and the 22-paper gap matrix are in "
                   "<font name='DejaVuMono'>Literature_Review_Paper_Pool.md</font> and "
                   "<font name='DejaVuMono'>Paper_Gap_Matrix.md</font>.", "small"))


def main():
    D = load()
    story = []
    cover(story, D)
    toc_page(story)
    executive_summary(story, D)
    project_context(story, D)
    data_section(story, D)
    method_section(story, D)
    results_section(story, D)
    answers_section(story, D)
    verification_section(story, D)
    limitations_section(story, D)
    reproducibility_section(story, D)
    appendices(story, D)
    doc = ReportDoc(OUT, title="Adversarial Robustness of Cataract Surgical Segmentation — Results & Verification",
                    author="Parikshat Singh, Venu Kulshreshtha, Sayyad Ahsan (BVCOE)",
                    subject="Version 1 micro-project: results and verification")
    doc.multiBuild(story, canvasmaker=NumberedCanvas)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
