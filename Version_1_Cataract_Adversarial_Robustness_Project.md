# Micro Project — Version 1
## Adversarial Attack and Robustness Analysis of AI-Based Cataract Surgical Instrument Detection

**Project Type:** Micro Project  
**Department:** Computer Science & Engineering  
**Institution:** Bharati Vidyapeeth’s College of Engineering, Delhi  
**Version:** 1 — Attack + Robustness Analysis  
**Status:** Working project specification for synopsis and research

---

## 1. Project Overview

This micro project studies the vulnerability of an AI-based cataract surgical instrument detection/segmentation system to adversarial attacks.

The project will first establish a **baseline AI model** for identifying surgical instruments in cataract surgical scenes. Its performance will be measured on clean, unmodified data. Adversarial perturbations will then be generated and applied to the input images/frames. The attacked inputs will be passed through the same model, and the change in model performance will be measured.

The primary purpose is **not to develop a defense or restoration method**. Version 1 is strictly focused on:

> **Attack → Measure degradation → Analyze vulnerability → Evaluate robustness**

The project will therefore provide an experimental analysis of how reliably a cataract surgical instrument AI model performs when its inputs are deliberately perturbed.

---

## 2. Final Project Title

**Adversarial Attack and Robustness Analysis of AI-Based Cataract Surgical Instrument Segmentation**

**LOCKED** (2 September 2026). "Detection" was replaced by "Segmentation" after the dataset was
verified: Cataract-1K's annotated subset is a surgical *scene segmentation* set with published
per-class Dice/IoU baselines and patient-disjoint folds. See §10 and §25.


---

## 3. Core Problem

AI-based computer vision systems are increasingly used to understand surgical scenes, including identifying or segmenting surgical instruments. In a surgical environment, reliable model predictions are important because incorrect predictions can affect downstream computer-assisted analysis.

Deep learning models can be vulnerable to **adversarial examples**: inputs containing carefully designed perturbations that may be visually subtle but can cause a model's predictions to change or degrade.

The project investigates this issue in the specific context of **cataract surgical instrument detection/segmentation**.

The central problem is:

> **How vulnerable is an AI-based cataract surgical instrument detection/segmentation model to adversarial perturbations, and how significantly do such attacks affect its performance?**

---

## 4. Research Motivation

The project is motivated by three areas:

1. **Surgical AI is becoming increasingly relevant** for understanding and analyzing surgical video.
2. **Deep learning models are not inherently robust** to adversarial perturbations.
3. **Cataract surgical scenes provide a focused domain** in which the effect of adversarial attacks can be experimentally studied.

Rather than attempting to solve every aspect of adversarial robustness, this micro project focuses on establishing a clear experimental baseline and quantifying vulnerability.

---

## 5. Research Gap — To Be Verified Through Literature

The final research gap must be established from the literature review rather than assumed.

The literature search will investigate:

- Existing AI methods for cataract surgical instrument detection/segmentation.
- Existing work using the Cataract-1K dataset.
- Adversarial attacks against medical imaging models.
- Adversarial attacks against object detection and segmentation models.
- Adversarial robustness of surgical AI/computer vision systems.
- Any work specifically studying adversarial attacks against cataract surgical instrument models.

A possible working hypothesis for the gap is:

> Existing research has extensively explored AI-based surgical instrument detection/segmentation and has separately studied adversarial vulnerabilities in medical AI, while the robustness of cataract surgical instrument detection/segmentation models against adversarial attacks may be comparatively underexplored.

**Important:** This statement must be confirmed and refined after collecting and analyzing relevant research papers.

---

## 6. Research Question

### Primary Research Question

> **How do adversarial attacks affect the performance and robustness of AI-based cataract surgical instrument detection/segmentation models?**

### Possible Supporting Questions

1. How much does model performance decrease when adversarial perturbations are introduced?
2. Which attack settings produce greater degradation?
3. Are some surgical instruments or scene conditions more vulnerable than others?
4. How does the model's performance on clean inputs compare with its performance on adversarial inputs?
5. How does attack strength affect the trade-off between perturbation magnitude and model degradation?

These supporting questions should be retained only if they are practical within the final experimental design.

---

## 7. Objectives

### Primary Objective

To experimentally evaluate the robustness of an AI-based cataract surgical instrument detection/segmentation model against adversarial attacks.

### Specific Objectives

1. Study existing research on cataract surgical AI and adversarial attacks.
2. Select an appropriate cataract surgical dataset and AI model.
3. Establish baseline performance on clean input images/frames.
4. Implement one or more feasible adversarial attack methods.
5. Generate adversarial examples against the selected model.
6. Evaluate model performance on adversarial inputs.
7. Quantify performance degradation relative to the clean baseline.
8. Analyze the relationship between attack strength and model degradation.
9. Identify patterns of vulnerability in the model and/or instrument categories.
10. Document the experimental findings and limitations.

---

## 8. Project Scope

### Included

- Cataract surgical scene analysis.
- AI-based surgical instrument detection/segmentation.
- Clean-data baseline evaluation.
- Adversarial input generation.
- Experimental attack evaluation.
- Quantitative robustness analysis.
- Comparison between clean and adversarial performance.
- Visualization of representative clean/adversarial examples.
- Analysis of attack strength and performance degradation.

### Not Included in Version 1

- Image restoration.
- Image purification.
- Adversarial defense mechanisms.
- Adversarial training.
- Robust model retraining as a defense.
- Clinical deployment.
- Real-time surgical intervention.
- Claims about actual patient safety.
- Development of a complete clinical decision-support system.

If a defense/restoration component is later considered, it will be treated as a separate Version 2 project rather than silently added to Version 1.

---

## 9. Dataset

### Planned Dataset

**Cataract-1K Surgical Scene Segmentation Set**

The previously established project plan identifies the Cataract-1K dataset as the intended dataset.

The current project notes specify:

- **Dataset:** Cataract-1K Surgical Scene Segmentation Set
- **Synapse ID:** syn53395479
- **Annotated frames:** 2,256

The exact class structure, train/validation/test split, annotation format, and task definition should be verified directly from the dataset source and the associated publication before they are stated in the final synopsis.

### Dataset Role

The dataset will be used to:

1. Train or evaluate the baseline model.
2. Establish clean-input performance.
3. Generate adversarial inputs.
4. Compare clean and attacked predictions.
5. Analyze robustness.

---

## 10. Detection vs Segmentation — Important Decision

The current project title uses **"detection"**, while the selected Cataract-1K project context is a **surgical scene segmentation** dataset.

Before finalizing the synopsis, the exact task must be aligned:

### Option A — Instrument Segmentation

The model predicts the pixels/regions belonging to each instrument.

This is naturally compatible with segmentation metrics such as:

- IoU
- mIoU
- Dice score
- Precision
- Recall

### Option B — Instrument Detection

The model predicts bounding boxes/classes for instruments.

This is evaluated using metrics such as:

- Precision
- Recall
- F1-score
- AP
- mAP

### Decision — RESOLVED (2 September 2026)

**Option A (segmentation) is selected.** Evidence, verified from the Cataract-1K repository and the
paper's full text:

- The annotated subset is a *surgical scene segmentation* set: 2,256 frames, 3 anatomy + 9 instrument
  classes, Supervisely polygons.
- Dice/IoU baselines for 11 architectures and 5 patient-disjoint folds are published for the
  segmentation task, so our clean baseline is externally checkable.
- Detection would require no extra annotation (COCO boxes with bounding boxes ship in the same
  release), so it is retained as **optional** scope, not core.

Final title: **Adversarial Attack and Robustness Analysis of AI-Based Cataract Surgical Instrument
Segmentation** (§2).


---

## 11. Proposed Methodology

The planned experimental pipeline is:

```text
Cataract Surgical Dataset
          ↓
Data Preparation / Preprocessing
          ↓
Baseline AI Model
          ↓
Clean Input Evaluation
          ↓
Baseline Performance
          ↓
Adversarial Attack
          ↓
Adversarial Images / Frames
          ↓
Same AI Model
          ↓
Adversarial Performance
          ↓
Performance Degradation Analysis
          ↓
Robustness / Vulnerability Analysis
```

### Step 1 — Literature Review

Research will be collected in four main areas:

1. Cataract surgical AI / instrument detection or segmentation.
2. Adversarial attacks in medical imaging.
3. Adversarial attacks against detection/segmentation models.
4. Adversarial robustness in surgical AI.

The literature review will determine the final model, attack method, evaluation protocol, and research gap.

### Step 2 — Dataset Preparation

The dataset will be inspected and prepared for experimentation.

Possible operations include:

- Reading image/frame data.
- Reading segmentation masks or labels.
- Resizing/cropping where required by the model.
- Normalization.
- Train/validation/test organization.
- Checking class distribution.
- Removing unusable/corrupted samples if necessary.

The exact preprocessing procedure will depend on the selected model and dataset format.

### Step 3 — Baseline Model

A suitable deep learning architecture will be selected based on:

- Compatibility with the dataset.
- Compatibility with the task.
- Availability of pretrained weights or established implementations.
- Computational feasibility.
- Relevance in existing literature.
- Ease of generating adversarial examples.

The final architecture should be selected after the literature review rather than assumed in advance.

### Step 4 — Clean Evaluation

Before any attack is applied, the model must be evaluated on clean inputs.

This produces the baseline performance against which attacked performance will be compared.

For segmentation, candidate metrics include:

- IoU / mIoU
- Dice score
- Precision
- Recall
- F1-score

For detection, candidate metrics include:

- Precision
- Recall
- F1-score
- AP
- mAP

Only metrics appropriate to the final task will be included.

### Step 5 — Adversarial Attack

An adversarial attack will be applied to the model inputs.

Potential attack methods include:

- FGSM
- PGD
- Other established gradient-based attacks if justified by the literature and computational resources.

The final attack selection should be based on:

- Literature relevance.
- Model compatibility.
- Implementation feasibility.
- Computational cost.
- Suitability for the selected task.

### Step 6 — Generate Adversarial Inputs

The attack will generate perturbed images/frames designed to cause incorrect or degraded model predictions.

Conceptually:

```text
Clean Input
     +
Adversarial Perturbation
     ↓
Adversarial Input
```

The perturbation magnitude will be controlled using the relevant attack parameters.

### Step 7 — Evaluate Attacked Model

The same baseline model will process the adversarial inputs.

The attacked performance will be measured using the same primary evaluation metrics used for clean data.

This makes the comparison meaningful:

```text
Clean Performance
        vs
Adversarial Performance
```

### Step 8 — Quantify Degradation

Performance degradation can be reported using measures such as:

```text
Absolute Degradation =
Clean Metric − Adversarial Metric
```

and, where appropriate:

```text
Relative Degradation (%) =
((Clean Metric − Adversarial Metric) / Clean Metric) × 100
```

The exact reporting method will depend on the selected metric.

### Step 9 — Robustness Analysis

The final analysis may examine:

- Effect of attack strength.
- Performance degradation.
- Attack success/failure.
- Per-class vulnerability.
- Qualitative visual changes in predictions.
- Differences between clean and adversarial outputs.

---

## 12. Adversarial Attack Strategy

The project should not implement many attacks simply to increase the number of experiments.

A small number of well-justified attacks is preferable.

### Initial Candidate: FGSM

**Fast Gradient Sign Method (FGSM)** is a strong candidate for a micro project because it is:

- Conceptually simple.
- Computationally inexpensive.
- Widely studied.
- Relatively easy to implement.
- Suitable for establishing an initial adversarial baseline.

### Initial Candidate: PGD

**Projected Gradient Descent (PGD)** can be considered as a stronger iterative attack.

It can help study how increasing attack strength/iterations affect model robustness.

### Final Selection

The final attack set should be decided after reviewing relevant papers and testing feasibility.

A possible experimental progression is:

```text
Baseline
   ↓
FGSM
   ↓
Different perturbation strengths
   ↓
Optional PGD
   ↓
Compare degradation
```

---

## 13. Experimental Variables

Important variables may include:

### Independent Variables

- Attack type.
- Perturbation strength.
- Number of attack iterations, for iterative attacks.
- Dataset sample/category.

### Dependent Variables

- IoU/mIoU.
- Dice.
- Precision.
- Recall.
- F1-score.
- AP/mAP if the task is detection.
- Attack success/degradation measures.

### Controlled Variables

- Same dataset split.
- Same model.
- Same evaluation protocol.
- Same preprocessing.
- Same hardware/software environment where possible.

---

## 14. Expected Results

The project is expected to determine whether adversarial perturbations cause measurable degradation in cataract surgical instrument AI performance.

Expected observations may include:

1. Clean inputs produce stronger model performance than adversarial inputs.
2. Increasing attack strength may cause greater performance degradation.
3. Different attack methods may produce different levels of vulnerability.
4. Some instrument classes or visual conditions may be more susceptible than others.
5. The model may produce incorrect, incomplete, or shifted predictions under attack.

These are **expected hypotheses**, not guaranteed results. The final synopsis should not present them as experimental findings before experiments are completed.

---

## 15. Expected Contribution

The project's contribution is intended to be an **empirical robustness analysis**, not a new defense algorithm.

The project can contribute:

- A reproducible experimental pipeline for attacking a cataract surgical AI model.
- Quantitative comparison of clean vs adversarial performance.
- Analysis of model vulnerability.
- Analysis of attack-strength effects.
- Evidence that can inform future work on robustness/defense in surgical AI.

---

## 16. Evaluation and Result Presentation

The results should ideally include:

### Quantitative Results

Example structure:

| Experiment | Attack Strength | Metric | Clean | Adversarial | Degradation |
|---|---:|---|---:|---:|---:|
| Baseline | — | mIoU | TBD | — | — |
| FGSM | ε₁ | mIoU | TBD | TBD | TBD |
| FGSM | ε₂ | mIoU | TBD | TBD | TBD |
| PGD | ε₁ | mIoU | TBD | TBD | TBD |

Values will be filled only after experiments.

### Visual Results

Show representative:

- Original frame.
- Adversarial frame.
- Ground-truth mask/label.
- Clean prediction.
- Adversarial prediction.

This can make the degradation easier to understand.

### Graphs

Possible plots:

- Attack strength vs model performance.
- Attack strength vs performance degradation.
- Clean vs attacked metric comparison.
- Per-class robustness comparison.

---

## 17. Software and Technologies

Potential tools:

- Python
- PyTorch
- torchvision
- OpenCV
- NumPy
- Pandas
- Matplotlib
- CUDA
- GPU environment such as Google Colab or a local GPU system

The exact software stack will depend on the selected architecture and attack implementation.

---

## 18. Hardware Requirements

A GPU is strongly preferable for training and adversarial experimentation.

The final hardware specification should be based on the selected model and dataset.

Possible environments:

- Google Colab GPU.
- Institutional GPU.
- Local NVIDIA GPU.

The project should remain computationally feasible for a micro-project timeline.

---

## 19. Research Paper Search Plan

The literature search is a major next step.

### Category 1 — Cataract Surgical AI

Search for papers containing combinations of:

- Cataract-1K
- cataract surgery AI
- cataract surgical instrument
- surgical instrument segmentation
- surgical scene segmentation
- cataract surgical video analysis

### Category 2 — Medical Adversarial Attacks

Search for:

- adversarial attacks medical imaging
- adversarial examples medical image segmentation
- adversarial attacks medical AI
- FGSM medical imaging
- PGD medical imaging

### Category 3 — Surgical AI Robustness

Search for:

- adversarial attacks surgical AI
- adversarial surgical image segmentation
- adversarial surgical instrument detection
- adversarial attacks surgical video
- robustness surgical computer vision

### Category 4 — Directly Related Work

Search specifically for:

- adversarial attacks cataract surgical instruments
- adversarial cataract surgery AI
- adversarial Cataract-1K
- robustness Cataract-1K
- adversarial surgical instrument segmentation

The goal is to determine whether closely matching work already exists.

---

## 20. Literature Review Table

**DONE (2 September 2026).** The literature record lives in two companion files rather than being
duplicated here:

- **[Literature_Review_Paper_Pool.md](Literature_Review_Paper_Pool.md)** — ~65 papers in six
  categories, with verified dataset facts and per-paper relevance notes.
- **[Paper_Gap_Matrix.md](Paper_Gap_Matrix.md)** — the 22 papers that matter, each with *what it did*
  / *limitation or gap* / *how our project differs*, plus the four questions a mentor will ask and
  the answers. This is the defence document.

Coverage against the targets originally set in this section:

| Category | Target | Papers | Status |
|---|---:|---|---|
| Cataract / surgical instrument AI | 5–8 | P1–P5 | met |
| Adversarial attacks in medical imaging | 5–8 | P12–P16, P21 | met |
| Adversarial attacks / robustness in surgical AI | 3–5 | P17–P20 | met |
| Directly relevant to the chosen model/task | 2–4 | P6–P11, P22 | met |

---

## 21. Synopsis Structure

The synopsis can be developed using the following structure:

1. Title Page
2. Abstract
3. Introduction
4. Problem Statement
5. Motivation
6. Research Gap
7. Research Question
8. Objectives
9. Scope
10. Literature Review
11. Dataset Description
12. Proposed Methodology
13. Model Selection
14. Adversarial Attack Method
15. Experimental Setup
16. Evaluation Metrics
17. Expected Results
18. Expected Contribution
19. Limitations
20. Tools and Technologies
21. Work Plan / Timeline
22. References

The exact institutional format should be checked against the official synopsis/report guidelines if available.

---

## 22. Limitations

Potential limitations include:

- Limited dataset size.
- Dependence on one dataset/domain.
- Dependence on the selected model architecture.
- Computational constraints.
- Limited number of adversarial attack methods.
- Laboratory evaluation rather than clinical deployment.
- Results may not generalize to all surgical AI systems.
- Adversarial perturbations used in experiments may not represent every real-world threat.

The project should explicitly avoid making clinical safety claims based solely on these experiments.

---

## 23. Risk Management

### Risk: Dataset/task mismatch

**Mitigation:** Verify the dataset annotations and task before finalizing the title and model.

### Risk: Attack implementation is incompatible with the model

**Mitigation:** Select an attack supported by the final architecture and task.

### Risk: Training takes too long

**Mitigation:** Use transfer learning/pretrained models where appropriate and reduce experimental scope while preserving valid comparisons.

### Risk: Attack produces weak degradation

**Mitigation:** Test multiple perturbation strengths and verify attack implementation rather than assuming failure means robustness.

### Risk: Too many experiments

**Mitigation:** Prioritize one strong baseline attack and optionally one iterative attack rather than implementing many methods.

---

## 24. What Must Be Verified Before Final Synopsis

The following points should not be treated as finalized until verified:

- Exact Cataract-1K dataset description.
- Exact number of images/frames and annotations.
- Dataset classes.
- Dataset split.
- Whether the project should use segmentation or detection terminology.
- Most suitable baseline model.
- Most appropriate adversarial attack(s).
- Attack parameters.
- Evaluation metrics.
- Existing directly related papers.
- Exact research gap.
- Novelty claim.

This verification is important because the synopsis should be based on published evidence and the actual dataset rather than assumptions.

---

## 25. Current Project Decision Log

### Finalized

- **Version:** Version 1.
- **Focus:** Adversarial attack + robustness analysis.
- **Domain:** Cataract surgical AI.
- **Dataset direction:** Cataract-1K.
- **Main purpose:** Measure vulnerability/performance degradation.
- **Restoration/defense:** Excluded from Version 1.

### Locked on 2 September 2026 (after literature review)

**Task wording** — Segmentation. Title fixed in §2.

**Model** — DeepLabV3+ with an ImageNet-pretrained ResNet-50 encoder, output stride 16, single
multi-class decoder (13 channels = background + 3 anatomy + 9 instruments), 512×512 input.
*Why:* it is one of the 11 architectures benchmarked in the Cataract-1K paper, so the clean baseline
is externally checkable; it is a conventional fully differentiable encoder–decoder with a clean
gradient path to the input; and it avoids the SAM-family confound in which FGSM/PGD are reported to be
ineffective (P22), which would make a weak result uninterpretable.
*Secondary (optional, only if time permits):* UNet++ / ResNet-34, for an architecture-sensitivity arm.

**Attack set** — three attacks, one threat model:
1. FGSM — single-step reference (P6).
2. PGD-20, L∞, random start, step α = ε/4 — iterative reference (P7).
3. **SegPGD-20 — primary attack** (P10), because a pixel-averaged loss saturates and understates
   segmentation vulnerability (P7, P11).
*Threat model:* white-box, untargeted, L∞, digital. Perturbation applied in [0,1] RGB **before**
normalisation and clipped to the valid range. Transfer/black-box, patch and frequency attacks are
Version 2.

**Attack parameters** — ε ∈ {1, 2, 4, 8}/255 for all three attacks (12 attacked evaluations + 1 clean
= 13 runs). Step-count ablation at ε = 4/255 with steps ∈ {5, 10, 20, 40} for PGD and SegPGD.

**Evaluation protocol** — Cataract-1K segmentation subset only (`syn53395479`); the authors'
patient-disjoint folds, single split (Fold 1 = test, 6 videos; remaining 24 videos → 20 train /
4 validation); mIoU and mean Dice as headline metrics, **per-class IoU always reported**, plus
absolute degradation, relative degradation (%), robustness ratio, and mIoU-vs-ε curves. Rare classes
(Katena forceps 1.29%, slit knife 2.22%) reported separately from the headline means.

**Research gap** — established in [Paper_Gap_Matrix.md](Paper_Gap_Matrix.md); wording in the synopsis.

**Novelty claim** — first quantified adversarial vulnerability profile of cataract surgical scene /
instrument segmentation, and first anatomy-vs-instrument vulnerability comparison within a single
surgical frame. Bounded as "no matching work surfaced across the searches listed in the paper pool",
not as an absolute claim, pending the Scopus/WoS/IEEE re-check.

### Still open (does not block the synopsis)

- GUARDIAN (P17) full text — could narrow the gap wording.
- Cataract-1K per-baseline Dice/IoU numbers, for the clean-baseline comparison sentence.
- Whether the optional detection arm and the optional second architecture are run at all.


---

## 26. Version 1 in One Sentence

> **This project will establish an AI-based cataract surgical instrument segmentation baseline, subject it to adversarial attacks, and quantitatively analyze the resulting degradation to assess the model's robustness and vulnerability.**

---

## 27. Immediate Next Steps

| # | Step | Status |
|---:|---|---|
| 1 | Verify the Cataract-1K dataset and its associated research paper | done — see paper pool §0 |
| 2 | Collect relevant research papers in the four literature categories | done — ~65 papers |
| 3 | Build the literature review table | done — [Paper_Gap_Matrix.md](Paper_Gap_Matrix.md) |
| 4 | Identify the defensible research gap | done — gap matrix, closing section |
| 5 | Decide detection vs segmentation | done — segmentation (§10) |
| 6 | Select the baseline model | done — DeepLabV3+/ResNet-50 (§25) |
| 7 | Select the attack method(s) | done — FGSM, PGD-20, SegPGD-20 (§25) |
| 8 | Finalize evaluation metrics | done — §25 |
| 9 | Write the complete synopsis | done — `Cataract_Adversarial_Robustness_Synopsis.docx` |
| 10 | Download `syn53395479`, run the experiments, replace expected results with measured ones | **next** |


---

## 28. Important Rule for Future Work

**Do not add restoration, purification, adversarial training, or defense mechanisms to this Version 1 project unless the project is explicitly changed to Version 2.**

The central experimental question remains:

> **How much does an adversarial attack affect the performance and robustness of an AI model used for cataract surgical instrument detection/segmentation?**
