# Paper-by-Paper Gap Matrix

**Project:** Adversarial Attack and Robustness Analysis of AI-Based Cataract Surgical Instrument Segmentation (Version 1)
**Compiled:** 2 September 2026
**Companions:** [Paper pool (~65 papers)](Literature_Review_Paper_Pool.md) · [Version 1 spec](Version_1_Cataract_Adversarial_Robustness_Project.md)

This is the defence document: 22 papers distilled from the 65-paper pool, chosen because a mentor can
ask "has this been done?" about any of them. Each row states what the paper actually did, the
limitation that keeps it from being our project, and how our work differs.

Anything not confirmed from a reachable primary source is marked **(verify)**.

---

## Block 1 — Cataract and surgical instrument segmentation (the task we attack)

| # | Paper (Year) | What it did | Limitation / Gap | How our project differs |
|---|---|---|---|---|
| P1 | Ghamsarian et al., **Cataract-1K**, *Scientific Data*, 2024 | Released 1,000 cataract surgery videos; segmentation subset = 2,256 frames from 30 videos at 1024×768, 3 anatomy + 9 instrument classes, 5 patient-disjoint folds; benchmarked 11 networks (DeepPyramid, Adapt-Net, UNet++, ReCal-Net, CPFNet, CE-Net, FED-Net, scSENet, DeepLabV3+, UPerNet, U-Net) by mean Dice/IoU at 512×512 | Clean-data benchmarking only. No perturbation, corruption or robustness axis anywhere in the paper; "reliability" is accuracy on held-out folds | We reuse its split, input size and one of its benchmarked networks, then add the missing axis — bounded adversarial perturbation — and report degradation against its clean numbers |
| P2 | **RP-SAM2**, 2025 (arXiv:2504.07117) | Point-prompt refinement for SAM2-based surgical instrument segmentation, evaluated **on Cataract-1K**: +2% mDSC, −21.36% mHD95, reduced variance across random single-point prompts | "Stability" here means invariance to prompt sampling, not to input perturbation. No adversarial evaluation. State of the art on our dataset, and still untested against attack | We define robustness as invariance to a worst-case bounded input perturbation, and deliberately avoid a SAM-family backbone (see P22) so a weak-attack artefact cannot be read as robustness |
| P3 | Sinha, Balazia, Bremond, IPAS 2025 (arXiv:2501.02618) | Optimised aggregation network for identifying surgical instruments in pedagogical cataract surgery videos, real-time oriented | Accuracy and latency on clean video only; instrument identification rather than pixel-level masks; no threat model | Same clinical domain, opposite question: we quantify failure under deliberate perturbation rather than performance under ideal input |
| P4 | Benchmarking CNN- and Transformer-Based Models for Surgical Instrument Segmentation, 2026 (arXiv:2604.09151) | Head-to-head UNet / Attention-UNet / DeepLabV3 vs SegFormer for surgical instrument segmentation | Compares on clean data; architecture is justified by accuracy alone | Supplies the accuracy-side justification for our architecture choice; we add the vulnerability axis, which P13 shows can rank architectures differently from accuracy |
| P5 | **Cataract-LMM**, *Scientific Data*, 2026 | 3,000 phacoemulsification videos, multi-centre, four annotation layers (phases, instance segmentation, instrument–tissue interaction, skill scores) | Larger, newer and multi-task, but still a clean-accuracy benchmark; defines no robustness or security task | Evidence that the field is still scaling clean benchmarks rather than stress-testing them; our protocol is dataset-agnostic and transfers to it later |

---

## Block 2 — Adversarial attacks on dense prediction (the method we use)

| # | Paper (Year) | What it did | Limitation / Gap | How our project differs |
|---|---|---|---|---|
| P6 | Goodfellow, Shlens, Szegedy, **FGSM**, ICLR 2015 | Explained adversarial examples as a consequence of local linearity; proposed the single-step sign-of-gradient attack | Image classification; one step; no dense-prediction formulation | Used as our single-step lower bound on attack strength, with the loss aggregated over the whole segmentation map rather than one label |
| P7 | Madry et al., **PGD**, ICLR 2018 | Cast robustness as a saddle-point problem; multi-step projected gradient descent inside an L∞ ball with random start — the standard strong first-order attack | Classification. On segmentation, an averaged cross-entropy objective saturates once most pixels are already misclassified, so plain PGD understates vulnerability | PGD is our iterative reference point, not our strongest attack, precisely because of that saturation |
| P8 | Xie et al., **DAG**, ICCV 2017 | First attack designed for dense prediction: optimises over a set of pixel/proposal targets to attack segmentation and detection | Pre-dates modern encoder–decoder baselines; natural images (VOC); no medical or surgical evaluation | We inherit its core insight — dense prediction needs a dense objective — in its current SegPGD form, in a surgical domain |
| P9 | Arnab, Miksik, Torr, CVPR 2018 | First systematic robustness study of semantic segmentation networks (Cityscapes, VOC); analysed architecture, multiscale processing and CRFs | Natural-scene driving and object data; no medical imaging; predates segmentation-specific attacks | We follow its evaluation logic (clean vs attacked mIoU, per-class analysis, architecture sensitivity) in surgical video and with a stronger attack |
| P10 | Gu et al., **SegPGD**, ECCV 2022 | Segmentation-specific PGD that progressively reweights correctly vs incorrectly classified pixels, producing stronger attacks at an equal iteration budget | Evaluated on natural-image benchmarks; the paper also uses it for adversarial training, which is outside Version 1 | Adopted as our **primary** attack. We use only the attack half; the adversarial-training half is explicitly deferred to Version 2 |
| P11 | Croce, Singh, Hein, **SEA**, 2023 (arXiv:2306.12941) | Showed that standard attacks *severely overestimate* segmentation robustness; proposed an ensemble of complementary losses for reliable evaluation | Natural images; the ensemble is expensive for a micro-project compute budget | We adopt its honesty requirement: any small measured degradation is reported as an upper bound on robustness, never as proof of it. SEA is listed as the first Version-2 extension |

---

## Block 3 — Adversarial robustness in medical image analysis (the evidence base)

| # | Paper (Year) | What it did | Limitation / Gap | How our project differs |
|---|---|---|---|---|
| P12 | Ma et al., *Pattern Recognition*, 2021 | Systematic study of adversarial attacks on medical deep-learning systems across modalities. Found medical models are **more** vulnerable than natural-image models, yet their adversarial examples are **easier to detect** | Classification of diagnostic images; no segmentation, no surgical video | Supplies the prior that our surgical model should be at least as fragile, and pre-empts the "why not just detect the attack" question. We test dense prediction, where the detectability result does not automatically carry over |
| P13 | Malik et al., BMVC 2024 (arXiv:2406.08486) | Benchmarked adversarial robustness of **volumetric medical segmentation** across CNN, transformer and Mamba models on four datasets, with pixel- and frequency-domain attacks, white-box and transfer black-box. Transformers > CNNs > Mamba; frequency attacks transfer best | 3D radiology volumes (CT/MRI); no 2D surgical video; no instrument classes; no intra-operative imaging conditions | **Our methodological template.** Same white-box untargeted protocol and ε sweep, applied to 2D intra-operative cataract video with instrument and anatomy classes in the same frame |
| P14 | Susceptibility of Adversarial Attack on Medical Image Segmentation Models, 2024 (arXiv:2401.11224) | Applied FGSM to several medical segmentation networks and correlated attack success with model properties; found **negligible correlation between parameter count and attack success** | Single-step attack only; non-surgical medical images; shallow ε analysis and no iterative or segmentation-aware attack | Closest prior experiment in spirit. We extend it from FGSM-only to single-step vs iterative vs segmentation-aware, and from diagnostic images to surgical video |
| P15 | Adversarial vulnerability and robustness of DL models for panoramic dental X-ray segmentation, *Scientific Reports*, 2026 | Attacked dental panoramic X-ray segmentation under minimal perturbation; **IoU fell from 0.851 to 0.649** | Static 2D radiograph, single structure type; no instrument-vs-anatomy contrast; not intra-operative | Gives us the comparable 2026 degradation magnitude to benchmark our ΔmIoU against. Our domain adds motion blur, specular highlights and thin moving metallic instruments |
| P16 | Adversarial Robustness of DL-Based Thyroid Nodule Segmentation in Ultrasound, 2026 (arXiv:2602.21452) | U-Net nodule segmentation under adversarial perturbation, DSC-based reporting, 192 nodules | Ultrasound diagnostics; one foreground class; small cohort; no class-type analysis | Same experimental shape, but our task has 12 foreground classes, letting us compare anatomy vs instrument vulnerability inside a single frame |

---

## Block 4 — Robustness in surgical AI (the closest prior work, and where it stops)

| # | Paper (Year) | What it did | Limitation / Gap | How our project differs |
|---|---|---|---|---|
| P17 | Khan, Nawaz, Sheikh, Hanif, Yaqub, **GUARDIAN**, MICCAI 2024 workshop | Addresses uncertainty and adversarial risk in robot-assisted surgery; reports improved reliability of **surgical tool detection** under adversarial conditions, with an action-triplet score rising from 8% to 23.58% *(verify: dataset, attack list, ε values, metrics — OpenReview and SpringerLink both block automated access)* | Robot-assisted/laparoscopic setting; detection and triplet recognition, not pixel-level segmentation; the contribution is a defensive reliability mechanism, so the clean-vs-attacked vulnerability profile of a standard segmentation baseline is never characterised. Not cataract, not Cataract-1K | **The closest published work — and it is on the other side of the problem.** Version 1 stays diagnostic: we measure vulnerability and add no defence. We use segmentation, and we use cataract surgery |
| P18 | Cheng et al., Why Deep Surgical Models Fail?, 2022 (arXiv:2209.08647) | Revisited surgical action-triplet recognition through robustness and explainability using an adversarial optimisation scheme under weak and strong perturbations; self-described first study of its kind in surgical AI | Recognition/classification of triplets on laparoscopic data; no dense prediction; no standard L∞ ε-budget sweep | Establishes the precedent that adversarial perturbation is a legitimate diagnostic lens in surgical AI. We point that lens at cataract instrument segmentation with a standard, reproducible ε protocol |
| P19 | **SegSTRONG-C**, EndoVis'24 / MICCAI 2024 (arXiv:2407.11906) | Community benchmark and challenge for robust surgical tool segmentation under **non-adversarial** generated corruptions: smoke, bleeding, low brightness | The name states the limitation. Corruptions are natural or simulated, not worst-case and not gradient-derived; endoscopic rather than ophthalmic | **Our single best gap anchor.** The surgical community's flagship tool-segmentation robustness benchmark explicitly excludes the adversarial case — which is precisely what we measure |
| P20 | Adversarial Attacks on Learned Policies for Surgical Robotic Tasks, 2026 (arXiv:2606.11535) | Disruptive vs steering attacks on learned surgical robot policies; 560 physical phantom experiments (debridement, suturing); ≈61% average reduction in task success | Attacks the control policy and state estimate, not the perception model's dense output; robotic manipulation, not ophthalmic video | Strongest recent evidence that adversarial threat in surgery is treated seriously. We address the perception stage that such policies consume |

---

## Block 5 — Ophthalmic adversarial work, and one design warning

| # | Paper (Year) | What it did | Limitation / Gap | How our project differs |
|---|---|---|---|---|
| P21 | Deceptive Tricks in AI: Adversarial Attacks in Ophthalmology, *J. Clin. Med.* 12(9):3266, 2023 | Reviewed adversarial attacks across ophthalmic AI (fundus, OCT, disc photography) and states that attack algorithms for the various ophthalmic image types "have yet to be developed" | Diagnostic still images and classification/grading throughout; no intra-operative video and no instrument segmentation. The same holds for every ophthalmic adversarial paper we found (JAMA Ophthalmol. 2020; diabetic-retinopathy and glaucoma studies) | We move ophthalmic adversarial analysis out of the diagnostic clinic and into the operating theatre's video stream — the one ophthalmic setting the review's own scope never reaches |
| P22 | Region-Guided Attack on the Segment Anything Model, 2024 (arXiv:2411.02974) | Proposed a region-guided adversarial attack for SAM, and reports that **FGSM and PGD are often ineffective** against it | Natural images; an attack-design paper, not a medical robustness study | Decides our model choice. We use a conventional encoder–decoder so that a small measured degradation cannot be an artefact of attacking a prompt-driven foundation model with the wrong attack |

---

## Coverage against the spec's §20 targets

| Category | Target | Papers here | Status |
|---|---|---|---|
| Cataract / surgical instrument AI | 5–8 | P1–P5 | met |
| Adversarial attacks in medical imaging | 5–8 | P12–P16 (+ P21) | met |
| Adversarial attacks / robustness in surgical AI | 3–5 | P17–P20 | met |
| Directly relevant to the chosen model/task | 2–4 | P6–P11, P22 | met |

---

## What the 22 rows collectively prove

Read the *Limitation* column top to bottom and one shape emerges. Adversarial robustness of medical
segmentation is established on **radiology volumes, dental radiographs and ultrasound** (P13–P16).
Robustness of **surgical** tool segmentation is established, but only against **non-adversarial
corruptions** (P19). Adversarial work **inside** surgery exists, but on **detection, triplets and
robot policies** (P17, P18, P20). Ophthalmic adversarial work exists, but only on **diagnostic
imaging** (P21). Cataract instrument segmentation itself is a healthy, active field with a public
dataset, published folds and a 2025 state of the art — and not one of those papers perturbs its input
(P1–P5).

Every neighbouring cell of the grid is occupied. The cell "gradient-based adversarial attack ×
cataract surgical instrument segmentation" is empty.

---

## The four questions a mentor will ask

**"Hasn't someone already attacked surgical segmentation models?"**
Not adversarially. P19 (SegSTRONG-C) is the flagship surgical tool-segmentation robustness benchmark
and it is non-adversarial by construction. P17 (GUARDIAN) is adversarial but is detection on
robot-assisted surgery, and is a defence paper rather than a vulnerability measurement.

**"Isn't this just FGSM on a new dataset?"**
No, for three reasons. First, the attack is segmentation-aware: SegPGD (P10) rather than plain PGD,
because P7 and P11 show a pixel-averaged objective understates segmentation vulnerability. Second,
the analysis unit is the class, not the image: Cataract-1K carries 3 anatomy and 9 instrument classes
in the same frame, so we can compare anatomy vs instrument vulnerability — a contrast no paper in this
matrix can run, because their datasets have one foreground type. Third, the domain is intra-operative
video, with motion blur, specular highlights on metal and thin elongated structures.

**"Why not use the state-of-the-art model?"**
Because the state of the art on Cataract-1K is SAM2-based (P2), and P22 reports that FGSM and PGD are
often ineffective against SAM. A near-zero degradation on a SAM2 baseline would be uninterpretable —
robust model, or wrong attack? A conventional encoder–decoder from P1's own benchmark list removes
that confound and keeps the clean baseline directly comparable to published numbers.

**"What if the model turns out to be robust?"**
Then that is the result, reported as an upper bound rather than as proof, exactly as P11 requires.
P14's finding that attack success is uncorrelated with parameter count means the outcome is not
predictable in advance, which is what makes the measurement worth making.

---

## Still to verify before submission

1. **P17 GUARDIAN full text** — dataset, attack list, ε values, metrics. Highest priority; it is the
   one paper that could narrow the gap claim. Get it via the institutional Springer subscription.
2. **P1 per-baseline Dice/IoU table** — needed to state our clean baseline is in the published range.
   The tables were truncated in the reachable HTML; pull them from the *Scientific Data* PDF.
3. Re-run the gap search in Scopus / Web of Science / IEEE Xplore, and check Google Scholar
   "cited by" on P1 and P19. Any matching paper almost certainly cites one of those two.
4. Label the arXiv-only 2026 entries (P4, P5, P16, P20) as preprints in the reference list.



