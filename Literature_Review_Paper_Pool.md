# Literature Review — Paper Pool

**Project:** Adversarial Attack and Robustness Analysis of AI-Based Cataract Surgical Instrument Segmentation (Version 1)
**Compiled:** 2 September 2026
**Companion to:** [Version 1 spec](Version_1_Cataract_Adversarial_Robustness_Project.md)

Everything below came from live searches. Where the source page was reachable, details are verified against it. Anything I could not open is marked **(verify)** — do not put a "(verify)" detail into the synopsis without opening the PDF first.

---

## 0. Verified dataset facts (closes several items in §24 of the spec)

Sources: [Cataract-1K repository](https://github.com/Negin-Ghamsarian/Cataract-1K) and the [arXiv full text](https://arxiv.org/html/2312.06295v1).

| Item | Verified value |
|---|---|
| Full dataset | 1,000 cataract surgery videos, eye clinic of Klinikum Klagenfurt, 2021–2023; MediLive Trio Eye on a ZEISS OPMI Vario microscope; avg. 7.12 min; non-annotated videos 25 fps @ 512×324 |
| Segmentation subset | **2,256 annotated frames from 30 videos**, 1024×768, sampled 1 frame / 5 s; frames with harsh motion blur or an out-of-scene iris were discarded |
| Anatomy classes (3) | iris, pupil, intraocular lens |
| Instrument classes (9) | slit/incision knife, gauge, spatula, capsulorhexis cystotome, phacoemulsifier tip, irrigation-aspiration, lens injector, capsulorhexis forceps, Katena forceps |
| Official sub-tasks | anatomy+instrument segmentation · binary instrument segmentation · multi-class instrument segmentation |
| Splits | 5 patient-disjoint folds, 6 videos each: 561 / 459 / 420 / 385 / 431 frames |
| Class frequency | iris & pupil in 100% of frames; IOL 537 (23.8%); ≥1 instrument 1,778 (78.81%); rarest = Katena forceps 29 (1.29%) and slit knife 50 (2.22%); most common = spatula 728 (32.27%) |
| Annotation format | Supervisely polygons (JSON + mask scripts) **and COCO, which includes bounding boxes** |
| Authors' metrics | mean Dice + mean IoU, 512×512 input, BCE + log-Dice loss (λ = 0.8) |
| Benchmarked baselines | DeepPyramid, Adapt-Net, UNet++, ReCal-Net, CPFNet, CE-Net, FED-Net, scSENet, DeepLabV3+, UPerNet, U-Net+ (VGG16 / ResNet34 / ResNet50 backbones, 22–60M params) |
| Synapse IDs | full `syn53404507` (89.9 GB) · phase `syn53395146` (3.87 GB) · **segmentation `syn53395479` (4.74 GB)** · lens irregularity `syn53395131` · pupil reaction `syn53395402` |
| License | CC BY 4.0 (data), MIT (repository code) |

**Two decisions this settles:**

1. **Segmentation is the defensible primary task.** The class list, patient-disjoint folds, and Dice/IoU baselines are all published for it. Retitle to "…Instrument Segmentation" (spec §10, Option A).
2. **A detection arm needs no extra annotation** — COCO bounding boxes ship with the same 2,256 frames. Keep detection as optional scope, not core.

---

## A. Cataract / ophthalmic surgical AI — the baseline task

**[A1] Ghamsarian et al., "Cataract-1K Dataset for Deep-Learning-Assisted Analysis of Cataract Surgery Videos", *Scientific Data* (2024).**
[Nature](https://www.nature.com/articles/s41597-024-03193-4) · [arXiv:2312.06295](https://arxiv.org/abs/2312.06295) · [code/data](https://github.com/Negin-Ghamsarian/Cataract-1K)
Your primary citation. Dataset chapter, baseline shortlist, clean-performance reference. (verify: volume/article number)

**[A2] Cataract-LMM: Large-Scale, Multi-Source, Multi-Task Benchmark, *Scientific Data* (2026).**
[Nature](https://www.nature.com/articles/s41597-026-07464-0) · [arXiv:2510.16371](https://arxiv.org/abs/2510.16371)
3,000 phacoemulsification videos, multi-centre, four annotation layers (phases, instance segmentation of instruments + anatomy, instrument–tissue interaction, skill scores). Use for: "latest dataset" framing, and as evidence the field is still building *clean* benchmarks rather than robustness ones.

**[A3] RP-SAM2: Refining Point Prompts for Stable Surgical Instrument Segmentation (2025).**
[arXiv:2504.07117](https://arxiv.org/abs/2504.07117)
Evaluated **on Cataract-1K**: +2% mDSC, −21.36% mHD95, reduced variance across random single-point prompts. State of the art on your exact dataset, and the clearest evidence that recent cataract work optimises accuracy/stability and never adversarial robustness.

**[A4] CataractSAM-2: A Domain-Adapted Model for Anterior Segment Surgery Segmentation (2026).**
[arXiv:2603.21566](https://arxiv.org/abs/2603.21566)
Domain-adapted SAM 2 for real-time cataract surgery segmentation. Candidate baseline if you want a foundation-model arm — but read [F2] first.

**[A5] Sinha, Balazia, Bremond, "Identifying Surgical Instruments in Pedagogical Cataract Surgery Videos through an Optimized Aggregation Network", IPAS 2025.**
[arXiv:2501.02618](https://arxiv.org/abs/2501.02618) · [HAL PDF](https://inria.hal.science/hal-04864972/file/2025_IPAS.pdf)
Real-time instrument identification in cataract video. The detection side of your §10 decision.

**[A6] Data-Efficient Surgical Phase Segmentation in Small-Incision Cataract Surgery (2026).**
[arXiv:2604.10514](https://arxiv.org/abs/2604.10514) — cataract surgical AI is an active 2026 topic; label efficiency, not robustness.

**[A7] Earlier cataract segmentation benchmarks.**
[2020 CATARACTS Semantic Segmentation Challenge, arXiv:2110.10965](https://arxiv.org/abs/2110.10965) · [cataract semantic segmentation dataset, arXiv:1906.11586](https://arxiv.org/abs/1906.11586) (verify authors/venue). Dataset-history paragraph.

**[A8] Cross-Dataset Adaptation for Instrument Classification in Cataract Surgery Videos.**
[arXiv:2308.04035](https://arxiv.org/abs/2308.04035) — domain shift across cataract datasets. Use to separate *distribution shift* from *adversarial perturbation* in your introduction.

**[A9] PhacoTrainer: Deep Learning for Cataract Surgical Videos to Track Surgical Tools (2023).**
[PMC10050900](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050900/) — clinical motivation: skill metrics built on top of tool localisation.

**[A10] Benchmarking CNN- and Transformer-Based Models for Surgical Instrument Segmentation (2026).**
[arXiv:2604.09151](https://arxiv.org/abs/2604.09151) — UNet / Attention UNet / DeepLabV3 vs SegFormer head-to-head. Use to justify your architecture choice with a recent comparison.

**[A11] Rethinking Text-Promptable Surgical Instrument Segmentation with Robust Framework.**
[arXiv:2411.12199](https://arxiv.org/abs/2411.12199) — note that "robust" in this literature means false-positive/prompt robustness, *not* adversarial.

---

## B. Adversarial attack foundations, and attacks on segmentation / detection

**[B1] Goodfellow, Shlens, Szegedy, "Explaining and Harnessing Adversarial Examples", ICLR 2015.**
[arXiv:1412.6572](https://arxiv.org/abs/1412.6572) — FGSM. Canonical definition citation.

**[B2] Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks", ICLR 2018.**
[arXiv:1706.06083](https://arxiv.org/abs/1706.06083) — PGD. Canonical definition citation.

**[B3] Xie et al., "Adversarial Examples for Semantic Segmentation and Object Detection", ICCV 2017.**
[arXiv:1703.08603](https://arxiv.org/abs/1703.08603) — Dense Adversary Generation (DAG); the first attack designed for dense prediction *and* detection. Cite when you explain why classification attacks do not transfer directly to your task.

**[B4] Arnab, Miksik, Torr, "On the Robustness of Semantic Segmentation Models to Adversarial Attacks", CVPR 2018.**
[arXiv:1711.09856](https://arxiv.org/abs/1711.09856) · [CVF PDF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Arnab_On_the_Robustness_CVPR_2018_paper.pdf) · [code](https://github.com/anuragarnab/adversarial-attacks)
The first systematic robustness study of segmentation networks. Your methodology ancestor.

**[B5] Gu et al., "SegPGD: An Effective and Efficient Adversarial Attack for Evaluating and Boosting Segmentation Robustness", ECCV 2022.**
[arXiv:2207.12391](https://arxiv.org/abs/2207.12391)
Segmentation-specific PGD that reweights correctly vs. incorrectly classified pixels; stronger than plain PGD at equal budget. **Recommended as your main iterative attack instead of vanilla PGD.**

**[B6] Croce, Singh, Hein, "Robust Semantic Segmentation: Strong Adversarial Attacks and Fast Training of Robust Models" (SEA).**
[arXiv:2306.12941](https://arxiv.org/abs/2306.12941)
Shows existing attacks *severely overestimate* segmentation robustness. Cite this in your limitations: a weak attack is not evidence of a robust model.

**[B7] Evaluating the Adversarial Robustness of Semantic Segmentation (2024).**
[arXiv:2407.09150](https://arxiv.org/abs/2407.09150) — evaluation methodology for segmentation robustness. (verify authors/venue)

**[B8] Transferability / black-box, optional.**
[TranSegPGD, arXiv:2312.02207](https://arxiv.org/abs/2312.02207) · [Transferable Attack for Semantic Segmentation, arXiv:2307.16572](https://arxiv.org/abs/2307.16572) — only if you add a black-box arm (better as Version 2).

**[B9] A Survey and Evaluation of Adversarial Attacks in Object Detection.**
[arXiv:2408.01934](https://arxiv.org/abs/2408.01934) — detection-side related work if you keep the detection arm.

**[B10] Winter et al., "Benchmarking Adversarial Robustness and Adversarial Training Strategies for Object Detection" (Feb 2026).**
[arXiv:2602.16494](https://arxiv.org/abs/2602.16494)
Argues detection robustness evaluation is not standardised; proposes metrics that **separate localisation error from classification error**; finds current attacks transfer poorly to transformer detectors. Use for evaluation-protocol design plus a 2026 citation.

**[B11] Nesti et al., "Evaluating the Robustness of Semantic Segmentation … Against Real-World Adversarial Patch Attacks", WACV 2022.**
[CVF PDF](https://www.openaccess.thecvf.com/content/WACV2022/papers/Nesti_Evaluating_the_Robustness_of_Semantic_Segmentation_for_Autonomous_Driving_Against_WACV_2022_paper.pdf) — for the digital-vs-physical threat-model paragraph.

---

## C. Adversarial attacks and robustness in medical imaging

**[C1] Finlayson et al., "Adversarial attacks against medical deep learning systems" (2018).**
[arXiv:1804.05296](https://arxiv.org/abs/1804.05296) — urges caution in clinical deployment. Opening motivation citation.

**[C2] Ma et al., "Understanding Adversarial Attacks on Deep Learning Based Medical Image Analysis Systems", *Pattern Recognition* (2021).**
[arXiv:1907.10456](https://arxiv.org/abs/1907.10456)
Medical models are **more** vulnerable than natural-image models, yet their adversarial examples are **easier to detect**. Essential motivation citation; also pre-empts the "why not just detect the attack" question.

**[C3] Bortsova et al., "Adversarial attack vulnerability of medical image analysis systems: Unexplored factors", *Medical Image Analysis* 73:102141 (2021).**
[PubMed 34246850](https://pubmed.ncbi.nlm.nih.gov/34246850/) · [full text](https://pure.uva.nl/ws/files/67039419/1_s2.0_S1361841521001870_main.pdf)

**[C4] Dong et al., "Survey on Adversarial Attack and Defense for Medical Image Analysis: Methods and Benchmarks".**
[arXiv:2303.14133](https://arxiv.org/abs/2303.14133) — the standard survey. Use for taxonomy and for the benchmark protocol.

**[C5] A Comprehensive Review and Analysis of Deep Learning-Based Medical Image Adversarial Attack and Defense, *Mathematics* 11(20):4272 (2023).**
[MDPI](https://www.mdpi.com/2227-7390/11/20/4272)

**[C6] Adversarial Robustness of Deep Learning in Medical Imaging: A Comprehensive Survey and Benchmark of State-of-the-Art Architectures, *IJACSA* 16(12) (2025).**
[PDF](https://www.thesai.org/Downloads/Volume16No12/Paper_78-Adversarial_Robustness_of_Deep_Learning_in_Medical_Imaging.pdf)

**[C7] Malik, Saeed, Hanif, Naseer, Yaqub, S. Khan, F. S. Khan, "On Evaluating Adversarial Robustness of Volumetric Medical Segmentation Models", BMVC 2024.**
[arXiv:2406.08486](https://arxiv.org/abs/2406.08486)
**The closest methodological template to your project.** Pixel- and frequency-domain attacks, white-box and transfer black-box, four datasets, CNN vs transformer vs Mamba. Findings: both attack families work white-box, frequency attacks transfer far better; transformers > CNNs, Mamba weakest; large-scale training helps. Model your experimental design on this paper and cite it as the "same question, different modality" work.

**[C8] Hanif et al., "Frequency Domain Adversarial Training for Robust Volumetric Medical Segmentation", MICCAI 2023.**
[arXiv:2307.07269](https://arxiv.org/abs/2307.07269) · [code (VAFA)](https://github.com/asif-hanif/vafa)
Defense paper, but the VAFA *attack* is reusable and the code is public. Version 2 material.

**[C9] Susceptibility of Adversarial Attack on Medical Image Segmentation Models (2024).**
[arXiv:2401.11224](https://arxiv.org/abs/2401.11224)
FGSM on several medical segmentation models; finds **negligible correlation between parameter count and attack success**. Closest published experiment in spirit to your Version 1. (verify venue)

**[C10] Adversarial Robustness Analysis of Vision-Language Models in Medical Image Segmentation (2025).**
[arXiv:2505.02971](https://arxiv.org/abs/2505.02971) — FGSM + PGD on adapter-finetuned VLSMs. Note the title pattern; yours will read very similarly.

**[C11] Adversarial vulnerability and robustness of deep learning models for panoramic dental X-ray segmentation, *Scientific Reports* (2026).**
[PDF](https://www.nature.com/articles/s41598-026-57108-x.pdf)
Reports IoU collapsing from 0.851 → 0.649 under minimal perturbation. **Your best directly comparable 2026 degradation number.**

**[C12] Adversarial Robustness of Deep Learning-Based Thyroid Nodule Segmentation in Ultrasound (2026).**
[arXiv:2602.21452](https://arxiv.org/abs/2602.21452) — U-Net, DSC-based, 192 nodules. Same experimental shape as yours, different organ.

---

**[C13] Slice-sparse: few-slice adversarial attacks for 3D medical segmentation, *Int. J. Mach. Learn. Cybern.* (2026).**
[Springer](https://link.springer.com/article/10.1007/s13042-026-03265-4)

**[C14] Resilience of UNet-Based Models Under Adversarial Conditions in Medical Image Segmentation, *Eng. Proc.* 104(1):3.**
[MDPI](https://www.mdpi.com/2673-4591/104/1/3) — UNet vs Att-UNet++ under adversarial modification. Short, easy to read, directly analogous.

**[C15] Adversarial Attacks on Medical Segmentation Model via Transformation of Feature Statistics, *Appl. Sci.* 14(6):2576 (2024).**
[MDPI PDF](https://www.mdpi.com/2076-3417/14/6/2576/pdf)

**[C16] Attacking medical images with minimal noise: exploiting vulnerabilities in medical deep-learning systems (2024).**
[PMC11651935](https://pmc.ncbi.nlm.nih.gov/articles/PMC11651935/) — useful for the perturbation-magnitude vs. degradation trade-off (your RQ 5).

**[C17] Adversarial robustness of a U-Net-based model observer for CT protocol optimization (2026).**
[arXiv:2606.30115](https://arxiv.org/abs/2606.30115) — fast gradient perturbations reach up to 75% misclassification.

**[C18] MedFM-Robust: Benchmarking Robustness of Medical Foundation Models (2026).**
[arXiv:2605.19027](https://arxiv.org/abs/2605.19027) — medical-specific perturbations disproportionately damage *segmentation*; LoRA-tuned visual grounding drops >40 points. Good 2026 anchor for "robustness of medical models is still underexplored".

**[C19] BREPS: Bounding-Box Robustness Evaluation of Promptable Segmentation (2026).**
[arXiv:2601.15123](https://arxiv.org/abs/2601.15123) — adversarial *prompts* rather than adversarial pixels. Relevant only if you use a promptable model.

**[C20] Unmasking adversarial attacks using a robust XAI-driven approach for secure medical image classification, *Scientific Reports* (2026).**
[Nature](https://www.nature.com/articles/s41598-026-61357-1) — detection/defense side. Park for Version 2.

---

## D. Robustness in surgical AI — closest prior work, and where the gap lives

**[D1] Khan, Nawaz, Sheikh, Hanif, Yaqub (MBZUAI), "GUARDIAN: Guarding Against Uncertainty and Adversarial Risks in Robot-Assisted Surgeries", MICCAI 2024 workshop.**
[Springer chapter](https://link.springer.com/chapter/10.1007/978-3-031-73158-7_6) · [OpenReview kW9StEs1a5](https://openreview.net/forum?id=kW9StEs1a5)
Claims resilience against a wide spectrum of adversarial attacks and improved reliability of **surgical tool detection** under adversarial conditions; reports a triplet (instrument, verb, target) improvement from 8% to 23.58%. **The closest published work to your topic — read it first.** (verify: exact dataset, attack list, ε values, metrics. Both OpenReview and SpringerLink block automated access; get the PDF through your institution.)

**[D2] Cheng, Liu, Wang, Jin, Schönlieb, Aviles-Rivero, "Why Deep Surgical Models Fail?: Revisiting Surgical Action Triplet Recognition through the Lens of Robustness" (2022/2023).**
[arXiv:2209.08647](https://arxiv.org/abs/2209.08647)
Self-described *first* study of surgical model failure through robustness and explainability; applies an adversarial optimisation scheme under weak and strong perturbations. Precedent that adversarial perturbation is a legitimate analysis tool in surgical AI — but it is recognition, not segmentation.

**[D3] SegSTRONG-C: "Segmenting Surgical Tools Robustly On Non-adversarial Generated Corruptions" — EndoVis'24 / MICCAI challenge.**
[arXiv:2407.11906](https://arxiv.org/abs/2407.11906) · [challenge site](https://segstrongc.cs.jhu.edu/)
**Your single best gap citation.** The surgical community's flagship tool-segmentation robustness benchmark is explicitly *non-adversarial* (smoke, bleeding, low brightness).

**[D4] Adversarial Attacks on Learned Policies for Surgical Robotic Tasks (2026).**
[arXiv:2606.11535](https://arxiv.org/abs/2606.11535)
Disruptive vs. steering attacks; 560 physical phantom experiments (debridement, suturing); ~61% average reduction in surgical success. Strongest recent evidence that adversarial threats in surgery are taken seriously.

---

**[D5] Prompt injection attacks on vision-language models for surgical decision support (2026).**
[Nature-family journal](https://www.nature.com/articles/s44484-026-00014-6) · [PubMed 40778151](https://pubmed.ncbi.nlm.nih.gov/40778151/)
Laparoscopic-video VLMs misled by bidirectional and covert prompt injection. Use for the "security of surgical AI" framing. (verify exact journal title)

**[D6] Performance and Non-adversarial Robustness of SAM 2 in Surgical Video Segmentation (2024).**
[arXiv:2408.04098](https://arxiv.org/abs/2408.04098) — again, explicitly non-adversarial. Reinforces [D3].

**[D7] From Generalization to Precision: Exploring SAM for Tool Segmentation in Surgical Environments (2024).**
[arXiv:2402.17972](https://arxiv.org/abs/2402.17972) — SAM over-segments badly at high corruption levels.

**[D8] Towards Robust Algorithms for Surgical Phase Recognition via Digital Twin Representation.**
[arXiv:2410.20026](https://arxiv.org/abs/2410.20026) — robustness via intermediate representations.

**[D9] On the Robustness of Temporal Vision-Language Models for Surgical Endoscopy Videos (2026).**
[arXiv:2608.14262](https://arxiv.org/abs/2608.14262) — introduces Endo-C6, six endoscopy-realistic corruptions. Most recent surgical robustness benchmark found (Aug 2026), still corruption-based.

**[D10] OccSAM-Bench: A Controlled Benchmark for Occlusion Robustness in Foundation Segmentation Models (2026).**
[arXiv:2604.11711](https://arxiv.org/abs/2604.11711) — synthesised surgical occlusion.

**[D11] Overcoming color domain shift in endoscopic spine surgery: robust instrument segmentation via aggressive photometric augmentation, *Scientific Reports* (2026).**
[Nature](https://www.nature.com/articles/s41598-026-61273-4)

**[D12] Risk and liability in the deployment of AI systems for surgery: a SAGES white paper (2026).**
[Springer](https://link.springer.com/article/10.1007/s00464-026-12881-8) — governance framing for your limitations chapter, and support for *not* making clinical safety claims.

---

## E. Ophthalmology-specific adversarial work — bridges your two halves

**[E1] Outcomes of Adversarial Attacks on Deep Learning Models for Ophthalmology Imaging Domains, *JAMA Ophthalmology* 138(11):1213–1215 (2020).**
[PMC7530819](https://pmc.ncbi.nlm.nih.gov/articles/PMC7530819/)

**[E2] Deceptive Tricks in Artificial Intelligence: Adversarial Attacks in Ophthalmology, *J. Clin. Med.* 12(9):3266 (2023).**
[MDPI](https://www.mdpi.com/2077-0383/12/9/3266)
States that attack algorithms for the various ophthalmic image types "have yet to be developed". **Quotable support for your gap.**

**[E3] Universal Adversarial Framework to Improve Adversarial Robustness for Diabetic Retinopathy Detection.**
[arXiv:2312.08193](https://arxiv.org/abs/2312.08193)

**[E4] Analyzing CNN-Based Glaucoma Decision Criteria Using Adversarial Examples, *Technologies* 14(7):435 (2026).**
[MDPI](https://www.mdpi.com/2227-7080/14/7/435) — uses adversarial perturbation as an *analysis* tool (which region matters most). A nice model for your per-class vulnerability analysis.

**[E5] Adversarial exposure attack on diabetic retinopathy imagery.**
[arXiv:2009.09231](https://arxiv.org/abs/2009.09231) — non-Lp, physically plausible perturbation.

**[E6] Adversarial Attacks vs. Data Augmentation Strategies … Diabetic Retinopathy Detection (2026).**
[Springer](https://link.springer.com/chapter/10.1007/978-3-032-35393-1_15)

**[E7] Evaluating Deep Learning Resilience in Retinal Fundus Classification with GAN-Generated Images, *Electronics* 13(13):2631 (2024).**
[MDPI](https://www.mdpi.com/2079-9292/13/13/2631)

> Note the pattern: **every** ophthalmic adversarial paper above is diagnostic imaging (fundus, OCT, disc photos). None is intra-operative surgical video. That contrast is the spine of your research gap.

---

## F. SAM / foundation-model attacks — read only if you pick a SAM2-based baseline

**[F1] On the Robustness of Segment Anything.**
[arXiv:2305.16220](https://arxiv.org/abs/2305.16220) — robust to most corruptions except blur; still susceptible to adversarial perturbation.

**[F2] Region-Guided Attack on the Segment Anything Model.**
[arXiv:2411.02974](https://arxiv.org/abs/2411.02974)
**Important warning:** reports that FGSM and PGD are *often ineffective* against SAM. If you choose RP-SAM2 [A3] or CataractSAM-2 [A4] as your baseline, a weak FGSM result would make you wrongly conclude the model is robust. Either pick a conventional encoder–decoder baseline, or use a region/prompt-aware attack.

**[F3] Transferable Adversarial Attacks on SAM and Its Downstream Models.**
[arXiv:2410.20197](https://arxiv.org/abs/2410.20197)

**[F4] On the Adversarial Robustness of Vision Foundation Models (2025).**
[arXiv:2504.08906](https://arxiv.org/abs/2504.08906) — SAM falls to 0.26 mIoU under attack.

---

## G. Research gap — what the searches actually support

Search combinations run: `Cataract-1K adversarial` · `adversarial cataract surgery instrument segmentation` · `adversarial examples surgical instrument segmentation FGSM PGD` · `adversarial ophthalmic surgery vulnerability` · `adversarial attack cataract instrument segmentation 2026`, plus the four category sweeps in §19 of the spec.

**Result: no study applying gradient-based adversarial attacks to cataract surgical scene/instrument segmentation surfaced, and no robustness study using Cataract-1K surfaced.**

What does exist, and where it stops:

| Neighbouring area | Representative work | Why it is not your project |
|---|---|---|
| Adversarial robustness of medical segmentation | [C7] [C9] [C10] [C11] [C12] [C14] | Radiology volumes, dental X-ray, ultrasound, VLSMs — not surgical video |
| Surgical robustness benchmarks | [D3] [D6] [D7] [D9] [D10] | Non-adversarial corruptions (smoke, bleeding, blur, occlusion) only |
| Adversarial work inside surgery | [D1] [D2] [D4] [D5] | Tool detection/triplets, robotic policies, VLM prompts — laparoscopic/robotic, not cataract segmentation |
| Ophthalmic adversarial work | [E1]–[E7] | Diagnostic imaging only, never intra-operative video |

Defensible gap statement for the synopsis:

> Adversarial robustness in medical image segmentation has been studied predominantly on radiological volumes and 2-D diagnostic images, while robustness research in surgical computer vision has concentrated on non-adversarial corruptions and on laparoscopic or robot-assisted domains. Ophthalmic adversarial studies address diagnostic imaging rather than intra-operative video. The adversarial robustness of cataract surgical scene and instrument segmentation models — the setting represented by the Cataract-1K dataset — therefore remains unquantified.

**Caveat to respect.** Absence of search results is weaker evidence than a systematic review. Before submission, re-run in Scopus / Web of Science / IEEE Xplore and use Google Scholar "cited by" on [A1] Cataract-1K and [D3] SegSTRONG-C. If a matching paper exists, it almost certainly cites one of those two.

---

## H. Core must-cite set (if you only read twelve)

| # | Paper | Role in the synopsis |
|---|---|---|
| [A1] | Cataract-1K | Dataset and task definition |
| [A3] | RP-SAM2 | State of the art on your dataset |
| [C2] | Ma et al. | Medical models are more vulnerable than natural-image models |
| [C7] | BMVC 2024 volumetric robustness | Experimental-design template |
| [C9] | Susceptibility of medical segmentation to FGSM | Closest prior experiment |
| [C11] | Dental X-ray 2026 | Comparable degradation figures |
| [B1] | FGSM | Attack definition |
| [B2] | PGD | Attack definition |
| [B5] | SegPGD | The correct strong attack for segmentation |
| [B6] | SEA / Croce et al. | Honest robustness reporting |
| [D1] | GUARDIAN | Closest surgical prior work |
| [D3] | SegSTRONG-C | Gap anchor |

---

## I. Methodology notes to carry into the synopsis

1. **Use SegPGD [B5] as the primary iterative attack, FGSM [B1] as the single-step reference.** Plain PGD on a pixel-wise cross-entropy loss under-attacks segmentation models; SEA [B6] shows how badly that inflates apparent robustness.
2. **Report per-class IoU, not only mIoU.** Katena forceps (1.29% of frames) and slit knife (2.22%) are rare — mIoU will hide their collapse, and per-class vulnerability is your RQ 3.
3. **Fix the pixel space.** Perturb in [0,1] *before* normalisation, and report ε as x/255. Sweep ε ∈ {1, 2, 4, 8}/255, which is the range the medical segmentation papers above use; anything larger stops being "visually subtle".
4. **Attack anatomy classes as well as instruments.** Iris and pupil appear in 100% of frames, IOL in 23.8%. An instrument-vs-anatomy degradation comparison is cheap to run and, as far as these searches show, unreported anywhere in your reference set.
5. **Keep one threat model: white-box, untargeted.** Transfer and black-box belong to Version 2 ([B8], [C7]).
6. **Use the published 5-fold patient-disjoint split.** It makes your clean baseline directly comparable to [A1]'s Dice/IoU tables instead of a private split nobody can check.

---

## J. Open items / not verified

- **[D1] GUARDIAN** — dataset, attack list, ε values and metrics unconfirmed; OpenReview and SpringerLink both block automated fetch. Highest-priority PDF to obtain.
- **[A1] Cataract-1K per-baseline Dice/IoU values** — the tables were truncated in the HTML I could read. Pull them from the *Scientific Data* PDF before writing your clean-baseline comparison.
- Volume / article numbers for [A1] and [A2]; exact journal title for [D5].
- Venues for the arXiv-only 2026 entries ([B10], [C12], [C13], [C17], [C18], [C19], [D4], [D9], [D10], [A4], [A6], [A10]) — several are preprints and should be labelled as such in the reference list.
- Authors/venue for [B7] and [A7]'s 1906.11586 entry.

---

## K. Suggested next actions

1. Download the segmentation subset: Synapse `syn53395479` (4.74 GB).
2. Obtain the four PDFs that decide your design: [A1], [C7], [C9], [D1].
3. Fill the §20 literature table in the spec using the categories above.
4. Lock the title to "…Instrument Segmentation" and update §2, §10 and §26 of the spec.
5. Pick the baseline from [A1]'s benchmark list (DeepLabV3+ or UNet++ are the safest for adversarial work — well-understood gradients, public implementations) and avoid a SAM-based baseline unless you also implement a region-aware attack ([F2]).







