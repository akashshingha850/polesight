# Fusion Architectures for Pixel-Registered Multimodal Images: Early vs Mid vs Late Fusion, SOTA Modules, and Relevance to Two-Channel LiDAR Instance Segmentation

Scope note: all numbers below were read from the primary paper text or table (PDF/HTML) during this session unless marked "(secondary)" (search snippet or LLM page summary that I could not cross-check against the raw table). "Controlled" means same backbone/detector and data, with only the fusion stage or module changed. PoleSight context (target task, not researched here): 870 training images, two single-channel 1024x128 renderings (intensity, range) from one LiDAR on one pixel grid, 4 pole classes, tiny masks (median 71 px²), Ultralytics YOLO-seg baselines.

## Q1. Which papers run controlled early/mid/late fusion comparisons for detection or instance segmentation, and when did late fusion match or beat learned fusion?

### Takeaway
Across the controlled studies I found, mid (feature-level) fusion is usually the best single learned fusion stage on RGB-thermal detection (KAIST, FLIR, M3FD). Naive early fusion (channel concatenation) is unstable: sometimes it is worse than the best single modality, sometimes about equal. Two exceptions matter for PoleSight. (1) Non-learned late fusion of well-tuned single-modal detectors (ProbEn, and even plain NMS) beat mid fusion on both KAIST and FLIR in Chen et al. (ECCV 2022). (2) On a small dataset (VEDAI, about 1.1k training images) with a small YOLO, pixel-level early fusion beat every feature-level fusion point tried (SuperYOLO, TGRS 2023). For instance segmentation, the only controlled stage comparison I found (SF Mask R-CNN on WISDOM) had early fusion worst, late fusion in the middle, and attention ("confidence") fusion best.

### Cited Findings

**Chen et al., "Multimodal Object Detection via Probabilistic Ensembling" (ProbEn), ECCV 2022: controlled early vs mid vs late, Faster R-CNN (Detectron2, COCO-pretrained)**
- KAIST (aligned RGB-T, sanitized train set of 7,601 images, LAMR↓, Table 1): RGB 18.67, Thermal 18.99, EarlyFusion 19.36, MidFusion 14.48. Late fusion: NMS (max score + argmax box) 10.78; average-score fusion 19.4–19.5 ("double counts class prior"); ProbEn (RGB+Thermal) 8.50; ProbEn3 (RGB+Thermal+MidFusion) 7.66. Early fusion was *worse* than either single modality on "All". Plain NMS late fusion already beat Early and MidFusion. — [ECCV 2022 paper PDF, Table 1](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690139.pdf); [arXiv 2104.02904](https://arxiv.org/abs/2104.02904)
- KAIST benchmark (Table 2): ProbEn3 with off-the-shelf mid-fusion detectors improves the prior SOTA GAFF from 6.48 to 5.14 LAMR (ProbEn3 w/ GAFF), i.e., late fusion stacked on top of a mid-fusion model still helps even though the conditional-independence assumption is violated. — [ECCV 2022 PDF, Table 2](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690139.pdf)
- FLIR (unaligned, thermal-only annotations, AP50↑, Table 3): Thermal 79.24, EarlyFusion 78.80 (below thermal-only), MidFusion 80.53. Every late-fusion variant over {Thermal, EarlyFusion, MidFusion} scored 82.65–83.76, with ProbEn3 + v-avg box fusion best at 83.76. — [ECCV 2022 PDF, Table 3](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690139.pdf)
- Learned late fusion (learning to fuse logits) was only marginally better than ProbEn (9.08 vs 9.16 LAMR, argmax box fusion). Score calibration (temperature scaling) was "crucial" when fusing detectors from different sources. — [ECCV 2022 PDF, Sec. 4.1](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690139.pdf)
- Authors' stated reasons late fusion wins: (1) it can use highly tuned single-modal detectors trained on large single-modal datasets; (2) it handles a modality missing a detection. They note NMS works "precisely because it exploits the same key insights." — [ECCV 2022 PDF, Sec. 5](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690139.pdf)
- Code: [GitHub Jamie725/Multimodal-Object-Detection-via-Probabilistic-Ensembling](https://github.com/Jamie725/Multimodal-Object-Detection-via-Probabilistic-Ensembling)

**Zhou et al., "Optimizing Multispectral Object Detection: A Bag of Tricks and Comprehensive Benchmarks," arXiv 2411.18288 (2024): pixel vs feature vs decision fusion, repeated runs (mean ± std)**
- KAIST, ResNet50 + YOLOv5 (MR↓, Table II): pixel-level fusion 29.31±1.21, feature-level 15.17±1.59, RGB-only 18.39±1.75, TIR-only 17.89±2.92. Pixel-level fusion was much worse than either single modality. — [arXiv 2411.18288 HTML, Table II](https://arxiv.org/html/2411.18288)
- FLIR, ResNet50 + YOLOv5 (mAP, Table II): pixel-level 63.53±2.81 vs feature-level 73.22±1.66. — [arXiv 2411.18288, Table II](https://arxiv.org/html/2411.18288)
- KAIST, ViT-L + RTMDet: decision-level 14.53±2.95 vs feature-level 15.11±2.65, so decision-level was slightly better here. The authors say decision-level "demonstrates instability with certain methods." — [arXiv 2411.18288, Table II](https://arxiv.org/html/2411.18288)
- Fusion modules (NIN vs ICFE, 100 repetitions, Fig. 1): which one wins depends on the backbone (NIN better with ResNet50, ICFE better with ViT-L). — [arXiv 2411.18288](https://arxiv.org/html/2411.18288)
- Caveat: the RGB-only KAIST value (18.39) is identical to the RGB-only value an LLM summary attributed to ICAFusion's Table 9 (below). It may be shared or reused. Check this against the PDFs before quoting both.

**Wan et al., YOLOv11-RGBT, arXiv 2506.14696 (2025): six fusion strategies on one YOLO framework (closest analogue to Ultralytics baselines)**
- M3FD, YOLOv11n (Table 15) (secondary: page summary, not raw table): Early 78.24 AP50 / 52.66 AP; Mid 82.02 / 55.38; Mid-to-late 80.72 / 54.83; Late 79.78 / 53.14; Score fusion 78.98 / 53.00; Weight-sharing 79.99 / 53.65. Mid fusion was best, and early fusion was the worst of the six. — [arXiv 2506.14696](https://arxiv.org/html/2506.14696)
- Fusing at a single node (P3 only) usually beat three-node mid fusion and used fewer parameters (FLIR: YOLOv11n Midfusion 38.41 AP vs Midfusion-P3 39.70 AP; YOLOv11x Midfusion 80.04M params vs Midfusion-P3 58.65M). — [arXiv 2506.14696](https://arxiv.org/html/2506.14696)
- Single-modality YOLOv11x: FLIR RGB 31.80 / IR 41.96 AP; LLVIP IR 69.93 AP vs best fusion (MCF) 70.26 AP (+0.33); M3FD RGB 63.47 / IR 61.39 AP. — [arXiv 2506.14696](https://arxiv.org/html/2506.14696)
- Authors: "mid-level fusion is superior in most scenarios"; "when multi-node mid-fusion is ineffective, single-node fusion is advantageous." Code (supports YOLOv3–v12, RT-DETR): [GitHub wandahangFY/YOLOv11-RGBT](https://github.com/wandahangFY/YOLOv11-RGBT)

**Zhang et al., SuperYOLO, IEEE TGRS 2023: VEDAI (RGB+IR aerial vehicles; 10-fold CV, 1,089 train / 121 test images per fold)**
- Fusion-stage ablation, YOLOv5s-noFocus (mAP50, Table III): pixel-level concat 69.5 (7.07M params, 20.4 GFLOPs); pixel-level MF module 70.3; feature-level concat after block 1/2/3/4: 66.0 / 68.5 / 64.8 / 63.8; multistage feature-level fusion 59.3 (7.75M, 34.6 GFLOPs). On this small dataset, early fusion beat every mid-fusion point. Authors suggest multi-stage fusion "can lead to the accumulation of redundant information." — [SuperYOLO PDF (arXiv 2209.13351), Table III](https://arxiv.org/abs/2209.13351); [author PDF](https://www.sfu.ca/~zhenman/files/J12-TGRS2023-SuperYOLO.pdf)
- Unimodal vs early-fusion "Multi" (mAP50, Table VII): YOLOv5s IR 49.94 / RGB 54.82 / Multi 56.79; YOLOv5m 52.19 / 58.80 / 60.69; YOLOv5l 54.06 / 60.81 / 62.16; YOLOv5x 54.18 / 62.09 / 62.65; YOLOv4 52.75 / 62.43 / 62.55; YOLOv3 51.54 / 61.06 / 61.26; SuperYOLO 65.60 / 72.49 / 75.09. Early fusion gains over the best single modality range from +0.1 to +2.6 mAP50. — [arXiv 2209.13351, Table VII](https://arxiv.org/abs/2209.13351)
- Resolution matters more than fusion for small objects (Tables II/IV): removing YOLOv5s's Focus downsampling raised early-fusion mAP50 from 62.2 to 69.5. A 1024 input reached 79.3, and the SR branch at 512 reached 78.0. — [arXiv 2209.13351, Table IV](https://arxiv.org/abs/2209.13351)
- Code: [GitHub icey-zhang/SuperYOLO](https://github.com/icey-zhang/SuperYOLO)

**Zhang/Xue et al., "Rethinking Early-Fusion Strategies for Improved Multispectral Object Detection" (ShaPE), arXiv 2405.16038 (2024), YOLOv5** (secondary: page summary)
- FLIR mAP50: RGB 63.77, Thermal 75.07, plain early fusion 74.77 (below thermal), two-branch mid fusion 76.07, ShaPE early fusion 75.77 (7.06M params, 15.78 GFLOPs). Authors: "the plain early-fusion strategy cannot achieve consistent improvement compared with single-modality input." They blame information interference, the RGB-T domain gap, and weak single-modality features. — [arXiv 2405.16038](https://arxiv.org/abs/2405.16038). The summary's M3FD numbers (early fusion = ShaPE = 82.90) look suspicious and should be checked against the table.
- Code (announced): github.com/XueZ-phd/Efficient-RGB-T-Early-Fusion-Detection — [arXiv 2405.16038](https://arxiv.org/abs/2405.16038)

**RSDet (Zhao et al., arXiv 2401.10731, 2024), Faster R-CNN R50, FLIR-aligned** (secondary: page summary)
- RGB-only 64.9 mAP50 / 28.9 mAP; IR-only 74.4 / 37.6; two-stream concat 73.1 / 37.1 (below IR-only); + CFT 77.5 / 39.2 (196.9M params); + CMX module 80.5 / 39.7 (305.2M); RSDet DFS 80.9 / 41.3 (68.5M). — [arXiv 2401.10731](https://arxiv.org/html/2401.10731). Code: [GitHub Zhao-Tian-yi/RSDet](https://github.com/Zhao-Tian-yi/RSDet)

**Instance segmentation (RGB-D): SF Mask R-CNN (Back et al., ICIP 2020), WISDOM, ResNet-50-FPN**
- Mask AP / Box AP: RGB-only 59.0 / 61.4; Depth-only 59.6 / 60.4; early fusion 55.5 / 57.2 (worst, below both single modalities); late fusion 58.7 / 59.0; confidence (self-attention-weighted) fusion 60.5 / 61.0. — [GitHub gist-ailab/SF-Mask-RCNN README](https://github.com/gist-ailab/SF-Mask-RCNN)

**Semantic segmentation (RGB-X), controlled stage/module comparisons**
- TokenFusion (CVPR 2022), NYUDv2 mIoU (Table 2): SegFormer-Ti RGB-only ("w/o fusion") 49.7 → feature Concat 50.8 → TokenFusion 53.3. Small: 50.6 → 51.4 → 54.2. Naive concat gave about +1 mIoU; the dedicated module gave +3.6. Random token exchange of 30% *hurt* (48.2, Table 5). Appending RGB to points also lowered 3D detection (Table 3). — [arXiv 2204.08721, Tables 2, 3, 5](https://arxiv.org/abs/2204.08721)
- CMX (T-ITS 2023), NYUv2, MiT-B2 (Table IX): no rectification + average fusion 50.3; +CM-FRM 52.8; +FFM 51.5; both 54.1 mIoU. — [arXiv 2203.04838, Table IX](https://arxiv.org/abs/2203.04838)

### Inferences
- Late fusion matched or beat learned fusion when (a) single-modal detectors were strong and separately tuned (COCO-pretrained Faster R-CNN), (b) the modalities fail in complementary situations (RGB by day, thermal by night), and (c) fusion increased scores when detections agreed (ProbEn) or at least did not lower them (NMS). Average-score fusion failed. For PoleSight, intensity and range likely fail on different pole types or backgrounds, so a cheap first experiment is ProbEn or NMS/WBF over two separately trained YOLO-seg models, with temperature calibration. Mask fusion for instance segmentation is not covered by ProbEn. Fused masks would need, e.g., IoU-matched mask averaging. That last point is my extrapolation, not from a paper.
- Plain early fusion (channel stacking) is the riskiest stage on RGB-thermal benchmarks with large domain gaps. It did best only on the small VEDAI set with a small YOLO. The intensity/range pair has no domain gap in geometry (same sensor, same grid), which removes the misalignment argument the ProbEn paper gives for mid fusion beating early fusion on FLIR.

### Gaps
- No controlled early/mid/late comparison on DroneVehicle, LLVIP or KAIST-360/DeLiVER *instance* segmentation was found. CMNeXt/DeLiVER report RGB-LiDAR semantic segmentation but no stage ablation.
- I found no paper that compares WBF specifically against learned fusion for RGB-T. ProbEn's "avg" box fusion is the closest analogue.
- C2Former, MMI-Det, WaveMamba and CFT stage ablations were not read in this session.

## Q2. SOTA fusion modules/models (2022–2026) with code: claimed gains over single modality and over simple fusion

### Takeaway
Dedicated fusion modules typically claim +2 to +9 mAP or mIoU over the best single modality on RGB-T/RGB-D benchmarks. Their gain over a *simple* two-stream concat/add baseline is smaller, usually +1 to +4. Many SOTA papers do not report the simple-early-fusion baseline at all. The parameter cost is often 2–4x a single-modal model (Fusion-Mamba 287.6M vs 76.7M for YOLOv8-l). For a ~1k-image, two-single-channel task, the most adaptable candidates are YOLO-native dual-stream code (YOLOv11-RGBT, SuperYOLO, ICAFusion, Fusion-Mamba) and segmentation encoders that take a second image modality (CMX, CMNeXt, DFormer/DFormerv2, GeminiFusion, Sigma). None of them provides instance-segmentation heads out of the box.

### Cited Findings

**Detection (RGB-IR)**
- Fusion-Mamba (arXiv 2404.09146, 2024; YOLOv5/YOLOv8 backbones, SSCS + DSSF modules) (secondary: page summary of Tables 1–3): FLIR-aligned mAP50/mAP: YOLOv8-l IR 72.9/38.3, RGB 66.3/28.2, CFT 78.7/40.2 (206M params), Fusion-Mamba-YOLOv8 84.9/47.0 (287.6M params, 78 ms). LLVIP: IR 95.2/62.1 → 97.0/64.3. M3FD: RGB 80.9/52.5 → 88.0/61.9. The page summary found no code link. — [arXiv 2404.09146](https://arxiv.org/html/2404.09146)
- ICAFusion (Pattern Recognition 2024, arXiv 2308.07504; YOLOv5 CSPDarkNet53, ~120M params) (secondary): KAIST MR 7.17 vs NIN-fusion baseline 8.33 vs RGB 18.39 / thermal 18.94; FLIR 79.2 mAP50 / 41.4 mAP; VEDAI 76.62 mAP50 vs baseline 74.66. Iterative fusion: one iteration was best, more degraded results. — [arXiv 2308.07504](https://arxiv.org/html/2308.07504); code [GitHub chanchanchan97/ICAFusion](https://github.com/chanchanchan97/ICAFusion)
- DAMSDet (ECCV 2024, arXiv 2403.00326; DETR-style, ResNet50) (secondary): M3FD 52.9 mAP vs RGB 46.3 / IR 35.0 and vs DINO with concatenated RGB+IR 43.3. FLIR 49.3 vs RGB 31.8 / IR 44.8. LLVIP 69.6 vs RGB 53.8 / IR 62.9. MCQS +1.1, MDCA +1.5 mAP50. — [arXiv 2403.00326](https://arxiv.org/html/2403.00326); code announced at github.com/gjj45/DAMSDet.
- RSDet (2024): FLIR mAP 41.3 (DFS) / 43.8 (full) vs two-stream concat 37.1 and IR-only 37.6, at 68.5M params vs CMX-module 305M. — [arXiv 2401.10731](https://arxiv.org/html/2401.10731)
- YOLOv11-RGBT MCF (ControlNet-style: freeze a strong single-modal detector, inject the second modality through a zero-initialized conv): FLIR YOLOv11x-MCF 47.61 AP vs IR-only 41.96. LLVIP 70.26 vs IR 69.93. — [arXiv 2506.14696](https://arxiv.org/html/2506.14696)
- SuperYOLO (pixel-level MF + super-resolution branch, 4.85M params): VEDAI 75.09 mAP50 vs RGB 72.49 / IR 65.60. — [arXiv 2209.13351, Table VII](https://arxiv.org/abs/2209.13351)

**Segmentation encoders (RGB-D / RGB-T / RGB-LiDAR / RGB-P)**
- CMX (T-ITS 2023), MiT-B2 vs SegFormer-B2 RGB-only (Table VIII, mIoU): NYUv2 48.0→54.1 (RGB-D); Cityscapes 81.0→81.6; MFNet 53.2→58.2 (RGB-T); ZJU-RGB-P 89.6→92.2 (RGB-P); EventScape 58.7→61.9; KITTI-360 61.3→64.3 (RGB-LiDAR). CMX-B2: 66.6M params, 67.6 GFLOPs. — [arXiv 2203.04838, Tables VIII, XII](https://arxiv.org/abs/2203.04838)
- CMNeXt (CVPR 2023, DeLiVER), MiT-B2, DeLiVER/KITTI-360 mIoU (Table 1): RGB-LiDAR: TokenFusion 54.55/53.01, CMX 64.31/56.37, CMNeXt 65.26/58.04. RGB-D-LiDAR CMNeXt 66.55/65.50; RGB-D-E-L 67.84/66.30. — [arXiv 2303.01480, Table 1](https://arxiv.org/abs/2303.01480)
- GeminiFusion (ICML 2024, pixel-wise cross-modal attention on matched tokens), Table 1: NYUDv2 MiT-B5 TokenFusion 55.1 → GeminiFusion 57.7 mIoU. DeLiVER RGB+LiDAR MiT-B2 TokenFusion 55.5 → GeminiFusion 58.6. Authors find TokenFusion's exchange "unstable" (Fig. 3: exchanging all tokens "almost invariably yields the best outcomes"). — [arXiv 2406.01210, Table 1, Fig. 3](https://arxiv.org/abs/2406.01210)
- DFormerv2 (CVPR 2025; depth used as a geometry prior inside self-attention rather than encoded by a second backbone), Table 1 NYUv2/SUN-RGBD mIoU: DFormerv2-S 26.7M/33.9G 56.0/51.5; -B 53.9M/67.2G 57.7/52.8; -L 95.5M/124.1G 58.4/53.3. Compare GeminiFusion MiT-B5 137.2M/256.1G 57.7/53.3, CMX-B5 181.1M 56.9/52.4, DFormer-T 6.0M 51.8/48.8. Table 3 (step-wise): RGB-only vanilla attention 51.7 → +depth prior 54.3 → +both priors 56.2. Table 2 DeLiVER: DFormerv2-L 67.1 vs GeminiFusion-B5 66.9. Backbones need RGB-D pretraining on ImageNet-1K (Sec. 4). — [arXiv 2504.04701, Tables 1–3](https://arxiv.org/abs/2504.04701); code [GitHub VCIP-RGBD/DFormer](https://github.com/VCIP-RGBD/DFormer)
- Sigma (WACV 2025, Siamese VMamba): MFNet mIoU Sigma-T 60.2 (48.3M) / -S 61.1 / -B 61.3 vs CMNeXt-B4 59.9 (119.6M). Without both Mamba fusion blocks 58.4 vs full 60.5 (Table 3). NYUv2 Sigma-S 57.0. — [arXiv 2404.04256 v3](https://arxiv.org/html/2404.04256v3); code [GitHub zifuwan/Sigma](https://github.com/zifuwan/Sigma)
- TokenFusion (CVPR 2022): NYUDv2 MiT-B3 54.2; see Q1 for the concat baseline. — [arXiv 2204.08721](https://arxiv.org/abs/2204.08721)

**Instance segmentation (multimodal)**
- SF Mask R-CNN (RGB-D, WISDOM): confidence fusion 60.5 mask AP vs best single modality 59.6 (+0.9). — [GitHub gist-ailab/SF-Mask-RCNN](https://github.com/gist-ailab/SF-Mask-RCNN)
- Casado-Coscolla et al., J. Imaging 2024 (see Q3): YOLOv8-seg on stacked Ouster range/reflectivity/ambient channels, mask AP50 93.7–94.9 depending on size. — [J. Imaging 10(12):325](https://pmc.ncbi.nlm.nih.gov/articles/PMC11728245/)

### Inferences
- Gains over *simple* fusion are where it counts for PoleSight, because channel stacking into Ultralytics YOLO-seg is free. Reported gaps: TokenFusion vs concat +2.5 to +2.8 mIoU (NYUv2); RSDet vs two-stream concat +4.2 mAP (FLIR); DAMSDet vs DINO concat +9.6 mAP (M3FD, a DETR setting where concat seems particularly weak); CMX FFM/FRM vs average +3.8 mIoU. All of these come from benchmarks with ≥1.4k–50k training pairs and RGB-pretrained branches.
- Adaptation cost: CMX/CMNeXt/DFormer/GeminiFusion/Sigma are semantic-segmentation encoders. For instance masks they would need a Mask2Former or Mask R-CNN head (e.g., via mmdetection), which is substantial engineering. YOLOv11-RGBT (multi-node and P3 mid fusion, MCF) and SuperYOLO are closest to the existing Ultralytics stack. These are detection repos, so seg heads would have to be ported. I did not verify whether YOLOv11-RGBT supports segmentation.

### Gaps
- I did not verify GitHub URLs for CMX, CMNeXt, GeminiFusion, TokenFusion, CFT, Fusion-Mamba or C2Former in this session.
- No 2022–2026 RGB-T or RGB-D *instance segmentation* benchmark with a SOTA fusion leaderboard was found. Searches returned mainly semantic segmentation (MFNet, PST900, NYUv2) and unseen-object instance segmentation (UOIS) works.
- WaveMamba, MMI-Det and C2Former numbers were not collected.

## Q3. Evidence specific to fusing highly correlated or same-sensor modalities (depth+HHA, multiple LiDAR channels, polarization, hyperspectral)

### Takeaway
The direct evidence is thin but points one way. When the second input is derived from the same sensor, simple input-level channel stacking already captures most of the benefit. In the closest analogue (Ouster LiDAR range + reflectivity + ambient into YOLOv8-seg), stacking the two active channels (range + reflectivity) gave the best precision. A CMX control shows that a dual-branch network gains +0.5 mIoU from a duplicate RGB input and +1.0 from pure noise. Part of any "mid-fusion gain" is therefore capacity or regularization, not complementary information. The HHA re-encoding of the *same* depth adds about +0.9 mIoU over raw depth.

### Cited Findings
- **Casado-Coscolla, Sanchez-Belenguer, Wolfart, Sequeira, "Point-Cloud Instance Segmentation for Spinning Laser Sensors," J. Imaging 10(12):325, 2024.** Ouster scans (14,364 scans, 28,198 person instances, mostly indoor). Range, reflectivity and ambient channels are stacked into the three YOLOv8-seg input channels. Each 1024x128 laser image is split into four overlapping segments and tiled into 640x640. Reflectivity and ambient are histogram-equalized; range is clamped and normalized. Ablation over all 7 channel subsets (A, D, R, A+D, A+R, D+R, A+D+R) x 5 YOLOv8 sizes, unused channels zeroed, 250 epochs each (Fig. 5). Conclusion: "the combination of the two active channels (range and reflectivity) have provided the highest precision values." Full pipeline A+D+R mask AP50 93.67% (N) to 94.94% (L), mAP50:95 79.34–82.41% (Table 3). The per-subset numbers are only in Fig. 5 (not extracted). — [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11728245/)
- **CMX supplementary-modality control (NYUv2, Table XI, no MLP decoder):** RGB 46.7; RGB+RGB 47.2; RGB+Noise 47.7; RGB+raw depth 51.1; RGB+HHA 52.0 mIoU. The authors suggest noise "may help prevent over-fitting." So dual-stream gains of roughly ≤1 mIoU can appear without any new information. — [arXiv 2203.04838, Table XI](https://arxiv.org/abs/2203.04838)
- **Polarization (CMX, ZJU-RGB-P, Table V):** RGB-AoLP and RGB-DoLP give near-identical mIoU (e.g., MiT-B2 monochromatic 91.9 vs 91.4 per a class/overall column read from the text table; full mIoU columns not cleanly extracted). Trichromatic polarization representations are "consistently better than monochromatic." Overall RGB-P gain over RGB-only SegFormer-B2 is 89.6 → 92.2 (Table VIII). — [arXiv 2203.04838, Tables V, VIII](https://arxiv.org/abs/2203.04838)
- **Depth as a prior vs a second encoder (DFormerv2, Table 7, LUSS dataset):** classification top-1 RGB 83.1, depth 43.8, RGB+D 83.4; foreground segmentation wF 0.818 / 0.715 / 0.868. Depth "mainly helps the model segment the objects and slightly helps capture semantics." — [arXiv 2504.04701, Table 7](https://arxiv.org/abs/2504.04701)
- **DS-RangeNet (Electronics 15(17):3983, 2026), lightweight dual-stream LiDAR range-image segmentation:** processes geometry (range) and intensity in separate streams during early encoding "when their physical origin, scale, noise, and failure modes differ most," and postpones interaction (secondary: search snippet only; the MDPI page returned 403, so no numbers were extracted). — [MDPI Electronics 15(17):3983](https://doi.org/10.3390/electronics15173983)
- **Lidar-as-camera (Yu et al., arXiv 2203.04064, 2022):** off-the-shelf YOLOv5/YOLOX/Faster R-CNN/Mask R-CNN work on Ouster signal images. Depth, near-IR and reflectivity images "did not perform as well with out-of-the-box deep learning models without further preprocessing." — [arXiv 2203.04064](https://arxiv.org/abs/2203.04064)

### Inferences
- For PoleSight's intensity + range pair (same sensor, same grid, no misalignment), the strongest analogue (Casado-Coscolla 2024) supports channel-stacked early fusion in YOLOv8-seg, with range + reflectivity the best subset. Whether a mid-fusion two-stream network would beat stacking there was not tested.
- Given the CMX RGB+RGB / RGB+Noise control, any PoleSight two-stream result should be compared against two controls: (a) a capacity-matched single-stream model (wider or deeper) and (b) a two-stream model with a duplicated input. Without them, a +0.5–1 mAP gain cannot be attributed to fusion.
- Preprocessing of each LiDAR channel (histogram equalization for intensity, clamped normalization for range) may matter as much as the fusion stage (Casado-Coscolla; Yu et al.).

### Gaps
- I found no controlled early-vs-mid-vs-late study on hyperspectral band subsets or multi-echo LiDAR for detection or instance segmentation.
- No paper compared HHA-as-separate-stream vs depth+HHA stacked.
- Per-channel-subset numbers from Casado-Coscolla Fig. 5 were not extracted (figure-only).

## Q4. Parameter- and data-efficiency: which fusion approaches work with ~1,000 training images?

### Takeaway
The best small-data evidence (VEDAI, 1,089 training images per fold, YOLOv5s) favours pixel-level early fusion or a single shallow fusion point over multi-stage mid fusion. Multi-stage fusion was 10 mAP50 worse. Late fusion of separately trained single-modal models (ProbEn) needs no paired multimodal training data and reuses strong single-modal detectors. "Frozen strong detector + zero-initialized side branch" (YOLOv11-RGBT MCF) and low-rank adapters (TensorFact) are the parameter-efficient options with evidence. Heavy transformer/Mamba fusion models (120–300M params) are validated only on 4k–50k-pair datasets.

### Cited Findings
- VEDAI (1,089 train / 121 test per fold): pixel concat 69.5 vs best feature-level 68.5 vs multistage feature fusion 59.3 mAP50, YOLOv5s-noFocus (~7.1M params). — [SuperYOLO, arXiv 2209.13351, Table III](https://arxiv.org/abs/2209.13351)
- VEDAI early-fusion gains over RGB-only are small (+0.1 to +2.0 mAP50 across YOLOv3/v4/v5s–x). — [SuperYOLO, Table VII](https://arxiv.org/abs/2209.13351)
- ICAFusion on VEDAI: 76.62 vs 74.66 baseline mAP50 (+1.96) (secondary). — [arXiv 2308.07504](https://arxiv.org/html/2308.07504)
- ProbEn needs no multimodal training data ("does not require any multimodal data for training"). Late fusion of single-modal detectors beat mid fusion on KAIST and FLIR. — [ECCV 2022 PDF](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690139.pdf)
- YOLOv11-RGBT single-node P3 fusion: fewer parameters than multi-node (58.65M vs 80.04M for x) and usually higher accuracy. MCF freezes a pretrained detector and adds the second modality through a zero conv. — [arXiv 2506.14696](https://arxiv.org/html/2506.14696)
- TensorFact (arXiv 2309.16592): factorized conv layers trained on RGB, then only the IR-specific low-rank factors trained. On FLIR ADAS v1 IR with 62 training images (1%): mAP50 0.5849 → 0.6205 while training only 1.86M of 37.2M parameters (Table 4). — [arXiv 2309.16592, Table 4](https://arxiv.org/abs/2309.16592)
- DFormerv2-S (26.7M params) matches CMX-B5 (181M) on NYUv2 (56.0 vs 56.9) but relies on RGB-D ImageNet pretraining. — [arXiv 2504.04701, Table 1](https://arxiv.org/abs/2504.04701)
- Fusion-Mamba-YOLOv8 uses 287.6M params vs 76.7M for single-modal YOLOv8-l (secondary). — [arXiv 2404.09146](https://arxiv.org/html/2404.09146)
- Bag-of-tricks: fusion-module ranking flips with backbone (NIN vs ICFE), with run-to-run std of 1.2–2.9 MR on KAIST. — [arXiv 2411.18288](https://arxiv.org/html/2411.18288)

### Inferences
- With 870 training images, PoleSight has fewer than the 1,089 per-fold images of VEDAI. The only small-data fusion-stage ablation favours early fusion or a single shallow fusion node. The bag-of-tricks paper shows run-to-run std of about 1–3 points. Differences of 1–2 mAP between fusion variants on PoleSight would therefore need multiple seeds or k-fold CV to be credible (SuperYOLO used 10-fold CV on VEDAI).
- SuperYOLO's Focus-removal result (+7.3 mAP50) suggests that for 71 px² masks, keeping stride or resolution high (no early downsampling; P2 head; tiling the 1024x128 strip as Casado-Coscolla did) may pay off more than the choice of fusion module. This is an inference from detection results on small aerial vehicles.
- Suggested low-risk ranking for a ~1k-image, same-sensor two-channel task, inferred from the above: (1) channel-stacked early fusion in YOLO-seg; (2) late fusion (ProbEn/NMS/WBF) of two single-channel YOLO-seg models; (3) single-node (P3) two-stream mid fusion or MCF-style frozen branch + zero conv; (4) heavy attention/Mamba fusion only if (1)–(3) plateau.

### Gaps
- No controlled fusion-stage study with explicit training-set-size sweeps (e.g., 10%/25%/100%) was found for detection or instance segmentation.
- No evidence was found on fusion for instance segmentation of tiny objects (<100 px²) specifically.
