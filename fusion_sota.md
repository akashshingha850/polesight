# Intensity–range fusion on PoleSight: state of the art and candidate methods

Companion to [fusion.md](fusion.md). Branch `fusion`, written 2026-10-04.

The question this document answers: **which published methods, architectures and models are most likely to make a model trained on fused `intensity_filtered` + `range_filtered` input beat the better of the two single-modality models**, and how should that be tested so that the result holds up in review.

---

## 0. How this was done

- **Search.** Web search across seven areas: range-view LiDAR segmentation, RGB-X dense fusion, RGB-infrared (multispectral) detection, decision-level fusion, foundation models for range and multimodal input, modality-imbalance training, and small/thin-object detection. Queries were seeded from the method families listed in `fusion.md` and expanded through related-work links.
- **Verification.** Every cited work was confirmed to exist through its arXiv, CVF, PMLR, publisher or official GitHub page. Most claims below come from abstracts and project pages; the full text was not read for most papers. Numbers quoted from a paper are the paper's own claims on its own benchmark and **say nothing about the gain to expect on PoleSight**.
- **Code checks.** Statements about Ultralytics behaviour were checked against the installed source, `ultralytics==8.4.19` in `.venv/`, the version pinned for the published sweep. File and line references are given so they can be rechecked.
- **Not done.** No systematic PRISMA screening and no risk-of-bias scoring. This is a structured, targeted review. The reference list is a starting set, not an exhaustive one.

---

## 1. Analysis of `fusion.md`

`fusion.md` gets the overall shape right: the two renderings are pixel-registered, the per-class reversal on `fence_pole` hints at complementarity, the baselines never saw two distinct channels, and the plan runs cheap experiments first. Six points need correcting or adding before any fusion run.

### 1.1 The input resolution is the largest uncontrolled factor

Every published run used `imgsz: 640` with `rect: false` (`results/intesity_filtered/*/args.yaml`). Ultralytics resizes the long side to `imgsz` (`ultralytics/data/base.py:240`), so each 1024×128 frame is shrunk to 640×80 before letterboxing: 0.625× linear and 0.39× in area. The median mask of 71 px² becomes about 28 px², and a pole one column wide in the source becomes sub-pixel.

The consequence for fusion research: a fusion gain of one or two points can be smaller than what training at native resolution (`imgsz: 1024`, `rect: true` or a 1024×128-aware letterbox) gives on its own, and the two effects may interact. Thin poles that disappear at 0.625× cannot be rescued by a second modality. **Resolution belongs in the experimental design as a factor, not left at 640.**

### 1.2 Naive N-channel input throws away the pretrained stem

`fusion.md` Approach B says to initialise extra input channels from the mean of the pretrained RGB filters. Ultralytics does not do this. Its weight transfer, `intersect_dicts` (`ultralytics/utils/torch_utils.py:518-529`), keeps a pretrained tensor only if its shape matches exactly. With `channels: 2` or `channels: 4` the first convolution, `model.0.conv.weight`, changes shape and is silently left at random initialisation, while every other layer loads. The log still reports a near-complete transfer ([issue #21278](https://github.com/ultralytics/ultralytics/issues/21278) shows a user puzzled by exactly this).

So a 2- or 4-channel run differs from the baselines in two ways at once: the input, and a randomly initialised stem. Either write the stem initialisation yourself (replicate or average the RGB filters, rescaled by 3/N) or stay at three channels.

### 1.3 Colour augmentation becomes active on packed input and mixes the modalities

The baselines are effectively greyscale: three identical channels, so hue and saturation are zero and `hsv_h: 0.015`, `hsv_s: 0.7` do nothing. Pack intensity and range into different channels and `RandomHSV` (`ultralytics/data/augment.py:1405-1421`) now runs a BGR→HSV→BGR round trip with random hue and saturation shifts. That remixes the two modalities across channels in a different way for each sample. It skips only images that do not have exactly three channels (line 1406).

For packed 3-channel runs, set `hsv_h: 0` and `hsv_s: 0`. Leave `hsv_v` on, or replace it with per-channel gain jitter. Otherwise the fused run differs from the baselines in augmentation as well as input.

### 1.4 16-bit precision is still not reachable through the stock loader

Ultralytics 8.4.19 supports multi-channel input: `channels` in the data YAML (`ultralytics/data/utils.py:446`), added in 8.3.112 ([PR #20223](https://github.com/ultralytics/ultralytics/pull/20223)), plus multi-page TIFF stacking (`ultralytics/utils/patches.py:36-41`). Two limits remain. Normalisation is a fixed `/255` (`ultralytics/models/yolo/detect/train.py:119`), and the mixup and HSV code paths assume `uint8`. A 16-bit TIFF would load but be scaled wrongly. Full-precision fusion needs a custom dataset or trainer `preprocess`. The practical route is a deliberate 8-bit encoding per modality: percentile clipping for intensity, and log or inverse range for range.

### 1.5 Modality imbalance is missing from the plan

Range is the stronger modality on 14 of 15 checkpoints. Jointly trained multimodal networks tend to fit the dominant modality and underfit the weaker one, and can end up no better than the best single-modality network ([Huang et al., 2022](https://proceedings.mlr.press/v162/huang22e.html); [Peng et al., 2022](https://arxiv.org/abs/2203.15332)). If the fused model only matches `range_filtered`, this is the first explanation to test, and the fix is a training change, not a new architecture (§3.6).

### 1.6 The test set is small for per-class claims

The test split has 611 instances: 95 `fence_pole`, 78 `gantry_sign_pole`, 314 `light_pole`, 124 `traffic_pole` (`data/dataset_analysis_report.json`). Per-class mask AP on 78–95 instances moves noticeably from seed to seed. The `fence_pole` reversal, the strongest argument for complementarity, needs seed repeats and a paired bootstrap over test images before it can carry a paper's claim.

---

## 2. What the literature says about this specific setting

Three properties of PoleSight make it unlike most fusion benchmarks, and they decide which methods transfer.

| Property | PoleSight | Typical fusion benchmark | Consequence |
|---|---|---|---|
| Registration | Exact, same projection, same pixel grid | RGB-T and RGB-D are approximately aligned; FLIR is unaligned | Pixel-wise fusion is safe; alignment and calibration modules add nothing |
| Sensor | Two renderings of one LiDAR return | Two physical sensors | Redundancy may be high, so complementarity is an empirical question |
| Target size | 98% COCO-small, median 71 px², 128-row images | Medium to large objects, 480–1080 rows | Methods that downsample early, such as ViT patchify at /16, are risky |

### 2.1 Range-view LiDAR networks already fuse these channels at the input

The range-view segmentation literature treats intensity (remission) plus range as the default input, usually alongside x, y, z:

- SqueezeSeg takes a 5-channel input (x, y, z, intensity, range), and its authors report a serious accuracy loss without intensity. SqueezeSegV2 adds a binary validity-mask channel and reports a clear gain on small classes such as cyclists ([Wu et al., 2019](https://arxiv.org/abs/1809.08495)).
- RangeNet++, SalsaNext, CENet and FIDNet all use stacked range, xyz and remission input in a single encoder ([SalsaNext](https://arxiv.org/pdf/2003.03653); [CENet](https://arxiv.org/abs/2207.12691); [FIDNet](https://arxiv.org/pdf/2109.03787)).
- Filling missing values in the range image (SU++ projection and range-dependent KNN interpolation) improves existing range-view models ([Chen, Gong & Röning, 2024](https://arxiv.org/abs/2405.10175)). It comes from a University of Oulu group, the same Finnish context as PoleSight.

**Implication.** Early fusion of intensity and range is the field's established baseline, so a PoleSight fusion paper must beat early fusion, not just the single modalities. Novelty has to come from mid-level fusion, training strategy, or task framing.

Unlike the SemanticKITTI pipelines, PoleSight's release has **no xyz vertex map**. `range_filtered` carries only the range magnitude, and elevation and azimuth are implicit in the pixel row and column. Row and column coordinates can be added as CoordConv-style channels at no cost. `rangegen` can emit the full vertex map, but only from the unreleased point clouds.

### 2.2 Range-view networks with pretrained image backbones

- **RangeViT** (CVPR 2023): an ImageNet-21k/Cityscapes-pretrained ViT with a convolutional stem and decoder for range images. It shows that image pretraining transfers to range input ([Ando et al., 2023](https://arxiv.org/abs/2301.10222); [code](https://github.com/valeoai/rangevit)).
- **RangeFormer** (ICCV 2023): names the many-to-one mapping, semantic incoherence and shape deformation as the obstacles in range view, and adds range-specific augmentation and the STR multi-view training scheme ([Kong et al., 2023](https://arxiv.org/abs/2303.05367)).
- **RangeSAM** (WACV 2026 workshop): a SAM2 Hiera encoder adapted to range view, with a stem that emphasises horizontal dependencies ([Kühn et al., 2025](https://arxiv.org/abs/2509.15886)).

All three are semantic or panoptic segmentation models on 64-beam scans. For PoleSight they matter as backbones and design patterns: a convolutional stem before any patchify, horizontally elongated kernels, and range-view augmentation. They are not drop-in instance segmenters.

### 2.3 Dense multimodal fusion (RGB-X)

These are the strongest mid-level fusion designs with public code. They are written for segmentation with an "RGB" primary stream and an "X" auxiliary stream. On PoleSight, intensity plays the appearance role and range plays the depth/geometry role.

| Method | Venue | Fusion mechanism | Fit to PoleSight |
|---|---|---|---|
| CMX ([Zhang et al.](https://arxiv.org/abs/2203.04838)) | IEEE T-ITS | Two streams; cross-modal feature rectification (channel and spatial) and a cross-attention fusion module at every stage. Tested on depth, thermal, polarisation, event and LiDAR | High. Modality-agnostic, multi-scale, exact registration helps rectification |
| CMNeXt / DeLiVER ([Zhang et al., 2023](https://arxiv.org/abs/2303.01480)) | CVPR 2023 | RGB hub with a lightweight Self-Query Hub for any number of auxiliary modalities (~0.1M parameters each); quad-modal reports +9.10 mIoU over mono-modal on DeLiVER | Medium. Designed for 3+ modalities; useful if derived channels (gradient, normals, CoordConv) are treated as extra modalities |
| TokenFusion ([Wang et al., 2022](https://arxiv.org/abs/2204.08721)) | CVPR 2022 | Detects uninformative tokens and replaces them with projected tokens from the other modality | Low to medium. Token pruning at ViT resolution is risky for 71 px² targets |
| GeminiFusion ([Jia et al., 2024](https://arxiv.org/abs/2406.01210)) | ICML 2024 | Pixel-wise intra- plus inter-modal attention at linear cost; reports beating token exchange and matching cross-attention | High. Built on the premise of aligned cross-modal pixels, which PoleSight satisfies exactly |
| DFormer ([Yin et al., 2024](https://arxiv.org/abs/2309.09668)) | ICLR 2024 | Backbone pretrained on RGB-D pairs, so depth is not pushed through an RGB-pretrained encoder | Medium. The pretraining argument holds; the pretraining data is not LiDAR |
| DFormerv2 ([Yin et al., 2025](https://arxiv.org/abs/2504.04701)) | CVPR 2025 | Geometry Self-Attention: depth and patch distance become an attention prior instead of a second encoded stream | **High and novel.** Range is literally depth on the same grid. Using range only as a geometric attention prior over an intensity stream is a principled, untested design for LiDAR range images |
| Sigma ([Wan et al., 2025](https://arxiv.org/abs/2404.04256)) | WACV 2025 | Siamese Mamba encoders, Mamba cross- and concat-fusion, channel-aware decoder | Medium. Linear-cost global context suits 1024-wide panoramas; the Mamba toolchain is heavier |

### 2.4 Multispectral (RGB-IR) detectors, the closest match for detection and instance heads

This line of work is the most direct template for a YOLO-family fused instance segmenter. It uses two backbones, fuses at P3–P5, and keeps a standard neck and head.

| Method | Venue | Mechanism | Notes for PoleSight |
|---|---|---|---|
| CFT ([Fang et al., 2021](https://arxiv.org/abs/2111.00273)) | arXiv | Transformer self-attention over concatenated two-stream tokens at several backbone stages, on YOLOv5 | Simple baseline for attention-based mid fusion in YOLO |
| ICAFusion ([Shen et al., 2024](https://arxiv.org/abs/2308.07504)) | Pattern Recognition | Dual cross-attention with iterative, parameter-shared blocks; works when one modality is degraded | Good parameter efficiency for an 870-frame training set |
| C²Former ([Yuan & Wei, 2024](https://arxiv.org/abs/2306.16175)) | IEEE TGRS | Inter-modality cross-attention plus adaptive feature sampling; calibrates misaligned modalities | Calibration is unnecessary here; the complementary-feature part still applies |
| SuperYOLO ([Zhang et al., 2023](https://arxiv.org/abs/2209.13351)) | IEEE TGRS | Pixel-level multimodal fusion plus a training-only super-resolution branch for small objects; reports +10 mAP50 over YOLOv5l/x on VEDAI at much lower cost | **High.** Targets the same small-object problem; the SR branch is dropped at inference |
| Fusion-Mamba ([Dong et al., 2025](https://arxiv.org/abs/2404.09146)) | IEEE TMM | Shallow channel-swap fusion plus deep gated Mamba fusion | Channel swapping is cheap and well suited to same-sensor inputs |
| WaveMamba ([Zhu et al., 2025](https://arxiv.org/abs/2507.18173)) | ICCV 2025 | Fuses the two modalities in the wavelet domain; reports +4.5 mAP on average over four benchmarks | Frequency-domain fusion fits thin, high-frequency vertical structure |
| MCF-YOLO ([Sensors, 2026](https://doi.org/10.3390/s26123938)) | Sensors | Dual YOLO11n branches with consistency-guided cross-modal attention for small RGB-IR objects | Same backbone family as the PoleSight baselines; lower-tier venue |

### 2.5 Decision-level fusion

- **ProbEn** (ECCV 2022 oral) fuses the outputs of independent single-modality detectors with a non-learned Bayesian rule. It reports more than 13% relative improvement over prior work on KAIST and FLIR ([Chen et al., 2022](https://arxiv.org/abs/2104.02904)). It is a strong argument that late fusion of good single-modality models is a hard baseline for learned fusion to beat.
- **Weighted Boxes Fusion** averages confidence-weighted boxes instead of suppressing them ([Solovyev et al., 2021](https://arxiv.org/abs/1910.13302)).

Neither handles masks. For instance segmentation the late-fusion step needs mask aggregation, for example per-pixel averaging of mask probabilities over the matched instances after box fusion. Writing that down cleanly is a small but genuine contribution.

### 2.6 Foundation models

- **MM-SAM** extends SAM to LiDAR+RGB, depth+RGB and thermal+RGB through unsupervised cross-modal transfer and weakly supervised fusion ([Xiao et al., 2024](https://arxiv.org/abs/2408.09085)).
- **RangeSAM** (above) shows that the SAM2 encoder can be adapted to range view.
- **DINOv3** offers strong frozen dense features with adapter-based transfer ([Siméoni et al., 2025](https://arxiv.org/abs/2508.10104)).

On 128-row inputs with 71 px² targets, a /16 or /14 patch embedding collapses most poles into one or two tokens. These models are only viable with input upsampling, a convolutional high-resolution stem (as RangeViT does) or tiling. They are a second-phase direction, not a first experiment.

### 2.7 Training strategies that target fusion failure modes

- **Modality competition** gives a theoretical account of why joint training can fail to learn the weaker modality ([Huang et al., 2022](https://proceedings.mlr.press/v162/huang22e.html)).
- **OGM-GE** (CVPR 2022 oral) slows the gradient of the dominant modality's branch in proportion to its lead, plus a Gaussian-noise term; it plugs into existing fusion models ([Peng et al., 2022](https://arxiv.org/abs/2203.15332)). It was designed for classification, so adapting it to a detection or segmentation loss is new work.
- **Modality dropout** (ModDrop) drops whole modalities during training, forcing each branch to stay useful on its own ([Neverova et al., 2016](https://arxiv.org/abs/1501.00102)).
- **Cross-modal knowledge distillation**: a fused teacher can train a single-modality student, or the reverse. This is mature for LiDAR-camera ([2DPASS](https://arxiv.org/abs/2207.04397); [U2MKD, TPAMI 2024](https://ieeexplore.ieee.org/document/10659158/)). For PoleSight it gives a deployment story: train fused, ship a range-only model.

### 2.8 Small and thin objects (needed whatever the fusion)

- **SAHI**: slicing-aided fine-tuning and inference; reports cumulative AP gains of about 13–15 points on aerial benchmarks ([Akyon et al., 2022](https://arxiv.org/abs/2202.06934)). On 1024×128 frames the natural slicing is horizontal: 4×256 or 2×512 crops with overlap.
- **Stride-4 (P2) head**: the installed Ultralytics ships `yolov8-p2.yaml` and `yolo26-p2.yaml` (detection only, no `-seg` variant; `ultralytics/cfg/models/`). A P2 segmentation variant needs a custom YAML.

### 2.9 Domain-adjacent prior work

- **Range-image pole extraction**: geometric pole extraction on range images, and a learned range-image pole segmenter trained on its pseudo-labels ([Dong, Chen & Stachniss, 2021](https://arxiv.org/abs/2108.08621); [Dong et al., 2023](https://arxiv.org/abs/2208.07364)). This is the closest prior work to PoleSight's task and should be cited and, if possible, compared against.
- **LiDAR-as-camera images**: general-purpose detectors and segmenters applied to Ouster signal, reflectivity, near-IR and range images; from Turku, also a Finnish group ([Yu et al., 2023](https://doi.org/10.3390/s23062936)). It is related evidence on running off-the-shelf 2D models on LiDAR renderings.
- **WHU-Infra3D**: a 2026 multimodal roadside-infrastructure benchmark (panoramic images plus LiDAR, 10 categories including street lights and traffic signs) ([Liu et al., 2026](https://arxiv.org/abs/2606.09882)). It is the most relevant recent dataset to position against.

---

## 3. Candidate methods for PoleSight, ranked

Each tier reuses the previous one's infrastructure, and each is a publishable ablation row. Run them all on the same model scale (`yolo11m-seg` or `yolo11l-seg`), the same resolution setting and three seeds.

### 3.1 Tier 0: fix the single-modality baselines first

Retrain intensity-only and range-only with (a) `imgsz: 1024` and rectangular batching or a 1024×128-aware letterbox, and (b) colour augmentation off. These become the reference numbers that fusion has to beat. Without this step a fusion gain cannot be separated from a resolution gain (§1.1).

### 3.2 Tier 1: early fusion, done properly

| Variant | Channels | Stem |
|---|---|---|
| E1 | `[I, R, I]`, 8-bit each, HSV hue/sat off | Pretrained, unchanged |
| E2 | `[I, R, ∂R/∂col]` (horizontal range gradient, edge-like for poles) | Pretrained |
| E3 | `[I, R, row, col]` (CoordConv) or `[I, R, mask]` | 4-channel, stem initialised by replicating the RGB filter mean and rescaling |

Literature basis: SqueezeSeg, RangeNet++ and SalsaNext input stacks (§2.1). Expectation: this is the baseline the field would expect, and a fair chance of matching or slightly beating range-only.

### 3.3 Tier 2: decision-level fusion

Late fusion of the Tier-0 intensity and range models with ProbEn or WBF for boxes, plus mask-probability averaging over matched instances. Include a per-class score weight tuned on the validation split to exploit the `fence_pole` reversal. **Control:** fuse two range models trained with different seeds, so that the modality gain is separated from the ensembling gain. ProbEn shows late fusion of single-modality models is a strong baseline, so learned mid fusion must beat this row to be worth its cost.

### 3.4 Tier 3: mid-level fusion in a YOLO-seg backbone (main candidate)

Architecture: two YOLO11 backbones, one per modality, **each warm-started from its own Tier-0 single-modality checkpoint** rather than from COCO. The two are fused at P3, P4 and P5 (plus P2 if a P2 head is used) and feed one shared neck and segmentation head. Fusion module options, in increasing cost:

1. Concatenation + 1×1 conv (the mid-fusion baseline from the multispectral literature).
2. Channel swap at shallow stages + gated fusion at deep stages (Fusion-Mamba's SFM idea, implemented with convolutions).
3. CMX rectification + fusion module (spatial and channel cross-modal rectification), or ICAFusion's parameter-shared cross-attention.
4. GeminiFusion-style pixel-wise attention, justified by the exact registration.

Why warm-starting matters: each branch starts from a model that already detects poles in its own modality, which removes most of the modality-competition risk at initialisation and makes the fusion layers the only new parameters. CFT, ICAFusion, C²Former and MCF-YOLO all report mid fusion beating early fusion on RGB-IR. Whether that holds for two renderings of one sensor is the open question this tier answers.

### 3.5 Tier 4: range as a geometric prior (most novel)

Following DFormerv2, use only the intensity stream as the encoded backbone and use range as the **prior for the attention or convolution weights**. For example, add a geometry bias to self-attention in the neck so that attention between two pixels decays with their range difference as well as their image distance. In 3D terms this separates a pole from the background directly behind it in the same image column, which is the hard case for thin vertical objects in a range image.

Two lighter variants: a range-gated depthwise convolution, or the range-guided masked attention of a Mask2Former-style decoder. No published work applies geometry-prior attention to LiDAR range-image instance segmentation, so this is the strongest novelty angle for a paper. It is also the highest-risk tier.

### 3.6 Tier 5: training strategy on top of the best architecture

- Modality dropout: zero one branch for 10–30% of iterations.
- OGM-GE-style gradient modulation adapted to the detection and segmentation loss: per-branch contribution measured with auxiliary single-modality heads.
- Distillation: the best fused model teaches a range-only `yolo11n-seg`, which gives the deployment-cost story.

### 3.7 What is not recommended first

- **ViT, SAM or DINO backbones at native 1024×128.** Patch embedding destroys sub-patch poles. Revisit with a convolutional stem and SAHI-style tiling.
- **Calibration and alignment modules** (C²Former's calibration, FLIR-style alignment). There is no misalignment to correct.
- **TokenFusion-style token pruning.** The tokens most likely to be judged "uninformative" are the thin structures being detected.

---

## 4. Experimental design for a defensible claim

**Primary claim to test.** Fused input gives higher test mask mAP50-95 than the better single modality, under identical training, resolution and augmentation.

1. **Factors.** Modality ∈ {I, R, early fusion, late fusion, mid fusion, geometry-prior} × resolution ∈ {640, 1024} × seed ∈ {0, 1, 2}. Hold the model scale fixed (`yolo11m-seg`, plus `yolo11l-seg` for the final tier).
2. **Controls.** `[I, I, I]` and `[R, R, R]` at the fused settings; a same-modality late-fusion ensemble; for mid fusion, a two-branch model fed the same modality twice (`R ⊕ R`). The last control separates "two modalities" from "twice the parameters".
3. **Statistics.** Report the mean ± standard deviation over seeds. For the headline comparison, use a paired bootstrap over the 109 test images (resample images, recompute mAP), giving a confidence interval for Δ(fused − best single). Per-class results carry the bootstrap interval; do not report bare per-class deltas.
4. **Stratification.** Per class (`fence_pole` is the case to watch), per route from `data/metadata.json`, and per object-size bin. Fusion should help most on the smallest masks if the complementarity story holds.
5. **Cost.** Report parameters, GFLOPs and latency for every row, as `results/` already does, so that mid fusion is weighed against its roughly 2× backbone cost.
6. **Pre-specify.** Fix the model scale, the metric and the comparison before running Tier 3 onwards. Fusion work has many knobs, and post-hoc selection is the usual reviewer complaint.

---

## 5. Implementation notes (Ultralytics 8.4.19, checked in source)

| Need | Status in 8.4.19 | Action |
|---|---|---|
| N-channel input | `channels:` key in data YAML (`data/utils.py:446`); multi-page TIFF stacked (`utils/patches.py:36-41`) | Use for E3; write packed TIFFs |
| Pretrained stem with N≠3 | Not transferred: shape-checked by `intersect_dicts` (`utils/torch_utils.py:529`) | Custom stem init before training |
| 16-bit input | Normalisation fixed at `/255` (`models/yolo/detect/train.py:119`) | Encode to 8-bit per modality, or override `preprocess` |
| HSV augmentation | Runs only on exactly-3-channel images (`data/augment.py:1406`) | `hsv_h: 0`, `hsv_s: 0` for packed 3-channel |
| Resize | Long side to `imgsz` (`data/base.py:240`); published runs used 640, `rect: false` | Test `imgsz: 1024` |
| P2 head | `yolov8-p2.yaml`, `yolo26-p2.yaml` (detect only) | Write a `-seg-p2` YAML |
| Two-stream backbone | Not supported by the YAML parser's single input | Custom `nn.Module` wrapping two backbones, or a fork such as the multispectral YOLO repos |

---

## 6. Risks and limitations

- **Redundancy.** Both renderings come from the same returns. If intensity adds little beyond what range already encodes, every tier will show small gains. The early-fusion and late-fusion rows with their controls are the cheapest way to find out before investing in Tiers 3–4.
- **Transfer gap.** Nearly all of the cited fusion evidence is RGB-thermal, RGB-depth or camera-LiDAR at much higher resolution. No cited work fuses two renderings of one LiDAR frame for instance segmentation of thin objects, so expected gains are unknown.
- **Data scale.** 870 training frames are small for two-stream transformers. Prefer convolutional fusion modules and warm-started branches.
- **Verification depth.** The cited claims come from abstracts and official pages. Read the method and results sections of any paper before using it as a baseline or quoting its numbers.
- **AI assistance.** This review was prepared with AI-assisted search and drafting tools.

---

## 7. References

Grouped as in §2. Links go to arXiv, CVF, PMLR or publisher pages.

**Range-view LiDAR segmentation**
- Ando, A., Gidaris, S., Bursuc, A., Puy, G., Boulch, A., & Marlet, R. (2023). RangeViT: Towards vision transformers for 3D semantic segmentation in autonomous driving. *CVPR*. https://arxiv.org/abs/2301.10222
- Chen, B., Gong, C., & Röning, J. (2024). Filling missing values matters for range image-based point cloud segmentation. *arXiv:2405.10175*. https://arxiv.org/abs/2405.10175
- Cheng, H., Han, X., & Xiao, G. (2022). CENet: Toward concise and efficient LiDAR semantic segmentation for autonomous driving. *ICME*. https://arxiv.org/abs/2207.12691
- Cortinhal, T., Tzelepis, G., & Aksoy, E. E. (2020). SalsaNext: Fast, uncertainty-aware semantic segmentation of LiDAR point clouds. *ISVC*. https://arxiv.org/abs/2003.03653
- Kong, L., Liu, Y., Chen, R., Ma, Y., Zhu, X., Li, Y., Hou, Y., Qiao, Y., & Liu, Z. (2023). Rethinking range view representation for LiDAR segmentation. *ICCV*, 228–240. https://arxiv.org/abs/2303.05367
- Kühn, P. J., Nguyen, D. A., Kuijper, A., & Sinha, S. N. (2025). RangeSAM: On the potential of visual foundation models for range-view represented LiDAR segmentation. *WACV Workshops 2026*. https://arxiv.org/abs/2509.15886
- Wu, B., Zhou, X., Zhao, S., Yue, X., & Keutzer, K. (2019). SqueezeSegV2: Improved model structure and unsupervised domain adaptation for road-object segmentation from a LiDAR point cloud. *ICRA*. https://arxiv.org/abs/1809.08495
- Zhao, Y., Bai, L., & Huang, X. (2021). FIDNet: LiDAR point cloud semantic segmentation with fully interpolation decoding. *IROS*. https://arxiv.org/abs/2109.03787

**RGB-X dense fusion**
- Jia, D., Guo, J., Han, K., Wu, H., Zhang, C., Xu, C., & Chen, X. (2024). GeminiFusion: Efficient pixel-wise multimodal fusion for vision transformer. *ICML*. https://arxiv.org/abs/2406.01210
- Wan, Z., et al. (2025). Sigma: Siamese Mamba network for multi-modal semantic segmentation. *WACV*. https://arxiv.org/abs/2404.04256
- Wang, Y., Chen, X., Cao, L., Huang, W., Sun, F., & Wang, Y. (2022). Multimodal token fusion for vision transformers. *CVPR*. https://arxiv.org/abs/2204.08721
- Yin, B., Zhang, X., Li, Z., Liu, L., Cheng, M.-M., & Hou, Q. (2024). DFormer: Rethinking RGBD representation learning for semantic segmentation. *ICLR*. https://arxiv.org/abs/2309.09668
- Yin, B.-W., Cao, J.-L., Cheng, M.-M., & Hou, Q. (2025). DFormerv2: Geometry self-attention for RGBD semantic segmentation. *CVPR*. https://arxiv.org/abs/2504.04701
- Zhang, J., Liu, H., Yang, K., Hu, X., Liu, R., & Stiefelhagen, R. (2023). CMX: Cross-modal fusion for RGB-X semantic segmentation with transformers. *IEEE T-ITS*. https://arxiv.org/abs/2203.04838
- Zhang, J., Liu, R., Shi, H., Yang, K., Reiß, S., Peng, K., Fu, H., Wang, K., & Stiefelhagen, R. (2023). Delivering arbitrary-modal semantic segmentation. *CVPR*. https://arxiv.org/abs/2303.01480

**Multispectral detection**
- Dong, W., et al. (2025). Fusion-Mamba for cross-modality object detection. *IEEE TMM*, 27, 7392–7406. https://arxiv.org/abs/2404.09146
- Fang, Q., Han, D., & Wang, Z. (2021). Cross-modality fusion transformer for multispectral object detection. *arXiv:2111.00273*. https://arxiv.org/abs/2111.00273
- Shen, J., Chen, Y., Liu, Y., Zuo, X., Fan, H., & Yang, W. (2024). ICAFusion: Iterative cross-attention guided feature fusion for multispectral object detection. *Pattern Recognition*, 145, 109913. https://arxiv.org/abs/2308.07504
- Yuan, M., & Wei, X. (2024). C²Former: Calibrated and complementary transformer for RGB-infrared object detection. *IEEE TGRS*. https://arxiv.org/abs/2306.16175
- Zhang, J., Lei, J., Xie, W., Fang, Z., Li, Y., & Du, Q. (2023). SuperYOLO: Super resolution assisted object detection in multimodal remote sensing imagery. *IEEE TGRS*. https://arxiv.org/abs/2209.13351
- Zhu, H., et al. (2025). WaveMamba: Wavelet-driven Mamba fusion for RGB-infrared object detection. *ICCV*. https://arxiv.org/abs/2507.18173
- MCF-YOLO: Consistency-guided cross-modal attention for small-object RGB-IR detection. (2026). *Sensors*, 26(12), 3938. https://doi.org/10.3390/s26123938

**Decision-level fusion**
- Chen, Y.-T., Shi, J., Ye, Z., Mertz, C., Ramanan, D., & Kong, S. (2022). Multimodal object detection via probabilistic ensembling. *ECCV*. https://arxiv.org/abs/2104.02904
- Solovyev, R., Wang, W., & Gabruseva, T. (2021). Weighted boxes fusion: Ensembling boxes from different object detection models. *Image and Vision Computing*, 107, 104117. https://arxiv.org/abs/1910.13302

**Foundation models**
- Siméoni, O., et al. (2025). DINOv3. *arXiv:2508.10104*. https://arxiv.org/abs/2508.10104
- Xiao, A., et al. (2024). Segment anything with multiple modalities. *arXiv:2408.09085*. https://arxiv.org/abs/2408.09085

**Training strategies**
- Huang, Y., Lin, J., Zhou, C., Yang, H., & Huang, L. (2022). Modality competition: What makes joint training of multi-modal network fail in deep learning? (Provably). *ICML*, 9226–9259. https://proceedings.mlr.press/v162/huang22e.html
- Neverova, N., Wolf, C., Taylor, G., & Nebout, F. (2016). ModDrop: Adaptive multi-modal gesture recognition. *IEEE TPAMI*, 38(8). https://arxiv.org/abs/1501.00102
- Peng, X., Wei, Y., Deng, A., Wang, D., & Hu, D. (2022). Balanced multimodal learning via on-the-fly gradient modulation. *CVPR*. https://arxiv.org/abs/2203.15332
- Yan, X., et al. (2022). 2DPASS: 2D priors assisted semantic segmentation on LiDAR point clouds. *ECCV*. https://arxiv.org/abs/2207.04397
- Uni-to-multi modal knowledge distillation for bidirectional LiDAR-camera semantic segmentation. (2024). *IEEE TPAMI*. https://ieeexplore.ieee.org/document/10659158/

**Small objects**
- Akyon, F. C., Altinuc, S. O., & Temizel, A. (2022). Slicing aided hyper inference and fine-tuning for small object detection. *ICIP*. https://arxiv.org/abs/2202.06934

**Domain-adjacent**
- Dong, H., Chen, X., & Stachniss, C. (2021). Online range image-based pole extractor for long-term LiDAR localization in urban environments. *ECMR*. https://arxiv.org/abs/2108.08621
- Dong, H., Chen, X., Särkkä, S., & Stachniss, C. (2023). Online pole segmentation on range images for long-term LiDAR localization in urban environments. *Robotics and Autonomous Systems*. https://arxiv.org/abs/2208.07364
- Liu, C., Fu, L., Feng, X., Dong, Z., & Yang, B. (2026). WHU-Infra3D: A full-stack multi-modal dataset and benchmark for 3D roadside infrastructure inventory. *arXiv:2606.09882*. https://arxiv.org/abs/2606.09882
- Yu, X., Salimpour, S., Peña Queralta, J., & Westerlund, T. (2023). General-purpose deep learning detection and segmentation models for images from a lidar-based camera sensor. *Sensors*, 23(6), 2936. https://doi.org/10.3390/s23062936

**Tooling**
- Ultralytics. (2025). New YOLO multispectral image support, v8.3.112 (PR #20223). https://github.com/ultralytics/ultralytics/pull/20223
