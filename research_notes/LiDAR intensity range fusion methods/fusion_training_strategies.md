# Training strategies that decide whether multimodal fusion beats the best single modality (imbalance, modality dropout, distillation, non-RGB initialisation, 16-bit encoding, range-view augmentation)

Scope note: every number below was read from the paper's PDF/HTML table unless it is marked **[abstract-only]** or **[search-snippet-only]**. Where a fetch tool's summary disagreed with the PDF text, the PDF text was used. Some numbers were read from a rendered figure, and those are marked **[read from figure]**.

PoleSight context, which these notes do not research: 1024x128 16-bit range and intensity renderings, 870 training images, 4 pole classes. Range-only beats intensity-only on 14 of 15 YOLO-seg models. Ultralytics loads 8-bit 3-channel PNGs, and when channels != 3 it drops the pretrained first conv.

---

## 1. Modality competition / imbalance (Huang et al. 2022, OGM-GE, PMR, AGM, MMPareto, MLA, ReconBoost, D&R, BalanceBenchmark): does anyone report detection/segmentation results, and what are the gains?

### Takeaway
Nearly all of the balanced-multimodal-learning literature is evaluated on classification: audio-visual, RGB+flow, image-text. BalanceBenchmark (2025) has 7 datasets and every one is a classification task. One finding matters directly for PoleSight. When one modality is much stronger, gradient-balancing methods (OGM-GE, AGM, PMR) can lower fused accuracy below plain concatenation. A 2026 paper (PDMP) gets its best results by deliberately prioritising the stronger modality rather than balancing. Detection and segmentation evidence is thin and comes from task-specific architectures (RGB-T pedestrian detection, RGB-D segmentation), not from the generic gradient-modulation methods.

### Cited Findings
- **BalanceBenchmark covers classification only.** 7 datasets: Kinetics-Sounds, CREMA-D, BalancedAV, VGGSound, UCF-101 (RGB+flow), FOOD-101 (image-text), CMU-MOSEI. The BalanceMM toolkit implements 17 methods. — [BalanceBenchmark, arXiv 2502.10816](https://arxiv.org/abs/2502.10816)
- BalanceBenchmark Table 2, baseline vs best method (accuracy): KS 65.63 → 74.55 (MMPareto); CREMA-D 65.50 → 75.85 (Modality-valuation); UCF-101 81.80 → 86.91 (AMCo); FOOD-101 91.65 → 93.14 (MLA); CMU-MOSEI 78.99 → 81.01 (ReconBoost); BalancedAV 73.33 → 75.49 (AGM); VGGSound 48.08 → 51.58 (UMT). The authors say "almost all related methods outperform the Baseline", but also that "greater balance between modalities does not guarantee better performance". They describe a "relative balance point" past which reducing imbalance lowers accuracy, because modalities carry different amounts of information. They also find no method gives a satisfactory performance/compute trade-off. — [BalanceBenchmark HTML v3](https://arxiv.org/html/2502.10816v3) (values via fetch summary of Table 2)
- **Balancing hurts when one modality clearly dominates.** PDMP Table I (ResNet-18; accuracy / macro-F1) on AVE, where audio-only is 62.16 and visual-only 31.40: concatenation 66.15; OGM-GE 65.62; AGM 64.50; PMR 63.62; Grad-Blending 67.40; MMPareto 68.22; MLA 70.92; D&R 69.62; PDMP 71.28. So OGM-GE, AGM and PMR all fall *below* plain concatenation when one modality is about 2x stronger. The authors' explanation: "balance-based methods could hinder the learning of performance-dominant modality". — [PDMP, arXiv 2604.05773](https://arxiv.org/abs/2604.05773), Table I
- The same PDMP Table I shows plain joint training underperforming the best single modality on CREMA-D: concatenation 58.83 vs visual-only 62.17. Balancing methods recover and exceed it: OGM-GE 64.34, MMPareto 70.19, MLA 73.21, D&R 73.52, PDMP 80.21. — [PDMP](https://arxiv.org/abs/2604.05773), Table I. Note that these are PDMP's re-implementations. The text claims "+6.03%" on AVE over concatenation, but the table difference is 5.13 accuracy points, so the text and table are internally inconsistent.
- PDMP's method: it identifies the "performance-dominant" modality from unimodal performance and applies asymmetric gradient coefficients (>1 for the dominant modality, <1 for the rest) so that the dominant modality leads the fused model. Its experiments are all classification (CREMA-D, KS, AVE, CEFA, UCF-101, VGGSound). — [PDMP](https://arxiv.org/abs/2604.05773), Sec. III–IV
- **Detection: RGB-thermal under modality imbalance.** A base-and-auxiliary detector with quality-based modality weighting, pseudo-degradation and a consistency constraint reportedly cuts miss rate by 55% under extreme imbalance. **[abstract-only]** No dataset-level mAP or comparison with OGM-GE is in the abstract. — [Learning a Robust RGB-Thermal Detector for Extreme Modality Imbalance, arXiv 2505.22154](https://arxiv.org/abs/2505.22154)
- A regulariser that makes both feature extractors "equivalently important" in multispectral pedestrian detection (KAIST, UTokyo) is said to improve on the state of the art. **[abstract-only, no numbers]** — [Revisiting Modality Imbalance in Multimodal Pedestrian Detection, arXiv 2302.12589](https://arxiv.org/abs/2302.12589)
- MBNet (ECCV 2020) frames RGB-T pedestrian detection as a "modality imbalance problem" and adds differential-modality-aware fusion and illumination-aware alignment. Its ablation numbers could not be extracted: the PDF was over the fetch size limit. — [Zhou et al., arXiv 2008.03043](https://arxiv.org/abs/2008.03043)
- Uni-Modal Teacher / Uni-Modal Ensemble (Du et al., ICML 2023) argue that insufficient unimodal feature learning in late-fusion training hurts generalisation. Evaluated on VGG-Sound, Kinetics-400, UCF101 and ModelNet40, with no segmentation or detection. **[abstract-only]** — [Du et al., arXiv 2305.01233](https://arxiv.org/abs/2305.01233)
- **Segmentation: fusion often fails to beat the strong modality without special care.** DeLiVER/CMNeXt Table 1a, KITTI-360 (mIoU): the RGB-only SegFormer MiT-B2 scores 67.04. *Every* bimodal fusion scores lower: CMNeXt RGB-Depth 65.09, RGB-LiDAR 65.26, RGB-Event 66.13; CMX RGB-D 64.43, RGB-LiDAR 64.31; TokenFusion RGB-D 57.44. Only the 4-modality CMNeXt (67.84, +0.80) beats RGB-only. The authors attribute this to depth and events being derived from RGB, which leaves "limited modal differences". On DeLiVER, RGB 57.20 → RGB-LiDAR 58.04 (+0.84) with CMNeXt, but CMX RGB-LiDAR scores 56.37, *below* RGB. — [Zhang et al., CVPR 2023, arXiv 2303.01480](https://arxiv.org/abs/2303.01480), Table 1a and Sec. 5.2
- Huang et al. (ICML 2022) and OGM-GE (Peng et al., CVPR 2022) were not re-fetched in this session. In the PDMP and BalanceBenchmark tables they appear only on classification datasets (CREMA-D, KS, AVE, VGGSound, etc.). I found no paper reporting OGM-GE, PMR, AGM, MMPareto, MLA, ReconBoost or D&R on detection or segmentation. A targeted search for "OGM-GE" with mIoU or detection returned no such work. — [search result set](https://arxiv.org/pdf/2405.15365) (null result)

### Inferences
- PoleSight is a strong/weak pair: range beats intensity on 14/15 models, and intensity wins on only one class. That is the regime where PDMP shows OGM-GE, AGM and PMR *hurting* (AVE). Generic "balance the modalities" methods are a risky first choice. Strategies that keep the dominant range branch at full strength and add intensity as an auxiliary are better supported: PDMP-style prioritisation, unimodal auxiliary losses (MLA, D&R, UMT), or late/class-wise fusion. This is an extrapolation from classification to segmentation.
- KITTI-360 shows that even state-of-the-art RGB-X segmentation fusion (CMX, CMNeXt) can land 1–2.6 mIoU *below* the stronger modality alone. PoleSight should treat "fusion ≥ best single modality" as something to be shown, and always report the range-only baseline with the same backbone and schedule.
- Cheap PoleSight checks that follow from this literature: (a) unimodal auxiliary heads or losses on each branch (MLA/UMT style); (b) a scalar weight >1 on the range-branch gradient (PDMP style); (c) logging each branch's unimodal mAP during fused training to detect under-optimisation of either branch.

### Gaps
- None of OGM-GE, PMR, AGM, MMPareto, MLA, ReconBoost or D&R has published detection or instance-segmentation results that I could find. Their transfer to YOLO-seg-style dense prediction is untested.
- MBNet's and the "Revisiting Modality Imbalance" paper's numeric ablations were not retrieved (PDF too large / abstract only).
- No study was found on imbalance between *two channels of the same sensor* (e.g., LiDAR range vs intensity) as opposed to two sensors.

---

## 2. Modality dropout / missing-modality training (ModDrop, CMNeXt sensor-failure tests, UniBEV, ConD): does it improve *full-modality* accuracy, not just robustness?

### Takeaway
Across several papers, modality dropout is roughly neutral to slightly positive (0 to +1 mIoU or accuracy) on full-modality accuracy. Its large, consistent gains are on missing or degraded modalities (+3 to +28 mIoU). When the dropped "modality" is a pretrained-but-weaker one, it also makes the strong branch work on its own. For PoleSight, its main value is guaranteeing that a fused model never falls below range-only. It is not a source of large full-modality gains.

### Cited Findings
- **ModDrop (Neverova et al., TPAMI 2016)**, ChaLearn 2014 + audio (Table 9). All modalities present: Dropout 96.77% / Jaccard 0.876 vs Dropout+ModDrop 96.81% / 0.880 (+0.04 pp). Missing signals: left hand 89.09 → 91.87; right hand 81.25 → 85.36; both hands 53.13 → 73.28. On MNIST quarters (Table 8), ModDrop leaves clean error unchanged (1.02 → 1.02) and with 2 segments covered cuts error from 35.91% to 7.19%. The authors say ModDrop works "while not affecting the overall performance". — [ModDrop, TPAMI](https://chriswolfvision.github.io/www/papers/pami2015.pdf), Tables 8–9
- **UniBEV (LiDAR+camera 3D detection, nuScenes).** Modality dropout with p_md = 0.5 (one modality dropped) and p_L = p_C = 0.5, i.e. 50% L+C, 25% L-only and 25% C-only iterations. Table I, UniBEV with MD: L+C 64.2 mAP / 68.5 NDS, L-only 58.2 mAP, C-only 35.0 mAP. BEVFusion with MD: L+C 58.7, L-only 49.1, C-only 22.6 mAP. Single-modality references without MD: CenterPoint (L) 57.0 mAP and BEVFormer-S (C) 40.9 mAP. UniBEV's L-only inference (58.2) slightly beats the separately trained LiDAR-only model (57.0). The paper also notes that training with 100% LiDAR-only drops during MD *decreases* both L-only and L+C performance compared with including 25% camera-only iterations. No direct "same model without MD" full-modality comparison is given. — [UniBEV, arXiv 2309.14516](https://arxiv.org/abs/2309.14516), Table I and text (via fetch summary)
- **ConD (condition dropout) for RGB-D segmentation**, Table I, mIoU, baseline → +ConD:
  - NYUv2 full modality: DFormer-B 55.6 → 55.7; Sigma-S 56.6 → 57.2.
  - SUN RGB-D full modality: DFormer-B 51.2 → 51.4; Sigma-S 51.9 → 52.9.
  - Single-input evaluation, NYUv2: "RGB-only" DFormer-B 25.5 → 43.5, Sigma-S 13.0 → 41.5; "Depth-only" DFormer-B 35.2 → 48.3, Sigma-S 50.3 → 52.9.
  - Average missing-modality drop on NYUv2 goes from −25.3 to −9.8 mIoU.
  - So the full-modality gain is +0.1 to +1.0 mIoU.
  - Caveat: the labels "RGB-only" and "depth-only" are as reported by the fetch summary. The Sigma-S depth-only baseline of 50.3 looks anomalous and should be checked in the PDF. — [ConD, arXiv 2607.20326](https://arxiv.org/abs/2607.20326), Table I
- **CMNeXt / DeLiVER** tests sensor failures (motion blur, over/under-exposure, LiDAR jitter, event low-resolution) and missing modalities. A later benchmark reports that CMNeXt drops from 66.30 (RGB-D-E-L) to 48.77 when RGB is removed (DEL), i.e. CMNeXt trained *without* modality dropout collapses when its dominant modality is missing. **[search-snippet-only]** — [Benchmarking Multi-modal Semantic Segmentation under Sensor Failures, arXiv 2503.18445](https://arxiv.org/abs/2503.18445); [DeLiVER](https://arxiv.org/abs/2303.01480)
- **Input Dropout (spatially aligned modalities):** stochastically hiding extra modalities (depth, thermal) during training while testing on RGB only improves RGB-only performance on dehazing, 6-DoF tracking, pedestrian detection and classification. **[abstract-only, no numbers]** — [Input Dropout, arXiv 2002.02852](https://arxiv.org/abs/2002.02852)
- Search-level summaries also claim that modality dropout "can regularize even full-modality performance" in medical segmentation. These come from an aggregator, so the primary numbers were not verified. — [emergentmind topic page](https://www.emergentmind.com/topics/modality-dropout) (aggregator, low weight)

### Inferences
- For PoleSight, branch-level modality dropout (zeroing the intensity branch, or sometimes the range branch, during training) is low-risk. Expect roughly 0 to +1 mask mAP on full input. Its main benefit is that the fused model degrades gracefully to range-only behaviour, which directly addresses "fusion < best single modality".
- UniBEV's observation (100% dropping of the weak modality is worse than a mix) suggests not dropping only intensity. Keep some range-dropped iterations so the intensity branch learns its own features. That matters for the one class where intensity wins.
- Early-fusion channel stacking (one YOLO with a 2- or 4-channel input) can use channel-level input dropout (Input Dropout) as the analogue.

### Gaps
- No modality-dropout study on LiDAR range+intensity channels, or on YOLO-style instance segmentation, was found.
- UniBEV gives no same-architecture with/without-MD full-modality ablation, so the full-modality effect of MD in 3D detection is inferred, not measured.

---

## 3. Cross-modal knowledge distillation (fused teacher → single-modality student; 2DPASS, U2MKD, etc.): evidence on LiDAR

### Takeaway
Distilling from a fused (camera+LiDAR) teacher into a LiDAR-only student gives large, verified gains for LiDAR segmentation. 2DPASS reports +3.7 mIoU on SemanticKITTI val, +5.5 test, +3.2 nuScenes test, with no inference cost. On nuScenes, the LiDAR-only student even beats an explicit L+C fusion network (PMF). The gain comes mostly from the *fused* teacher's predictions; plain feature alignment adds under 1 mIoU. This is the best-evidenced way to obtain "fusion-level" accuracy from the stronger single modality.

### Cited Findings
- **2DPASS Table 5** (SemanticKITTI val, mIoU): baseline 65.58; + KL feature alignment 66.34; + fusion-to-single (modality fusion) 69.13; + 2D learner 69.32. The authors: "simply using feature alignment between two modalities cannot effectively improve the result… improvement mainly comes from the knowledge provided by the stronger fusion prediction." — [2DPASS, ECCV 2022, arXiv 2207.04397](https://arxiv.org/abs/2207.04397), Table 5
- **2DPASS Table 4**, distillation variants on SemanticKITTI val: Hinton KD 66.34; Huang et al. 66.46; Yang et al. 66.75; xMUDA cross-modal alignment 67.88; 2DPASS 69.32. Standard KD gives only +0.8–1.2 mIoU over the 65.58 baseline. — [2DPASS](https://arxiv.org/abs/2207.04397), Table 4
- **2DPASS Table 1** (SemanticKITTI test), baseline → 2DPASS: mIoU 67.4 → 72.9. Per class:
  - pole 63.7 → 65.0; traffic-sign 70.2 → 70.4
  - bicycle 51.1 → 63.6; truck 54.9 → 61.1; motorcyclist 30.3 → 74.1
  - Gains are largest on small or rare dynamic classes and marginal on poles and signs.
  - Within 10 m, val mIoU rises from 61.2 to 89.1 (Fig. 6a). — [2DPASS](https://arxiv.org/abs/2207.04397), Table 1, Fig. 6
- **2DPASS on nuScenes test (Table 3):** LiDAR-only baseline 77.6 → 2DPASS 80.8 mIoU. Fusion methods in the same table: PMF (L+C) 77.0 and 2D3DNet (L+C) 80.0. A LiDAR-only distilled student beats both fusion networks at 44 ms. Generality (Fig. 6b): MinkowskiNet 63.1 → 66.2 and SPVCNN 63.8 → 66.9 on SemanticKITTI val. — [2DPASS](https://arxiv.org/abs/2207.04397), Table 3, Fig. 6b
- **2DPASS supplementary Table 1** (nuScenes val), naive fusion vs distillation:
  - PointPainting-FCN 76.54 / DeepLabV3 76.56; multi-branch logit concat 77.25; multi-branch with interaction 79.12, all at about 2.3–3.3 s per frame.
  - Light baseline 76.04 → 2DPASS-light 78.87 at 40 ms.
  - Authors: naive combination "cannot improve the segmentation results obviously". — [2DPASS](https://arxiv.org/abs/2207.04397), Supp. Table 1
- **U2MKD (TPAMI 2024):** uni-modal teacher → multi-modal student distillation with bidirectional LiDAR-camera fusion and imputation of missing image features. Reports +8.3 mIoU over the LiDAR-only baseline on nuScenes val, plus SOTA on nuScenes, Waymo and SemanticKITTI. **[abstract-only]** — [U2MKD, IEEE TPAMI, DOI 10.1109/TPAMI.2024.3451658](https://doi.org/10.1109/TPAMI.2024.3451658)

### Inferences
- PoleSight analogue: train a range+intensity fused teacher, then distil its logits or masks (multi-scale) into a range-only YOLO-seg student. This keeps range-only deployment and Ultralytics' 3-channel pretrained path while importing intensity information. The 2DPASS ablation says to distil from the *fused* teacher's predictions, not just to align intensity features with range features.
- 2DPASS's per-class gains on poles and signs were small (+1.3, +0.2 test IoU), with the large gains on small dynamic classes. Expect modest gains on pole classes unless intensity carries class-discriminative cues (e.g., the one class where intensity wins).

### Gaps
- No distillation study between two channels of the same LiDAR (intensity ↔ range) was found.
- U2MKD's ablation tables were not retrieved; only the abstract claim was.

---

## 4. Input encoding of depth / LiDAR / 16-bit sensor values for pretrained CNNs (HHA vs raw vs jet, log/inverse range, percentile clipping, histogram equalisation)

### Takeaway
With ImageNet-pretrained CNNs:
- **Raw replicated depth works but is weakest.** Single-channel depth replicated to 3 channels transfers surprisingly well (Gupta: 11.3 AP without fine-tuning, 20.1 with) but trails encoded inputs.
- **Encodings beat raw depth by 1.5–5 points.** On detection, HHA beats fine-tuned disparity by 5.1 AP (25.2 vs 20.1, +25% relative). On classification, jet, normals and HHA beat grey depth by 1.0–2.7 pp.
- **Jet beats HHA when HHA's height channel carries no information.**
- **For 14/16-bit data, how you squeeze to 8 bits is a large effect.** Feeding raw 14-bit thermal loses 5–27 mAP against any 8-bit rescaling. No fixed rescaling (min-max, 1–99% clip, histogram equalisation, AGC) wins on every detector: spreads are up to 12.8 mAP on YOLOX.
- **Multi-channel encodings help.** Stacking several tone-mappings as channels (multi-channel "thermal embedding") beats any single 8-bit tone-map for 2 of 3 detectors.
- **Bit depth itself matters.** Keeping 13-bit rather than 8-bit satellite imagery gave >32% detection improvement.
- **No LiDAR-range-image study measured log/inverse range vs linear vs histogram equalisation.**

### Cited Findings
- **Gupta et al. ECCV 2014, Table 2** (NYUD2 val detection, mean AP):
  - RGB CNN without fine-tuning 16.4; RGB fine-tuned 19.7.
  - Disparity replicated ×3 into the RGB net, no fine-tuning, 11.3; disparity fine-tuned from ImageNet 20.1.
  - HHA fine-tuned 25.2; HHA + 2× synthetic data 26.1; HHA + 15× synthetic 25.6.
  - RGB+HHA (late feature combination) 32.5.
  - Method notes: for disparity they "replicate each one-channel disparity image three times… and scaled the input so as to have a distribution similar to RGB images". They found "it was always better to finetune from the ImageNet initialization than to train starting with a random initialization". — [Gupta et al., arXiv 1407.5736](https://arxiv.org/abs/1407.5736), Table 2 and Sec. 3.2
- **Eitel et al. IROS 2015, Table III** (Washington RGB-D Object, depth-only CaffeNet accuracy, 10 splits):
  - Depth-gray single channel from scratch 80.1 ± 2.6; depth-gray fine-tuned 82.0 ± 2.8.
  - Surface normals 84.7 ± 2.3; HHA 83.0 ± 2.7; depth-jet 83.8 ± 2.7.
  - Fusion (Table I): RGB 84.1, depth-jet 83.8, Fus-CNN (jet) 91.3 ± 1.4, Fus-CNN (HHA) 91.0 ± 1.9. Surface-normal fusion gave 91.1 ± 1.6, no gain over jet.
  - Jet is "normalize all depth values to lie between 0 and 255 … apply a jet colormap". HHA underperforms here because turntable objects all sit at the same height, so the height channel is uninformative. — [Eitel et al., arXiv 1507.06821](https://arxiv.org/abs/1507.06821), Tables I and III, Sec. IV-D
- **(DE)²CO, learned depth colourisation:** a residual network learns the depth → 3-channel mapping. Reported up to 16% gain over prior depth-only results on Washington, JHUIT-50 and BigBIRD. ColorJet and SurfaceNormals were the strongest hand-crafted mappings. **[abstract/search-snippet-only]** — [Carlucci et al., arXiv 1703.10881](https://arxiv.org/abs/1703.10881)
- **Thermal Chameleon (RA-L 2024), Table II**, FLIR-ADAS detection from RAW 14-bit thermal, mAP. Baselines replicate the single channel to 3 for ImageNet init.

  | Input | RetinaNet | YOLOX | Sparse-RCNN |
  |---|---|---|---|
  | RAW | 20.1 | 37.9 | 12.3 |
  | FLIR AGC | 25.2 | 53.5 | 37.5 |
  | 1–99% clip | 24.1 | 49.1 | 38.4 |
  | Histogram equalisation | 23.4 | 40.7 | 39.1 |
  | Min-max | 22.3 | 50.8 | 37.9 |
  | Thermal embedding TE(3) | 29.1 | 53.3 | 41.6 |
  | Learned TCNet | 34.7 | 56.0 | 45.7 |

  Authors: "direct use of RAW images worsened performance compared to 8-bit rescaled images". The best fixed tone-map differs by detector. On the STheReO dataset (Table III, YOLOX) HE gives 45.3, min-max 46.5 and TCNet 48.3. — [Thermal Chameleon, arXiv 2410.18340](https://arxiv.org/abs/2410.18340), Tables II–III
- **Overhead Detection: Beyond 8-bits and RGB:** R-FCN building detection on SpaceNet improves by "over 32%" using 13-bit rather than 8-bit 3-band data. "No significant performance improvement when adding additional bands." **[abstract-only]** — [arXiv 1808.02443](https://arxiv.org/abs/1808.02443)
- **Dai et al. (ACRA 2018), LiDAR-only YOLO from upsampled Velodyne images.**
  - Channels: R = intensity, G = inverse depth, B = depth, chosen "after a qualitative analysis" and used "without histogram equalisation". Per-channel histogram equalisation was optional.
  - KITTI 2D (person/vehicle): LiDAR 81.27% mAP vs RGB 80.67% mAP with ImageNet-pretrained YOLO (Table 2).
  - No quantitative comparison of encodings is given. — [Dai, Le Gentil, Vidal-Calleja, UTS OPUS PDF](https://opus.lib.uts.edu.au/rest/bitstreams/9c4b4e78-897e-452c-ac3f-39ce4ec2aafc/retrieve), Sec. 4–6 and Table 2
- **LiDAR intensity in point-cloud networks:** intensity ranges differ by sensor (0–255, 0–3000, 0–4000). It is commonly min-max normalised to [0,1]; for some datasets (ARVC, NCLT) MinkUNeXt-SI uses histogram equalisation. **[search-snippet-only, no ablation numbers seen]** — [MinkUNeXt-SI, arXiv 2505.17591](https://arxiv.org/abs/2505.17591)
- **DS-RangeNet (Electronics 2026):** splitting a 16-channel range image into an intensity stream and a geometry stream reportedly beats intensity-only by +5.7 pp and geometry-only by +4.2 pp mIoU (so geometry-only > intensity-only by about 1.5 pp), on industrial indoor scans. **[search-snippet-only; publisher page returned 403]** — [DS-RangeNet, doi 10.3390/electronics15173983](https://doi.org/10.3390/electronics15173983)
- **MultiMAE standardises depth "in a robust manner"** (robust per-image standardisation) and sets invalid depth to 0 before fine-tuning. This normalisation is not ablated. — [MultiMAE, arXiv 2204.01678](https://arxiv.org/abs/2204.01678), Sec. 4.3

### Inferences
- PoleSight renders 16-bit range/intensity to 8-bit PNG for Ultralytics, so the rendering function is a first-order hyper-parameter. The thermal results show 5–27 mAP swings between raw and rescaled inputs, and up to 12.8 mAP between fixed rescalings on one detector. Worth testing for both range and intensity:
  - linear min-max over a fixed sensor range
  - per-image or per-dataset 1–99% percentile clip
  - log or inverse range (Dai et al.'s G channel)
  - histogram equalisation or CLAHE
  - a 3-channel stack of different encodings of the *same* signal (e.g., range, inverse range, log range), analogous to Thermal Chameleon's TE(3) and Dai et al.'s intensity/inv-depth/depth stack
- The stacked-encoding option keeps channels = 3, so Ultralytics transfers the pretrained first conv. It is the cheapest way to add information without losing pretrained stems.
- The best fixed encoding was detector-dependent in Thermal Chameleon, so expect encoding rankings to differ between YOLO variants. Evaluate on more than one model.
- HHA-style geocentric channels (height, angle to gravity) are computable from LiDAR xyz. The evidence: +5.1 AP for detection when height is informative (Gupta), but no gain when height is constant (Eitel). Pole classes are vertical structures, so height and normal angle are plausibly informative. This is a hypothesis, not measured.

### Gaps
- No paper was found that measures log vs inverse vs linear range, or percentile clipping vs histogram equalisation, for **LiDAR range/intensity images** with pretrained 2D detectors or segmenters. The closest measured evidence is thermal (14-bit) and RGB-D depth.
- Colour maps such as jet are measured only for Kinect depth classification (Eitel; DE²CO). Nothing was found for LiDAR range-image detection or segmentation.
- The DS-RangeNet numbers are unverified (403 on publisher page).

---

## 5. First-layer initialisation and pretraining for 1-, 2- or 4-channel non-RGB inputs (replication, mean init, separate stems, depth-specific pretraining: DFormer, MultiMAE, Omnivore)

### Takeaway
- **ImageNet init beats random init even for depth and LiDAR range images**, but the gains are modest: +0.5 to +2.8 mIoU for range-view LiDAR segmentation (RangeFormer, RangeViT).
- **Modality-matched pretraining beats RGB pretraining for the non-RGB branch by a large margin.** On depth-only input, a backbone pretrained on depth beats one pretrained on RGB by 15.2 mIoU (DFormer). RGB-D pretraining beats RGB-pretrained with replicated depth by 2.3 mIoU in fusion.
- **Bolting depth onto an RGB-only-pretrained model can make fusion worse than RGB alone.** MAE RGB-D 49.3 vs RGB 50.8 on NYUv2. Multimodal pretraining (MultiMAE) turns this into a +4.0 gain.
- **A learned convolutional stem in front of an image-pretrained trunk is a large, verified win for range images** (+4.3 mIoU in RangeViT).
- **No paper was found that directly compares channel replication vs mean-of-RGB-filter init vs random init for an extra first-layer channel on detection or segmentation.**

### Cited Findings
- **RangeFormer (ICCV 2023), Table 6:** range-view LiDAR segmentation (SemanticKITTI val 64×2048 / nuScenes val 32×1920, mIoU), no pretraining → ImageNet:
  - FIDNet 60.4 → 61.6 (+1.2) / 71.4 → 72.1 (+0.7)
  - CENet 63.4 → 64.1 (+0.7) / 73.3 → 73.9 (+0.6)
  - RangeFormer 68.1 → 68.9 (+0.8) / 77.1 → 77.6 (+0.5); Cityscapes pretraining 69.6 (+1.5) / 78.1 — [Kong et al., arXiv 2303.05367](https://arxiv.org/abs/2303.05367), Table 6
- **RangeViT (CVPR 2023), Table 4** (nuScenes val, ViT-S/16 encoder; stem, decoder and refiner always randomly initialised): random 72.37; DINO IN-1k 73.33; IN21k supervised 74.77 (+2.4); IN21k→Cityscapes 75.21 (+2.8). "Despite the large domain gap, using ViT models pre-trained on RGB images is always better than training from scratch." Pretrained init also converges faster: Fig. 3 shows reaching the same mIoU with 63% fewer epochs. — [RangeViT, arXiv 2301.10222](https://arxiv.org/abs/2301.10222), Table 4, Fig. 3
- **RangeViT Table 1** (stem design for range images, nuScenes val): linear patch-embedding stem + linear decoder 65.52 → convolutional stem 69.82 (+4.3) → + UpConv decoder 73.83 → + 3D refiner 74.60. The conv stem "can effectively steer the distribution of input range images towards the image-based feature distribution the ViT has been pre-trained on". Rectangular patches suit wide range images (Table 3: 16×16 68.45 vs 2×8 75.21). — [RangeViT](https://arxiv.org/abs/2301.10222), Tables 1 and 3
- **DFormer (ICLR 2024), Table 12:** same 11.2M backbone pretrained on ImageNet RGB vs on ImageNet *depth maps* (estimated), evaluated on depth-only NYUv2 segmentation: 27.6 vs 42.8 mIoU (+15.2). — [DFormer, arXiv 2309.09668](https://arxiv.org/abs/2309.09668), Table 12 (appendix)
- **DFormer Table 3:** fusion-model pretraining "RGB+RGB" (depth stem changed from 1 to 3 channels, depth replicated ×3 at fine-tuning) 53.3 vs RGB+D pretraining 55.6 mIoU on NYUv2 (+2.3). — [DFormer](https://arxiv.org/abs/2309.09668), Table 3, Sec. 4.3
- **MultiMAE (ECCV 2022), Table 2** (NYUv2 segmentation with ground-truth depth, mIoU; ViT-B):
  - MAE (RGB-only pretrain): RGB 50.8 / depth-only 23.4 / RGB-D 49.3. Adding depth makes it *worse* than RGB alone.
  - MultiMAE (RGB+pseudo-depth+pseudo-seg pretrain): RGB 52.0 / depth-only 41.4 / RGB-D 56.0.
  - Hypersim: MAE RGB-D 36.9 vs MultiMAE 47.6.
  - Authors: "The standard MAE … is not able to sufficiently make use of the additional depth, since it was only trained on RGB images." — [MultiMAE, arXiv 2204.01678](https://arxiv.org/abs/2204.01678), Table 2 and Sec. 4.3
- **Gupta et al.:** an RGB-pretrained CNN fed replicated disparity reaches 11.3 AP without any fine-tuning, and fine-tuning from ImageNet always beat random init for depth. — [arXiv 1407.5736](https://arxiv.org/abs/1407.5736), Sec. 3.2
- **Eitel et al.:** single-channel depth from scratch 80.1 vs fine-tuned from ImageNet (grey replicated) 82.0, i.e. +1.9 pp for ImageNet init. — [arXiv 1507.06821](https://arxiv.org/abs/1507.06821), Table III
- **First-layer extension practices** (methods, not measured comparisons):
  - Initialise extra channels with the average of the RGB filters, or replicate the red-channel filters (multispectral).
  - Randomly initialise the new depth filters while keeping the RGB ones.
  - timm has a pull request to support non-RGB pretrained input-channel transfer.
  - One summary says replicating RGB weights across bands "outperformed random initialization". This is from the search snippet; the primary table was not seen. — [search results incl. Multi³Net arXiv 1812.01756](https://arxiv.org/abs/1812.01756); [timm PR #2780](https://github.com/huggingface/pytorch-image-models/pull/2780)
- Omnivore was not fetched in this session. No numbers are recorded for it.

### Inferences
- **PoleSight / Ultralytics.** With channels != 3, Ultralytics silently drops the pretrained first conv. The rest of the network is still pretrained, but the stem is random, which is the same situation as RangeViT's random conv stem in front of a pretrained trunk. RangeViT suggests this costs little if the stem is trained. Gupta/Eitel suggest the alternative, replicating 1-channel range to 3 channels and keeping the pretrained stem, is a strong baseline. A cheap, low-risk 2/4-channel option is a custom first-conv init: copy pretrained weights, set new-channel weights to the mean of the RGB filters, and rescale so activations keep their magnitude. No measured detection or segmentation comparison against random init was found, so PoleSight would be producing new evidence here.
- **Late fusion is the riskiest route.** MAE→MultiMAE and DFormer show that bolting a second modality onto an RGB-only-pretrained network can make the fused model *worse* than the RGB-pretrained single modality (49.3 < 50.8). For PoleSight, a late-fusion model with an intensity branch initialised from COCO RGB weights is the configuration most at risk of failing to beat range-only. Options:
  - pretrain each branch on its own modality first (e.g., train range-only and intensity-only YOLO-seg models, then initialise the fusion branches from those)
  - use the 3-channel encoding trick from section 4 so both branches keep COCO stems
- **Expected size of the pretraining effect.** ImageNet pretraining gains for range images are modest (+0.5 to +2.8 mIoU) next to the gains from augmentation (section 6) and stem design.

### Gaps
- No controlled comparison of channel replication vs mean-filter init vs random init for an *extra* (4th) first-layer channel on detection or segmentation was found.
- Nothing was found on pretraining specifically on LiDAR range/intensity images and then transferring to 2D detection.
- Omnivore numbers were not collected.

---

## 6. Range-image augmentation (LaserMix, PolarMix, RangeMix / RangeAug, instance copy-paste in range view): measured gains, especially on small classes

### Takeaway
Mixing-based LiDAR augmentation gives some of the largest gains in this whole topic. Range-view RangeAug adds +6.2 to +10.5 mIoU over plain training and +2.5 to +7.4 over standard augmentation. Instance paste and rotate-paste give most of the small-class gains (bicycle +42 to +48 IoU with PolarMix vs no or global-only augmentation). Pole and traffic-sign IoU gains are smaller (about +1 to +3 fully supervised, about +6 with very few labels). For a small dataset like PoleSight (870 images), augmentation is likely a bigger lever than the choice of fusion operator.

### Cited Findings
- **RangeFormer, RangeAug** = RangeMix (row/inclination-band mixing between scans) + RangeUnion (fill empty grid cells from another scan) + RangePaste (copy tail-class pixels from another scan) + RangeShift (azimuthal roll). Probabilities [0.9, 0.2, 0.9, 1.0]. Fig. 5b (SemanticKITTI val, 64×512 input, mIoU; Plain / Common / Mix3D / RangeAug):
  - FIDNet 50.8 / 53.9 / 54.5 / 61.3
  - CENet 56.2 / 59.9 / 59.6 / 62.4
  - RangeFormer 56.0 / 61.9 / 61.5 / 66.0
  - "The extra overhead needed for RangeAug is negligible on GPUs." — [RangeFormer, arXiv 2303.05367](https://arxiv.org/abs/2303.05367), Fig. 5b **[read from figure]**, Sec. 3 and 4.3
- **PolarMix (NeurIPS 2022), Table 1** (SemanticKITTI val, mIoU), relative to no augmentation:
  - MinkNet: none 55.9; +CGA (global scaling/rotation) 58.9; +CutMix 60.6; +CopyPaste 62.4; +Mix3D 62.4; +PolarMix 65.0 (+9.1).
  - SPVCNN: none 58.0; +CGA 60.7; +PolarMix 66.2 (+8.5).
  - Small classes, MinkNet none → CGA → PolarMix: bicycle 3.7 → 8.7 → 51.2; motorcycle 44.9 → 52.3 → 75.6.
  - Pole 62.1 → 62.8 → 64.6; traffic-sign 43.7 → 46.8 → 49.9. — [PolarMix, arXiv 2208.00223](https://arxiv.org/abs/2208.00223), Table 1
- **PolarMix Table 2** (mIoU): nuScenes-lidarseg MinkNet 67.1 → 72.0, SPVCNN 68.4 → 72.1; SemanticPOSS MinkNet 52.1 → 57.4, SPVCNN 50.7 → 58.6. — [PolarMix](https://arxiv.org/abs/2208.00223), Table 2
- **PolarMix Table 6** (SPVCNN trained on sequence 00 only, a small-data regime): baseline 48.9; scene-level swap 50.8; simple instance paste 50.9; rotate-paste (multiple rotated copies) 53.2; full 54.8 (+5.9). Rotated multi-copy instance paste is worth about twice simple paste. — [PolarMix](https://arxiv.org/abs/2208.00223), Table 6
- **LaserMix (CVPR 2023), Table 1**, semi-supervised, range-view FIDNet, sup-only → LaserMix (mIoU):
  - SemanticKITTI 1% 36.2 → 43.4, 10% 52.2 → 58.8, 20% 55.9 → 59.4, 50% 57.2 → 61.4
  - nuScenes 1% 38.3 → 49.5, 10% 57.5 → 68.2, 50% 67.6 → 73.0
  - Range view benefits more than voxel (+5.4 to +11.2 vs +2.0 to +5.2). — [LaserMix, arXiv 2207.00026](https://arxiv.org/abs/2207.00026), Table 1
- **LaserMix Table 9** (SemanticKITTI 1%, range view, sup-only → LaserMix IoU): pole 52.5 → 59.0; traffic-sign 35.7 → 41.7; bicycle 0.6 → 37.1; bicyclist 6.6 → 40.7. — [LaserMix](https://arxiv.org/abs/2207.00026), Table 9
- **RangeViT augmentation** (for reference): y-axis flips, random translation, ±5° rotations (p = 0.5 each), then a random crop to 32×384 (nuScenes) or 64×384 (SemanticKITTI) of the 2048-wide range image. Not ablated. — [RangeViT](https://arxiv.org/abs/2301.10222), Sec. 4.1
- **Point-cloud 3D augmentation gains are additive to range-view training.** RangeFormer's "Common" set already gives +3.1 to +5.9 over plain; mix-style RangeAug adds a further +2.5 to +7.4. — [RangeFormer](https://arxiv.org/abs/2303.05367), Fig. 5b **[read from figure]**

### Inferences
- PoleSight's images are fixed 1024×128 range/intensity renderings, not raw point clouds. The 2D-applicable analogues are:
  - **RangeShift:** horizontal roll of the 360° image, label-preserving and trivially implementable.
  - **RangeMix:** swap horizontal bands (beam rows) or azimuth sectors between two scans. This is a PolarMix scene-swap in image form.
  - **RangePaste:** copy pole-instance masks between scans at the *same rows*, so pixel-registered range and intensity stay geometrically plausible. Paste range and intensity together to keep channel alignment.
  - **Random width crops.**
- Standard YOLO augmentations (HSV jitter, mosaic, perspective, vertical flip) can break range semantics. HSV jitter on a range PNG changes "distance". This is an inference; no paper measured it.
- Pole IoU gains from mixing are smaller than for bicycles and motorcycles, about +2 to +3 IoU fully supervised. Copy-paste of rare pole classes is still the most targeted tool for class imbalance among the 4 pole classes.

### Gaps
- No paper measured range-view augmentation on 2D *instance* segmentation or detection (YOLO-style) of range images, or with intensity and range as separate fusion branches.
- RangeFormer's per-component RangeAug ablation (each of Mix/Union/Paste/Shift separately) was not found in the main text. Only the combined effect (Fig. 5b) was extracted.
- No study measured photometric augmentation (brightness/contrast jitter) on LiDAR intensity or range images.
