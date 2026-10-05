# Architectures for tiny, thin, elongated objects in low-resolution, wide-aspect images (LiDAR range view): detection, instance segmentation, and how they interact with multimodal fusion

Scope note. These notes were written for PoleSight: 1024x128 range images, 98% of instances below COCO small (<32² = 1024 px²), median mask 71 px², median box h/w 2.1, and a mask that fills 0.35 of its box. The current baseline is Ultralytics YOLOv8/11/26-seg at imgsz 640, which downscales frames to 640x80. Every number below names its paper and table. "Abstract-only" or "search-snippet-only" marks a claim I could not check against the full text. Where a number came through a web-page summarizer rather than the PDF text, the entry says so.

Quick geometry check (my arithmetic, not from a source). A 71 px² mask at fill 0.35 gives a box of about 203 px². With h/w = 2.1, that is about 9.8 px wide by 20.6 px tall at native resolution. Downscaling to 640x80 multiplies both sides by 0.625, giving about 6.1 x 12.9 px. That box covers about 0.8 x 1.6 cells of a stride-8 (P3) grid and about 1.5 x 3.2 pixels of a stride-4 mask prototype. The pole shaft is narrower than the box (fill 0.35), so it is probably 1–2 px wide after downscaling, which is less than one prototype pixel.

## Q1. Which architectures do best on tiny-object detection and instance-segmentation benchmarks, and why? (YOLO-seg vs Mask R-CNN/Cascade vs Mask2Former/Mask DINO/RT-DETR-seg)

### Takeaway
No detector family wins on tiny objects in every benchmark. Older query-based models (Deformable DETR, Mask2Former) are clearly weaker on tiny or small objects than anchor-based two-stage CNNs (Cascade R-CNN, Mask R-CNN). Modern dense-query DETRs (DINO with 900 queries) and tiny-specific DETRs (DQ-DETR) now beat two-stage CNNs on AI-TOD-V2. Across benchmarks, three things predict small-object accuracy better than the detector family: how labels are assigned (IoU vs a distance-based metric), how many queries or anchors cover each object, and the effective feature/mask resolution. I found no peer-reviewed, same-protocol comparison of YOLO-seg against Mask R-CNN or Mask2Former on a tiny-object instance-segmentation benchmark.

### Cited Findings
**SODA-D (driving, small objects; size bins eS 0–144 px, rS 144–400, gS 400–1024, Normal 1024–2000 px area)**
- Cascade R-CNN is best: 31.2 AP, 59.9 AP50, 27.8 AP75, 14.1 AP_eS, 27.5 AP_rS, 37.1 AP_gS, 46.9 AP_N. Deformable-DETR gets 19.2 AP, 6.3 AP_eS, 15.4 AP_rS, 24.9 AP_gS, 34.2 AP_N. Faster R-CNN gets 28.9 AP and 13.9 AP_eS; CenterNet 21.5 AP and 5.1 AP_eS; CornerNet 24.6 AP and 6.5 AP_eS; RFLA (a tiny-object-specific method) 29.7 AP. Source: SODA benchmark table — [Cheng et al., "Towards Large-Scale Small Object Detection: Survey and Benchmarks" (TPAMI 2023)](https://arxiv.org/html/2207.14096). Numbers came from a search snippet plus an HTML summary of the benchmark table; I could not get the table number.
- The authors attribute the DETR gap to "the sparse query paradigm [which] could not cover small objects adequately". They also find that query-based and anchor-free methods "significantly lag behind anchor-based methods, particularly for extremely small and relatively small sizes", and that "deeper models might not be better for size-limited objects": ResNet-101 sometimes scored below ResNet-50, and ConvNeXt-T was best. — [SODA paper](https://arxiv.org/html/2207.14096)

**AI-TOD (aerial; mean absolute object size 12.8 px, against 99.5 px for COCO; size bins very tiny 2–8 px, tiny 8–16, small 16–32, medium 32–64)**
- Table 4 of the NWD paper (AI-TOD test, ResNet-50 FPN, 12 epochs). The IoU-based anchor detectors all score **AP_vt = 0.0**: Faster R-CNN (11.1 AP), Cascade R-CNN (13.8 AP), DetectoRS (14.8 AP). Replacing IoU with the Normalized Gaussian Wasserstein Distance (NWD) gives Faster R-CNN* 17.8 AP / 2.5 AP_vt / 17.0 AP_t, Cascade R-CNN* 18.7 / 3.6 / 17.4, and DetectoRS* 20.8 / 6.4 / 19.7. CenterNet (DLA-34), an anchor-free model, scores 13.4 AP with 3.8 AP_vt. FCOS scores 9.8 AP. — [Wang et al., "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection" (arXiv 2110.13389), Table 4](https://arxiv.org/pdf/2110.13389v1)
- The reason, quantified (NWD paper, Sec. 1). For a 6x6 px object, a small location shift drops IoU from 0.53 to 0.06. For a 36x36 px object, the same shift drops IoU only from 0.90 to 0.65. Under default IoU thresholds, each ground-truth box gets on average 0.72 positive anchors with IoU, 0.19 with DIoU/CIoU, and 1.05 with NWD (Sec. 4.1, Table 1 discussion). — [NWD paper](https://arxiv.org/pdf/2110.13389v1)

**AI-TOD-V2: query-based models (DQ-DETR, ECCV 2024, Table 2)**
- Faster R-CNN 12.8 AP / 0.0 AP_vt; Cascade R-CNN 15.1 / 0.1; DetectoRS 16.1 / 0.1; NWD-RKA 24.7 / 9.7; RFLA 25.7 / 9.2; Deformable DETR (300 queries) 18.9 / 6.5; DAB-DETR 22.4 / 9.0; **DINO-DETR (900 queries) 25.9 / 12.7**; **DQ-DETR 30.2 / 15.3** (AP_t 30.5, AP_s 36.5). — [Huang et al., DQ-DETR, Table 2](https://arxiv.org/html/2404.03507v2)
- DQ-DETR's explanation: fixed learned positional queries "do not have explicit physical meaning", and a fixed number of queries K gives "low recall rate" in dense images. DQ-DETR therefore changes K per image (300/500/900/1500) using a counting module and a density map. — [DQ-DETR](https://arxiv.org/html/2404.03507v2)

**COCO small objects, instance segmentation (Mask2Former, CVPR 2022, Table 2 / Appendix)**
- Mask R-CNN R50 (400 epochs, LSJ): AP 42.5, **AP_S 23.8**, AP_L 60.0. Mask2Former R50 (50 epochs): AP 43.7, **AP_S 23.4**, AP_L 64.8. R101: Mask R-CNN 43.7 / AP_S 24.6; Mask2Former 44.2 / AP_S 23.8. Swin-L: HTC++ 49.5 / AP_S 31.0; Mask2Former 50.1 / AP_S 29.9; QueryInst 48.9 / AP_S 30.8. Mask2Former wins on overall AP and large objects and loses on small objects. — [Cheng et al., Mask2Former](https://ar5iv.labs.arxiv.org/html/2112.01527)
- The authors write that "Mask2Former struggles with segmenting small objects and is unable to fully leverage multi-scale features" (Limitations). Mask2Former predicts masks at 1/4 stride and computes its mask loss on K=12544 sampled points (equivalent to 112x112). — [Mask2Former](https://ar5iv.labs.arxiv.org/html/2112.01527)

**iSAID (aerial instance segmentation)**
- PointRend reaches AP_S ≈ 18.87 with overall mask AP ≈ 34.35 on iSAID. — [iSAID-related comparison, search-snippet-only](https://doi.org/10.3390/math12182905). I did not verify the table.

**YOLO-seg vs Mask R-CNN**
- The only comparisons I found are application papers (PCB defects, microplastics) with non-standard protocols. One reports YOLO-seg stronger on AP_S/AP_M and Mask R-CNN stronger on AP_L. — [Comparison of Mask R-CNN and YOLOv8-seg, Sci. Rep. 2025 (search-snippet-only)](https://www.nature.com/articles/s41598-025-02131-7); [Microplastics study (search-snippet-only)](https://arxiv.org/pdf/2409.13688). This is low-quality evidence and should not be used as a benchmark claim.

### Inferences
- For PoleSight, objects at native resolution have a box-root-area of about 14 px. That falls in AI-TOD's "tiny" bin (8–16 px), and after downscaling to 640x80 it sits near the "very tiny" boundary. In that regime, IoU-based assignment of positives looks like a primary failure mode: AP_vt is 0.0 for three anchor-based detectors in NWD Table 4. The YOLOv8+ task-aligned assigner also uses CIoU, so the same sensitivity plausibly applies. This is my inference; it has not been measured on YOLO.
- Query-based models are not ruled out. DINO with dense queries beats Cascade and DetectoRS on AI-TOD-V2. Older DETRs and Mask2Former do underperform on small objects, so any DETR-family candidate should use dense queries (≥900) and stride-4 features or masks.
- Overall COCO AP is a poor proxy here. Mask2Former beats Mask R-CNN on AP yet loses on AP_S.

### Gaps
- I found no published head-to-head of YOLOv8/11-seg vs Mask R-CNN/Cascade vs Mask2Former/Mask DINO/RT-DETR-seg on a tiny-object **instance segmentation** benchmark (iSAID small, or an instance-mask variant of SODA/AI-TOD) under one protocol.
- I did not retrieve Mask DINO's or RT-DETR's AP_S tables, so no numbers are given for them.
- I could not verify the iSAID PointRend AP_S value against a full table, and I did not retrieve Mask R-CNN vs Cascade AP_S on iSAID.
- TinyPerson and VisDrone segmentation results were not found (both are detection-only benchmarks).

## Q2. What is the evidence for input upscaling, P2/stride-4 heads, high-resolution backbones, SAHI slicing, mask resolution, thin-structure losses, and anisotropic or aspect-aware design?

### Takeaway
The strongest and most transferable evidence is about effective resolution. Keeping or adding high-resolution features or inputs gives large gains: removing YOLOv5's Focus downsampling adds about +7–9 mAP50, an auxiliary super-resolution branch adds a further +8.5 mAP50, and SAHI slicing plus fine-tuning adds +12.7 to +14.5 AP50. Ultralytics YOLO-seg prototypes are fixed at imgsz/4, so structures thinner than about 4 input pixels cannot be represented by the mask head. Anisotropic or rectangular tokenization (patch 2x8) beats square tokenization on wide range images by about 6.8 mIoU. Thin-structure losses (clDice, Skeleton Recall) give gains of less than one Dice point in semantic segmentation. Distance-based box metrics (NWD) give much larger gains on tiny boxes.

### Cited Findings
**Downsampling, resolution, and super-resolution branches (SuperYOLO, IEEE TGRS 2023, VEDAI, mAP50)**
- Removing the Focus (space-to-depth stride-2) stem raised YOLOv5s from 62.2 to 69.5 (+7.3), YOLOv5m from 64.5 to 72.2 (+7.7), and YOLOv5l from 63.7 to 72.5 (+8.8). YOLOv5s GFLOPs rose from 5.3 to 20.4. — [Zhang et al., SuperYOLO, Table II](https://ar5iv.labs.arxiv.org/html/2209.13351)
- An auxiliary SR branch at 512x512 input raised YOLOv5s (noFocus) from 69.5 to **78.0** (+8.5) with identical inference parameters and GFLOPs (7.07 M, 20.4), because the SR branch is dropped at inference. — [SuperYOLO, Table IV](https://ar5iv.labs.arxiv.org/html/2209.13351). Table V reports 80.9 when the ablation adds a small-scale detector and an EDSR decoder with L1 loss. Final SuperYOLO is **75.09 mAP50 averaged over 10 folds**, with 4.85 M parameters and 17.98 GFLOPs. — [SuperYOLO, Tables V and VII](https://ar5iv.labs.arxiv.org/html/2209.13351)

**Slicing (SAHI, ICIP 2022; VisDrone Table 1 and xView Table 2; all AP50)**
- Sliced inference alone adds +6.8 AP (FCOS), +5.1 (VFNet), and +5.3 (TOOD). Adding slicing-aided fine-tuning brings the cumulative gains to +12.7, +13.4, and +14.5 AP — [Akyon et al., SAHI (abstract)](https://arxiv.org/abs/2202.06934)
- VisDrone Table 1, full pipeline vs baseline: FCOS 25.8→38.5 AP50 (AP50_s 14.2→25.9); VFNet 28.8→42.2 (AP50_s 16.8→29.6); TOOD 29.4→43.5 (AP50_s 18.1→31.7). Slices were 480–640 px for VisDrone and 300–500 px for xView, with 25% overlap. — [SAHI paper, Tables 1–2](https://ar5iv.labs.arxiv.org/html/2202.06934). Caveat: AP50_l also rose a lot (e.g., FCOS 45.1→59.8), so the gains are not confined to small objects.
- Range-view analogue of slicing (RangeFormer STR, ICCV 2023, Table 2, SemanticKITTI test). Training on native-resolution 64x384 azimuth crops ("views", Z=5) of the 64x2048 scan gives **72.2 mIoU**. Training on a scan downsampled to 64x512 gives 70.0, and full 64x2048 training gives 73.3. For CENet, STR training gives **65.8**, which beats CENet trained at full 64x2048 (64.7). FIDNet goes from 59.5 at 2048 to 60.1 with STR. — [Kong et al., RangeFormer, Table 2](https://arxiv.org/html/2303.05367v3). The authors explain that a low horizontal resolution W "intensifies the 'many-to-one' conflict, causes more severe shape distortions" (Sec. 3.3).

**P2 / stride-4 heads**
- LAF-YOLOv10 adds "an auxiliary P2 detection head at 160×160 resolution [that] extends localization to objects below 8×8 pixels" and removes P5. With all its changes together it reaches 35.1 mAP50 on VisDrone (+3.3 over YOLOv10n). It reports no P2-only ablation. — [LAF-YOLOv10 (abstract)](https://arxiv.org/abs/2602.13378)
- Other P2 ablations on VisDrone (e.g., YOLOv8s+P2: +3.2 AP50, +2.2 AP50-95) appeared only in search snippets, and I could not confirm which paper they come from. — [search results incl. HierLight-YOLO](https://arxiv.org/pdf/2509.22365); [Tencent YOLO-Master issue #98 (non-peer-reviewed): stride-4 head +2.2 mAP](https://github.com/Tencent/YOLO-Master/issues/98). Treat these as unverified.

**High-resolution backbones**
- HRDNet feeds a high-resolution input to a shallow backbone and a low-resolution input to a deep one, for small objects. — [HRDNet (abstract-only)](https://arxiv.org/pdf/2006.07607). HRNet/HRFPN-based UAV detectors report +4.4 mAP over YOLOv8-m on VisDrone. — [Sensors 2024 (search-snippet-only)](https://doi.org/10.3390/s24154966). Neither has been checked against a full table.

**Mask resolution in Ultralytics YOLO-seg**
- "The mask prototype tensor is always imgsz/4, whatever mask_ratio is set to". mask_ratio "changes how the targets are rasterised; it cannot raise the resolution the head is able to predict at". For 2-px bands slanted 10° at imgsz 640, "a third of the length of the band is gone before training has even started". The issue proposes adding P2 to the prototype branch and estimates 170.6 vs 535.7 GFLOPs for 2-px mask resolution. — [ultralytics/ultralytics issue #26436 (opened 2026-09-29, labelled enhancement/fixed)](https://github.com/ultralytics/ultralytics/issues/26436); docs clarification in [PR #26441](https://github.com/ultralytics/ultralytics/pull/26441)
- Default predicted masks are 160x160 at imgsz 640 and are upsampled in post-processing. `retina_masks=True` returns masks at original image size, but this is still upsampled from the same prototypes. — [Ultralytics predict docs discussion](https://github.com/orgs/ultralytics/discussions/7932) (search-snippet level)

**Anisotropic / aspect-aware design**
- RangeViT (CVPR 2023, Table 3, nuScenes val) patch-size ablation, mIoU: 16x16 68.45; 8x8 72.04; 4x16 72.72; 4x8 73.30; 4x4 73.70; 2x16 73.88; **2x8 75.21** (769 tokens, 1.43x training time). The authors write: "The wide range images benefit more from rectangular patches… smaller patches enable … more precise predictions for smaller and thinner objects". — [Ando et al., RangeViT, Table 3](https://arxiv.org/html/2301.10222v2)
- RangeViT's convolutional stem (the first 4 SalsaNext residual blocks at full resolution, then average-pooling to the patch grid) improved mIoU from 65.52 to 69.82 over a linear patch embedding. An UpConv decoder with skips from the stem brought it to 73.83, and a KPConv 3D refiner to 74.60 (Table 1). — [RangeViT, Table 1](https://arxiv.org/html/2301.10222v2)
- Strip R-CNN uses sequential orthogonal large strip convolutions for high-aspect-ratio objects. Its 30 M model reaches 82.75 mAP on DOTA-v1.0. — [Yuan et al., Strip R-CNN (AAAI 2026; abstract-only)](https://arxiv.org/abs/2501.03775)
- RangeSAM uses horizontally elongated attention windows (8x64 in stages 1 and 4, 16x128 in stages 2 and 3) and a stride-1 stem with overlapping 7x7 patches at 64x2048. It reports **no ablation** for either choice. — [RangeSAM (WACVW 2026)](https://arxiv.org/html/2509.15886v1)

**Losses and metrics for tiny and thin objects**
- NWD in place of IoU (AI-TOD): +4.5 AP for RetinaNet, +0.7 ATSS, +6.7 Faster R-CNN, +4.9 Cascade, +6.0 DetectoRS. Most of the gain comes from NWD in RPN label assignment (11.1 → 17.3 AP, Table 2). — [NWD paper, Tables 2 and 4](https://arxiv.org/pdf/2110.13389v1)
- SAFit (from the RGBT-Tiny benchmark) blends IoU and NWD with a size-dependent sigmoid weight (C = 32 px), so small objects are judged more by NWD. — [Ying et al., RGBT-Tiny (arXiv 2406.14482)](https://arxiv.org/html/2406.14482)
- clDice vs Dice baseline (nnU-Net, Skeleton Recall paper Table 2): Roads Dice 78.99→79.15 (clDice metric 88.79→89.00); DRIVE 80.87→81.05. Skeleton Recall loss gives similar or slightly better results (Roads 79.25 / 89.06). clDice adds about 88% training time and 52% VRAM and runs out of memory on multi-class TopCoW. Skeleton Recall adds about 8% time and 2% VRAM. — [Kirchhoff et al., Skeleton Recall Loss (ECCV 2024), Table 2 / Fig. 6](https://arxiv.org/html/2404.03010v1); original clDice: [Shit et al., CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.pdf)

### Inferences
- PoleSight's 640x80 training downscale works against every resolution finding above: SuperYOLO's Focus removal, RangeFormer's W ablation (Q3), the STR crops, and the YOLO-seg imgsz/4 prototype limit. With the shaft about 1–2 px wide after downscaling, the stride-4 prototype cannot represent it. The low mask mAP50-95 of about 8–12 may therefore be limited partly by mask resolution rather than by detection.
- Interventions in order of evidence strength and cost (inference; none has been tested on PoleSight):
  1. Train at native 1024x128 (rect, no downscale), or upscale ×2 to 2048x256 so a 2-px shaft maps to at least one stride-4 prototype pixel.
  2. Add P2 to both the detection heads and the prototype branch (issue #26436).
  3. Use native-resolution horizontal crops for training (STR/SAHI-style, e.g., 128x256 or 128x384 windows), with full-width or sliced inference.
  4. Use an NWD/SAFit-style assignment or regression term.
  5. Use anisotropic early downsampling (vertical stride 1) and horizontally elongated kernels or windows.
- An auxiliary SR branch (SuperYOLO-style, dropped at inference) is a cheap way to get high-resolution supervision when native upscaling is too costly.
- Thin-structure losses (clDice, Skeleton Recall) are designed for connected networks of vessels and roads. With masks of about 71 px², their benefit is probably small next to resolution fixes, but Skeleton Recall is cheap enough to try as an auxiliary mask loss.

### Gaps
- I found no study that adds an SR auxiliary branch, P2 prototypes, or NWD to a YOLO-**seg** (instance mask) model and reports mask AP on tiny objects.
- I did not retrieve boundary-loss (Kervadec et al.) results on thin structures.
- I could not verify a clean P2-only ablation from a peer-reviewed source.
- Evidence for HRNet as a detector backbone on tiny objects is snippet-level only.

## Q3. How do range-view LiDAR networks handle low vertical resolution, and what gains are measured for pole, sign, and trunk classes?

### Takeaway
Range-view networks either never downsample vertically or keep a full-resolution stem and cap total stride at 8. RangeNet++ downsamples only horizontally. CENet and RangeFormer use symmetric stride 2 but max stride 8 and a stride-1 stem; CENet also upsamples every scale back to full resolution. RangeViT pools to 2x8 rectangular patches, and RangeSAM uses horizontally elongated windows. The clearest per-class evidence concerns horizontal resolution. At 64x512 → 64x1024 → 64x2048, pole IoU rises 36.0 → 43.8 → 47.9 for RangeNet++, 49.5 → 55.9 → 61.5 for CENet, and 59.4 → 64.2 → 66.4 for RangeFormer. Training on native-resolution crops (STR) nearly matches or beats full-width training.

### Cited Findings
**How each network handles the vertical axis**
- RangeNet++ (IROS 2019) downsamples only horizontally. A 32x encoder reduces width by 32 while "64 pixels still remain intact in the vertical direction". — [Milioto et al., RangeNet++](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/milioto2019iros.pdf). KPRNet says the same: "Input scans have relatively low vertical resolution of 64x2048. To account for this RangeNet++ uses only horizontal strides" — [KPRNet](https://ar5iv.labs.arxiv.org/html/2007.12668)
- SalsaNext replaces strided convolutions with average pooling, at 64x2048 input. The paper does not state whether pooling is anisotropic. — [Cortinhal et al., SalsaNext](https://arxiv.org/html/2003.03653v4)
- CENet (ICME 2022), official code: a stride-1 stem of three 3x3 convolutions (5→64→128→128 channels), then stride 2 in layer2–layer4 applied to both H and W (max stride 8). All scales are bilinearly upsampled to stem resolution and concatenated. Multiple auxiliary heads are used in training only. Input is 64x2048 with 5 channels (x, y, z, range, remission). — [CENet code, modules/network/ResNet.py](https://raw.githubusercontent.com/huixiancheng/CENet/main/modules/network/ResNet.py); [CENet paper](https://arxiv.org/html/2207.12691v1)
- RangeFormer: four stages with overlapping 3x3 patch embeddings, stride 1 in stage 1 and stride 2 in stages 2–4. Features are at (H,W), (H/2,W/2), (H/4,W/4), (H/8,W/8), with input 64x512/1024/2048. — [RangeFormer](https://arxiv.org/html/2303.05367v3)
- RangeViT: a full-resolution convolutional stem, then average pooling to a 2x8 patch grid. Training crops are 64x384 (SemanticKITTI) and 32x384 (nuScenes) (Tables 1, 3, 11). — [RangeViT](https://arxiv.org/html/2301.10222v2)
- FRNet groups points into 64x512 frustums (SemanticKITTI). In its Table V, 64x512 frustums score 67.6 mIoU on val, against 66.8 for 64x2048 and 64.3 for 32x256. FRNet's per-point frustum features make it less sensitive to W than pure 2D networks. — [Xu et al., FRNet, Table V](https://arxiv.org/html/2312.04484)
- RangeSAM (SAM2-Hiera-tiny) uses a stride-1 stem with overlapping 7x7 patches, 8x64 and 16x128 windows, and 64x2048 input. It reaches 60.9 mIoU on the SemanticKITTI test set, with pole 58.9, sign 62.4, trunk 66.9 (Table 1). It has no component ablations. — [RangeSAM](https://arxiv.org/html/2509.15886v1)

**Per-class IoU vs horizontal resolution (RangeFormer Table 2, SemanticKITTI test, %)**

| Method | W | mIoU | pole | sign | trunk | bicycle | motorcyclist |
|---|---|---|---|---|---|---|---|
| RangeNet++ | 512 | 41.9 | 36.0 | 50.0 | 48.4 | 26.2 | 4.0 |
| RangeNet++ | 1024 | 48.0 | 43.8 | 47.2 | 52.5 | 20.6 | 7.1 |
| RangeNet++ | 2048 | 52.2 | 47.9 | 55.9 | 55.1 | 25.7 | 4.8 |
| CENet | 512 | 60.7 | 49.5 | 59.1 | 60.5 | 45.4 | 29.7 |
| CENet | 1024 | 62.3 | 55.9 | 61.0 | 65.4 | 50.5 | 32.5 |
| CENet | 2048 | 64.7 | 61.5 | 67.6 | 69.7 | 58.6 | 43.5 |
| RangeFormer | 512 | 70.0 | 59.4 | 63.6 | 68.6 | 60.5 | 55.4 |
| RangeFormer | 1024 | 72.1 | 64.2 | 65.8 | 71.5 | 66.2 | 56.5 |
| RangeFormer | 2048 | 73.3 | 66.4 | 66.6 | 73.3 | 69.4 | 58.1 |
| CENet w/ STR (crops 384 of 2048) | 384† | 65.8 | 60.5 | 65.4 | 66.2 | 60.2 | 19.7 |
| RangeFormer w/ STR | 384† | 72.2 | 62.8 | 65.0 | 70.4 | 67.1 | 57.5 |

Source: [RangeFormer, Table 2](https://arxiv.org/html/2303.05367v3), read from the paper's PDF text. Other 64x2048 rows from the same table: SalsaNext pole 54.3 / sign 62.1 / trunk 63.6; RangeViT 60.8 / 64.7 / 70.6; KPRNet 58.7 / 64.1 / 69.8; LiteHDSeg 59.5 / 67.7 / 65.8. SalsaNext's own Table I gives the same SalsaNext values (54.3 / 62.1 / 63.6) — [SalsaNext](https://arxiv.org/html/2003.03653v4).
- FRNet test set (Table X, via HTML summary): pole 67.3, sign 67.3, trunk 73.2. — [FRNet](https://arxiv.org/html/2312.04484). **Conflict:** the same summary listed SalsaNext pole as 62.1, which contradicts SalsaNext's own 54.3. The FRNet row should be checked against the PDF before use.
- Range-image pole segmentation in practice: SalsaNext trained on geometry-derived pseudo-labels reaches pole precision/recall/F1 of 0.675/0.674/0.674 on NCLT and 0.607/0.582/0.594 on SemanticKITTI (Table 1). The authors note that pole "range values … are usually significantly different than the backgrounds", which makes poles distinctive on range images. — [Dong et al., "Online pole segmentation on range images…" (RAS 2023), Table 1](https://arxiv.org/pdf/2208.07364)

### Inferences
- PoleSight's 1024→640 horizontal downscale (factor 0.625) falls between RangeFormer's W=512 and W=1024 rows. Halving W costs CENet 6–12 IoU points on pole, sign, and trunk, so the YOLO 640 downscale likely costs a comparable fraction on poles. This is my inference: those are semantic-segmentation numbers on a different sensor and task.
- No range-view network trades vertical resolution for capacity early. A YOLO stem that downsamples H by 4 at P2 and 8 at P3 maps PoleSight's 128 rows (80 after downscale) to 20 or 10 rows. That is far coarser vertically than any range-view segmentation network uses. An anisotropic stem with stride (1,2) at the first one or two stages, or a stride-1 stem in the style of CENet and RangeFormer, is the range-view convention.
- STR shows that the 2D model does not need the full panorama in context to segment small classes. Native-resolution crops recover most of the full-resolution accuracy (RangeFormer pole 62.8 with STR vs 59.4 at W=512). This supports cropping or tiling over downscaling.

### Gaps
- I found no range-view **instance**-segmentation or detection study (rather than semantic segmentation) that ablates vertical vs horizontal stride for pole-like classes.
- RangeNet++'s per-class gain from horizontal-only striding has not been isolated against a symmetric-stride control.
- SalsaNext's exact pooling kernel and stride are not stated in the paper text I retrieved.
- RangeSAM gives no ablation of its stem or windows.
- RangeViT's patch ablation is on nuScenes as a whole; per-class pole results by patch shape were not reported.

## Q4. In multimodal small-object detectors, do fusion gains grow or shrink with object size?

### Takeaway
The evidence is thin and indirect, but it mostly points to fusion helping more where single-modality evidence is weakest. In LiDAR-camera 3D detection, fusion gains grow monotonically with distance, which correlates with smaller and sparser objects. In SuperYOLO, cheap early (pixel-level) fusion beat every feature-level variant for small vehicles: 70.3 mAP50, against 63.8–68.5 for single-stage feature fusion and 59.3 for multistage fusion. The fusion gain over RGB alone was modest (+2.6 mAP50) and shrank for larger YOLOv5 baselines (+0.56 for YOLOv5x). On the tiny-object RGBT-Tiny benchmark, the best fusion detectors did not clearly beat the strongest visible-only detectors. I found no paper that reports fusion-vs-single-modality AP split by object size bin (AP_vt/AP_t/AP_s) on a small-object benchmark.

### Cited Findings
- SuperYOLO (VEDAI, mAP50, Table VII): YOLOv5s RGB 54.82 / IR 49.94 / fused 56.79; YOLOv5x 62.09 / 54.18 / 62.65; **SuperYOLO 72.49 / 65.60 / 75.09**. — [SuperYOLO, Table VII](https://ar5iv.labs.arxiv.org/html/2209.13351)
- SuperYOLO fusion-strategy ablation (Table III; YOLOv5s-noFocus; first fold of VEDAI val; mAP50; read from the paper's PDF text):
  - Pixel-level fusion: concat 69.5 (7.07 M params, 20.37 GFLOPs); MF **70.3** (21.67 GFLOPs).
  - Feature-level concatenation after backbone block 1: 66.0; block 2: 68.5; block 3: 64.8; block 4: 63.8.
  - Multistage feature-level fusion: 59.3 (34.56 GFLOPs).
  - For these small vehicles, fusing earlier was better and cheaper, and fusing later or at multiple stages was worse. — [SuperYOLO, Table III](https://arxiv.org/pdf/2209.13351)
- SuperYOLO per-class AP, RGB → fused (Table VII, from PDF text): car 90.30→91.13; pickup 82.66→85.66; camping 76.69→79.30; truck 68.55→70.18; other 53.86→57.33; tractor 79.48→80.41; boat 58.08→60.24; van 70.30→76.50. IR-only: 87.90 / 81.39 / 76.90 / 61.56 / 39.39 / 60.56 / 46.08 / 71.00. The fusion gain is positive for every class, ranging from +0.8 to +6.2. The weakest RGB classes (other, van) gain the most, but the table is not stratified by object size. — [SuperYOLO, Table VII](https://arxiv.org/pdf/2209.13351)
- Same table for YOLOv5s, RGB → fused: car 80.07→80.81, tractor 64.38→64.29 (a slight drop), mAP 54.82→56.79. Fusion gains also shrink as the baseline grows: YOLOv5x gets only +0.56 mAP50 (62.09→62.65). — [SuperYOLO, Table VII](https://arxiv.org/pdf/2209.13351)
- Removing the Focus stem also helps YOLOv5x: 64.0→69.2 (+5.2) (Table II). — [SuperYOLO, Table II](https://arxiv.org/pdf/2209.13351)
- BEVFusion (NeurIPS 2022, Appendix Table 11, nuScenes mAP by distance), LiDAR-only → LiDAR+camera:
  - TransFusion-L: <15 m 76.3→77.3 (+1.0); 15–30 m 66.1→69.4 (+3.3); >30 m 43.2→49.2 (**+6.0**).
  - CenterPoint: 73.1→77.7 (+4.6); 57.8→65.0 (+7.2); 33.6→42.9 (**+9.3**).
  - PointPillars: 28.2→32.5; 21.2→27.7; 15.1→20.9.
  - The authors: "our fusion framework gives a larger performance boost for distant regions where 3D objects are difficult to detect or classify in LiDAR modality". — [Liang et al., BEVFusion, Tables 11–12](https://arxiv.org/pdf/2205.13790)
- DeepFusion (Waymo): fusion gains of +6.6 for LEVEL_2 objects beyond 50 m vs +1.5 within 30 m. — [Li et al., DeepFusion (CVPR 2022)](https://arxiv.org/pdf/2203.08195) (search-snippet-only)
- RGBT-Tiny (81% of targets <16x16 px; 23 methods). A summarizer reported the best visible-only detector (DiffusionDet) at 38.4 AP (SAFit) and the best RGBT fusion detector (QFDet) at 31.8 AP on visible and 33.2 on thermal (Table III). The authors still conclude that "RGBT detection methods can make full use of RGBT complementary information for performance improvements in both modalities". — [Ying et al., RGBT-Tiny, Table III](https://arxiv.org/html/2406.14482). This needs checking against the PDF, because the summary looks internally inconsistent with the conclusion.

### Inferences
- The distance results (BEVFusion, DeepFusion) suggest that when one modality has too little signal for an object (few LiDAR points far away), a second modality adds the most. For PoleSight, intensity and range are two renderings of the same LiDAR returns and are pixel-aligned. Fusion therefore cannot add more samples per object, only complementary contrast cues: range edges against background, and retro-reflective intensity on signs and poles. The gain may be smaller than in cross-sensor fusion, and it may be largest for the thinnest or lowest-contrast poles. This is speculative.
- SuperYOLO's result that pixel-level fusion beats feature-level fusion for tiny vehicles, together with the misalignment problem cross-sensor fusion faces, suggests early or channel-stacked fusion of intensity and range for PoleSight. Feature-level fusion at stride ≥8 would fuse features in which a 6-px-wide pole is already sub-cell. This is my inference.
- Because PoleSight objects are below the stride-8 cell size, fusion should be applied at or before stride 4. Otherwise fusion gains will be capped by the same resolution bottleneck identified in Q2 and Q3.

### Gaps
- I found no small-object multispectral paper (SuperYOLO, MCF-YOLO, DroneVehicle or VEDAI works) that reports fusion gain stratified by object size bins. MCF-YOLO itself was not retrieved.
- There is no study of intensity+range fusion for thin objects in range images that reports per-size or per-class gains. Adjacent researchers on this project cover intensity/range fusion in general.
- KAIST far-scale miss rates (fusion vs RGB) appeared only in snippets without verifiable tables, so they are excluded.
