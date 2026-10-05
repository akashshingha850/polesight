# Input-channel fusion in range-view (spherical projection) LiDAR networks: channel ablations and evidence

Verification legend used throughout (how I got each number):
- **[PDF-T]**: I read the number myself in the table text extracted from the paper PDF (pdftotext). Highest confidence.
- **[HTML]**: number taken from the arXiv/ar5iv HTML page through a summarising fetch tool. I did not see the raw table, so medium confidence. Verify before quoting in a paper.
- **[ABS]**: abstract or search-engine snippet only. Low confidence on details.

Scope note: "range-view" here means a 2D CNN/transformer on a spherical/cylindrical projection. Point- and voxel-based results are included only where they isolate the intensity channel, and they are labelled as such.

---

## Q1. Which range-view papers contain channel ablations, and what exactly did they find?

### Takeaway
Clean channel ablations in range-view LiDAR segmentation are rare. Most flagship papers (RangeNet++, SalsaNext, CENet, FIDNet, MaskRange, FLARES, FMVNet, RangeViT) stack 5 channels (x, y, z, range, remission), sometimes adding a mask or normals, and never ablate them. Where ablations exist, the pattern is consistent. Intensity is worth roughly +0.4 to +2.6 mIoU in-domain on SemanticKITTI, and more for small or under-represented classes (+5.4 vehicle IoU in SalsaNet SFV, +3.5 car IoU in SqueezeSeg). x, y, z add little or nothing once range and pixel position are known. The validity mask is often one of the most valuable cheap channels. Intensity becomes a liability under sensor or weather shift.

### Cited Findings

#### Master evidence table (range-view unless stated)

| Paper (year, venue) | Arch. / input res. | Dataset / sensor | Channels compared and how combined | Result | Verif. |
|---|---|---|---|---|---|
| SqueezeSeg (2018, ICRA) | SqueezeNet+CRF, 64×512 front 90° | KITTI-derived (HDL-64E), car/ped/cyclist | x,y,z,i,r stacked vs. same **without intensity** | Car IoU (class-level) 64.6 with intensity (w/ CRF, Table I) vs. **57.1** trained on KITTI w/o intensity (Table III). Authors: "the IoU score is worse, due to the loss of the intensity channel" | [PDF-T] |
| SqueezeSegV2 (2019, ICRA) | SqueezeSeg+BN+mask+focal+CAM, 64×512 | KITTI (HDL-64E) | Adds binary **LiDAR mask** channel to (x,y,z,i,d) | +BN → +BN+M: car 71.6→70.0, ped 15.2→17.1, **cyclist 25.4→32.3**, avg 37.4→39.8 (Table I) | [PDF-T] |
| SqueezeSegV2 (2019) | same | GTA-LiDAR→KITTI (sim-to-real) | Synthetic data lacks intensity; adds **learned intensity rendering (LIR)** | Car IoU 30.0→**42.0**, ped 2.1→16.7 with LIR (Table II) | [PDF-T] |
| SalsaNet (2020, IV) | encoder-decoder; SFV 64×512 (front 90°) | KITTI road/vehicle (HDL-64E) | Leave-one-out over SFV channels X,Y,Z,I,R,M (stacked) | Full: vehicle IoU 71.44 / avg 79.71. −I: **66.06 / 77.41**. −R: 64.57 / 77.30. **−Mask: 59.53 / 75.20**. −X: 71.34 / 79.75. −Y: 71.21 / **80.05** (better without). −Z: 69.77 / 79.50 (Table III). Authors: SFV "contains redundant information that mislead the feature learning" | [PDF-T] |
| SalsaNet (2020) BEV branch (not range-view) | BEV 256×64×4 | same | mean/max elevation, reflectivity, density | **−Reflectivity: vehicle 69.19→62.46** (largest drop of the four channels) (Table II) | [PDF-T] |
| Triess et al., scan-based study (2020, IV) | RangeNet variant | SemanticKITTI | Footnote to Table I: "we drop x, y, and z channels from the input as our experiments showed that these features do not influence the performance in a significant way" | Qualitative only. No ablation table for it | [HTML] |
| RIU-Net (2019) | U-Net, 512×64 | KITTI (SqueezeSeg split) | Uses only **2 channels: depth + elevation**. Validity mask used only in the loss | No ablation. Shows a 2-channel input is workable. Note: some search snippets wrongly say "reflectance + depth" | [PDF-T] |
| 3D-MiniNet (2020, RA-L) | learned projection + 2D net, 64×2048 | SemanticKITTI | C1={x,y,z,depth,remission} vs. C2 adding relative (to neighbourhood mean) versions + dEuc (11 features) | Relative features +1.3 mIoU (49.9→51.2, Table I). Not a per-channel ablation | [HTML] |
| Meta-RangeSeg (2022, RA-L) | SalsaNext-like | SemanticKITTI multi-scan | 5-ch range image vs. 9-ch "range residual" image | 42.8→43.3 (+0.5) (Table IV). Authors: "As the x, y and z information has been encoded in the horizontal and vertical coordinates under range view, the range channel is the most important one", so range is processed in a separate branch | [PDF-T] |
| FIDNet (2021, IROS; arXiv v-later) | ResNet + interpolation decoder, 64×2048 | SemanticKITTI | 5 ch → 8 ch (add normal vector n1..n3) | No number given: "adding a normal vector for each point will make the training more stable". Updated net ≈60.0 test mIoU | [PDF-T] |
| CENet (2022, ICME) | 64×512 ablation setting | SemanticKITTI val | CENet drops FIDNet's normals | Reports +3.5% over FIDNet "in which the required normal vector has been removed" (Table 4). Confounded by architecture changes | [PDF-T] |
| Reflectivity Is All You Need (2024 arXiv; IFAC 2025) | SalsaNext | Rellis-3D (Ouster OS1-64) | rxyzi (raw intensity) vs. rxyzn (calibrated reflectivity) vs. rxyzirn (both) vs. rxyzi_gau (learned calibration). All stacked | mIoU 38.4 / **42.4** / 41.2 / 41.9 (Table I) | [PDF-T] |
| same | SalsaNext | SemanticKITTI val (HDL-64E) | same configs | mIoU 54.0 / 54.2 / **54.8** / 53.7 (Table III). Urban gain is marginal | [PDF-T] |
| same | SalsaNext | Rellis Ouster→Velodyne VLP-32 (cross-sensor) | rxyzi vs. rxyzn | **7.5 vs 13.8** mIoU (Table II) | [PDF-T] |
| Learning to Simulate Realistic LiDARs (2022, IROS) | RangeNet21 (Milioto), cropped range images, vehicle-only | Waymo test | Real data with vs. without intensity | Vehicle IoU **84.8 vs 83.4** (Table III). Synthetic+rendered intensity: 59.4 vs 55.9 w/o intensity (Table IV) | [PDF-T] |
| Sanchez et al., DG of 3D seg. (2023, ICCV); repeated in 3DLabelProp (2025) | **CENet (range-view)**; plus point/voxel nets | Train SemanticKITTI; test SK (seq 08) and SemanticPOSS | with vs. without reflectivity (reflectivity rescaled for SP) | CENet: SK→SK **61.4 vs 58.8** (−2.6); SK→SP 27.5 vs 27.9. Point/voxel: KPConv 59.9/58.3 & 33.1/**39.1**; SPVCNN 63.8/62.3 & 36.9/**45.4**; Cylinder3D 66.9/60.7 & 33.8/**41.9**; Helix4D 63.1/60.0 & 35.0/36.0; SRUNet 61.9/58.6 & 45.2/45.3 (Table 3 ICCV; Table VI 3DLabelProp) | [PDF-T] |
| Towards Generalized Range-View Seg. in Adverse Weather (2025, arXiv 2506.08979) | SalsaNext, RangeViT, CENet, RangeNet++ | SemanticKITTI→SemanticSTF (fog/rain/snow) | Baseline (stacked xyz+r+intensity) vs. **geometry-only** stem vs. **two-branch stem** (StemG for depth/xyz, StemR for reflectance) | SalsaNext (Table 4), target avg / source mIoU: baseline 7.9/59.7; geometry-only **20.1/58.2**; two-branch plain 8.8/59.7; two-branch+RDC 15.8/60.1; full 28.2/60.2. RangeViT (Table 5): baseline 10.2/59.6; geometry-only 16.8/59.2; two-branch plain 10.8/**60.6**; full 28.9/59.3 | [PDF-T] |
| Reichert et al., high-res automotive (2025, arXiv 2504.21602) | ResNet34 encoder-decoder, 128×2048 | SemanticTHAB (Ouster OS2-128) | 8-ch stacked: r, reflectivity, xyz, **normals** (from cross-product of image-gradient vectors) | Adding normals: 47.86→**49.57** mIoU (reported as "+4.24%", a relative gain vs the 47.55 baseline) (Table V). No reflectivity/range removal | [HTML] |
| FMVNet "Filling Missing Values Matters" (2024, arXiv 2405.10175) | ConvNeXt-based. Inputs: range, x, y, z, intensity, **mask** (6 ch) | SemanticKITTI val | No channel ablation. Ablates missing-value filling (see Q4) | n/a | [HTML] |
| MaskRange (2022), FLARES (2025), MSCNet (2026, PLOS One) | 5-ch stacked | SemanticKITTI | **No channel ablation** (checked) | n/a | [HTML] |
| RangeDet (2021, ICCV), detection | Meta-Kernel; 8 ch: range, intensity, elongation, x,y,z, azimuth, inclination | Waymo | No input-channel ablation. Meta-Kernel uses relative coords: pedestrian AP 69.06→74.16 | n/a for channels | [PDF-T] |
| What Matters in RV 3D Detection (2024) | RV detector | AV2 (x,y,z,r,i), Waymo (+elongation) | No per-feature ablation. Only feature dimensionality | n/a | [HTML] |
| LaserNet (2019, CVPR), detection | RV | ATG4D | Input: range, height, azimuth, intensity, **validity flag** | No channel ablation | [PDF-T] |
| Dong et al., pole segmentation on range images (2022, RAS) | SalsaNext retrained, poles vs non-poles, 32×256 | NCLT (HDL-32E), MulRan (OS1-64), SemanticKITTI pseudo-labels | Feeds "range images" with valid range normalised to [0,1]. Text does not say whether intensity is fed | No channel ablation | [PDF-T] |

#### Additional individual findings
- SqueezeSeg authors on why they dropped intensity for sim data: "our simulated point cloud does not contain intensity measurements; we therefore excluded intensity as an input feature" — [SqueezeSeg, Sec. IV](https://arxiv.org/abs/1710.07368) [PDF-T]
- SqueezeSegV2 mask definition: "a binary mask indicating if each pixel is missing or existing … significantly improves segmentation accuracy for cyclists" — [SqueezeSegV2, Sec. III-C, Table I](https://arxiv.org/abs/1809.08495) [PDF-T]
- SalsaNet SFV channel ablation numbers (Table III): full X,Y,Z,I,R,M = 71.44 vehicle / 79.71 avg. Dropping I gives 66.06/77.41, dropping R 64.57/77.30, dropping mask 59.53/75.20 — [SalsaNet, Table III](https://arxiv.org/abs/1909.08291) [PDF-T]
- Reflectivity paper, SemanticKITTI val (Table III), full rows in standard class order (car … traffic-sign, mIoU). rxyzi: 90.9, 37.1, 51.8, 83.4, 43.0, 68.7, 82, 0.1, 77.4, 44.8, 60.1, 6.0, 73.2, 45.9, 46.1, 64.1, 47.9, 56.3, 46.8, 54.0. rxyzn: 92.4, 44.8, 52.5, 71.4, 46.9, 69.5, 84.6, 0, 78.6, 44.4, 59.6, 0.3, 73.6, 49.9, 45.5, 64.7, 48.0, 58.1, 45.6, 54.2. rxyzirn: …, fence 49.8, veg 51.6, trunk 66.0, terrain 54.0, pole 57.0, sign 47.0, mIoU 54.8. Caveat: road (77.4) and vegetation (46.1) are far below typical SalsaNext validation values, so treat their absolute numbers with caution — [Reflectivity Is All You Need, Table III](https://arxiv.org/abs/2403.13188) [PDF-T]
- Learning to Simulate: "CNNs trained on real range images benefit from having access to the intensity channel" (84.8% vs 83.4% vehicle IoU) — [Guillard et al. 2022, Table III](https://arxiv.org/abs/2209.10986) [PDF-T]
- Sanchez et al.: "reflectivity is a very efficient feature for source-to-source semantic segmentation, we believe it is detrimental to domain generalization". Their generalization benchmark was then built without reflectivity — [Sanchez et al. ICCV 2023, Sec. 3.4, Table 3](https://arxiv.org/abs/2212.04245) [PDF-T]
- Sanchez et al. also found the range-view CENet has "a very large sensitivity of the sparsity" (SK→SK32 shift) when trained geometry-only — [Sanchez et al. 2023, Sec. 3.3](https://arxiv.org/abs/2212.04245) [PDF-T]
- MSCNet (2026) processes each of the 5 channels independently in a "Single Channel Multi-Scale Feature" block before fusion, but reports no channel ablation — [Feng et al., PLOS One 2026](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0345761) [HTML]
- Piewak et al. (cross-sensor portability, VLP-32C→VLS-128) keep reflectivity in the input and do not isolate it. Cross-sensor drop 70.9→45.2 mIoU is attributed to density differences — [Piewak et al. ITSC 2019, Table II](https://arxiv.org/abs/1907.02149) [HTML]

### Inferences
- In-domain, intensity is a modest but real gain for range-view nets: −2.6 mIoU for CENet without reflectivity, −1.5 for SalsaNext and −0.4 for RangeViT in the adverse-weather paper's "geometry-only" rows. In the older 3-class KITTI tasks the drop for the dominant class is −2 to −7 IoU points (SqueezeSeg car, SalsaNet vehicle).
- Explicit x, y, z channels give little once range and pixel (row, col) are present. Evidence: Triess's footnote; SalsaNet improving slightly when X or Y is removed; Meta-RangeSeg's argument that xyz is implicit in image coordinates. **PoleSight transfer:** with only intensity+range released, the missing xyz is probably not a big handicap. If the angular grid is known, pseudo-xyz can be reconstructed from range + row (elevation) + column (azimuth). Inference only; no paper tests MLS 1024×128.
- The validity mask is the cheapest high-value channel: SalsaNext-era SFV −11.9 vehicle IoU without it; SqueezeSegV2 +6.9 cyclist IoU with it. **PoleSight transfer:** derive it from range==0 / no-return pixels and add it as a third channel. This matters most for small objects with ragged silhouettes, such as poles at 71 px² median.

### Gaps
- No primary channel ablation found for RangeNet++, SalsaNext (PDF checked), RangeFormer (PDF checked: inputs are coordinates, depth, intensity and an existence flag; no channel ablation), FRNet (PDF checked; only notes that mean-variance normalisation of intensity across datasets gives "slight performance gains" in multi-dataset training, with no table number extracted), RangeSeg 2301.04275 (PDF checked), KPRNet, Lite-HDSeg, RangeViT, LENet, MINet, PolarNet-style comparisons, or Fast FMVNet V1–V3. KPRNet/Lite-HDSeg/RangeViT/LENet/MINet were not opened in full, so a supplementary-material ablation cannot be excluded.
- Triess et al.'s per-class Table I numbers came back inconsistent across two fetches. Only the xyz footnote is reported here.

---

## Q2. Do range-view networks process channels in separate branches, and does it beat stacking?

### Takeaway
Yes, a few do: two-branch geometry/reflectance stems, per-channel encoders, range-only branches, and intensity-conditioned attention. The only direct, same-backbone comparison against stacking (Yang et al. 2025) shows that **splitting alone does nothing in-domain for SalsaNext (59.7 vs 59.7) and gives +1.0 for RangeViT (59.6 → 60.6)**. The benefits come from what each branch does: weather-noise suppression on geometry and AdaIN-style calibration on reflectance. They show up mainly out-of-domain.

### Cited Findings
- Yang et al. 2025 split the range-view stem into "two parallel branches: one for geometric attributes and the other for reflectance intensity". StemG takes "geometric attributes (e.g depth and XYZ coordinates)". StemR takes reflectance, followed by Reflectance Distortion Calibration (memory-guided adaptive instance normalisation) — [Towards Generalized Range-View LiDAR Segmentation in Adverse Weather, Sec. 3](https://arxiv.org/abs/2506.08979) [PDF-T]
- Same paper, SalsaNext ablation (Table 4), SemanticSTF avg / SemanticKITTI source mIoU: stacked baseline 7.9 / 59.7; geometry-only 20.1 / 58.2; geometry-only+GAS 24.0 / 58.3; **two branches, no extra modules 8.8 / 59.7**; two branches+RDC 15.8 / 60.1; full 28.2 / 60.2. Authors: "the inclusion of reflectance alone does not offer generalization benefits", and removing it gives "an average mIoU gain of +12.2" on SemanticSTF — [Yang et al. 2025, Table 4](https://arxiv.org/abs/2506.08979) [PDF-T]
- RangeViT ablation (Table 5): baseline 10.2 / 59.6; geometry-only 16.8 / 59.2; two branches 10.8 / **60.6**; two branches+RDC 20.2 / 60.7; full 28.9 / 59.3 — [Yang et al. 2025, Table 5](https://arxiv.org/abs/2506.08979) [PDF-T]
- Authors' explanation of why range-view amplifies intensity problems: "In range-view projections, this sensitivity is further compounded by densely packed pixels, where even slight variations in reflectance tend to propagate across neighboring pixels" — [Yang et al. 2025](https://arxiv.org/abs/2506.08979) [HTML]
- GRC (CVPR 2025) uses a hybrid: a voxel geometric branch plus a **2D range-view reflectance branch**. On SemanticKITTI→SemanticSTF, MinkNet with reflectance gets 24.4 mIoU, without reflectance 37.3, and naive addition of a reflectance encoder 36.5. Adding a complementarity constraint and local/global fusion reaches 42.5 (Tables 1, 4) — [Yang et al., Towards Explicit Geometry-Reflectance Collaboration, CVPR 2025](https://arxiv.org/abs/2506.02396) [HTML]
- Meta-RangeSeg extracts the range channel "separately" into a context module and fuses it with multi-scale features via attention. The full model improves 4.1 mIoU over SalsaNext, but the separate-range branch is not isolated (Table IV) — [Meta-RangeSeg, Sec. III, Table IV](https://arxiv.org/abs/2202.13377) [PDF-T]
- SqueezeSegV3's Spatially-Adaptive Convolution computes its attention map from the raw input image X0 (coordinates + remission) via a single conv. SemanticKITTI val ablation at 64×512 with SSGV3-21 (Table 3): baseline 44.0 → SAC-S 44.9, SAC-IS 44.0, SAC-SK 45.4, **SAC-ISK 46.3** mIoU (PAC 45.2, SE 44.2, CBAM 44.8, CAM 42.1). The attention-conv kernel size matters: 1×1 45.5, 3×3 44.5, 5×5 45.4, 7×7 46.3 (Table 4). Authors: "Comparing SAC-S and SAC-IS, adding the input channel dimension does not improve the performance". Motivation: "For a LiDAR image, its features are converted by spherical projection, which introduces very strong spatial priors" — [SqueezeSegV3, ECCV 2020](https://arxiv.org/abs/2004.01803) [PDF-T for tables; HTML for motivation quote]
- DS-RangeNet (2026) runs separate geometry (voxel-PCA descriptors) and intensity streams (normalised range, local intensity statistics, boundary strength, intensity curvature), with intensity–geometry cross-attention. 73.2% mIoU on its industrial indoor UBPC-9 split. No stacked-input comparison seen (abstract only; MDPI page returned 403) — [DS-RangeNet, Electronics 2026](https://www.mdpi.com/2079-9292/15/17/3983) [ABS]
- MSCNet (2026) has a per-channel multi-scale block (each of x, y, z, d, r processed independently before fusion). No ablation vs stacking — [MSCNet, PLOS One](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0345761) [HTML]
- Mangrove3D (TLS spherical projection 540×1440, 2025/26) uses a multi-encoder design: one pretrained encoder per 3-channel group, fused at the bottleneck. The authors cite prior work saying it beats naive early fusion but **run no comparison themselves**. Single channels (raw/enhanced intensity, range, Z, geometric descriptors) gave 0.702–0.731 mIoU. A preprocessed I.R.Z stack gave 0.745, 6-ch stacks 0.754–0.761, and 9-ch 0.768, after which gains saturate — [Through the Perspective of LiDAR, arXiv 2510.06582](https://arxiv.org/abs/2510.06582) [HTML]
- SqueezeSegV2's intensity-rendering network is a related "intensity reconstruction" design. It predicts intensity from xyz (10-bin classification + regression) to fill the missing channel in synthetic data, giving +12.0 car IoU in sim-to-real — [SqueezeSegV2, Sec. IV-B, Table II](https://arxiv.org/abs/1809.08495) [PDF-T]

### Inferences
- For a same-sensor, same-domain setting like PoleSight, the evidence does not predict a large win from separate branches over simple stacking (0.0 / +1.0 mIoU in the only controlled test). The main argument for branches is robustness to intensity distribution shift, such as other scanners or weather, which PoleSight's single-sensor data may not exercise.
- A two-branch design has a practical upside for PoleSight: separate stems allow ImageNet-pretrained weights per rendering, as in Mangrove3D's multi-encoder design, and channel-wise dropout or calibration of the intensity branch. That is a design inference, not a measured result for MLS poles.

### Gaps
- No paper compares late fusion, cross-attention and early stacking of range vs intensity on the same range-view backbone *in-domain on thin classes*.
- No range-view instance-segmentation paper (Mask R-CNN-style on range images) with a verifiable intensity-vs-range channel comparison was found. A search snippet claimed "range-only and range+intensity two-channel Mask R-CNN performed similarly", but I could not identify or verify the source, so it is excluded.

---

## Q3. Measured effect of intensity on thin / pole-like classes

### Takeaway
Direct per-class evidence is thin and mixed. On SemanticKITTI, swapping raw intensity for calibrated reflectivity moved pole +1.8 and traffic-sign −1.2 IoU, with fence +4.0 and trunk +0.6. On off-road Rellis-3D, calibration **reduced** pole IoU (56.1→52.3) while raising mIoU +4.0. No paper found reports SemanticKITTI pole/traffic-sign IoU for intensity vs. no intensity in a range-view net. Under weather or sensor shift, thin classes collapse with intensity and recover partly without it.

### Cited Findings
- SemanticKITTI val, SalsaNext (Table III), raw intensity (rxyzi) vs calibrated reflectivity (rxyzn) vs both (rxyzirn): **pole 56.3 / 58.1 / 57.0; traffic-sign 46.8 / 45.6 / 47.0; trunk 64.1 / 64.7 / 66.0; fence 45.9 / 49.9 / 49.8**; bicycle 37.1 / 44.8 / 41.3; person 68.7 / 69.5 / 70.5 — [Reflectivity Is All You Need, Table III](https://arxiv.org/abs/2403.13188) [PDF-T]
- Rellis-3D (Ouster OS1-64), SalsaNext (Table I), rxyzi / rxyzn / rxyzirn / rxyzi_gau: **pole 56.1 / 52.3 / 51.0 / 52.5**; fence 0.6 / 1.4 / 2.7 / 3.2; log 14.5 / 10.3 / 9.9 / 12.6; person 62.8 / 76.0 / 82.7 / 82.6; tree 49.6 / 55.7 / 55.8 / 58.4 — [Reflectivity Is All You Need, Table I](https://arxiv.org/abs/2403.13188) [PDF-T]
- Cross-sensor (Ouster→Velodyne VLP-32) pole IoU is near zero either way: 0.1 (intensity) vs 1.1 (reflectivity); person 2.1 vs 34.1 — [Reflectivity Is All You Need, Table II](https://arxiv.org/abs/2403.13188) [PDF-T]
- Adverse-weather paper: the SalsaNext baseline (stacked with intensity) on SemanticKITTI→SemanticSTF is near zero on many classes. The proposed two-branch+GAS+RDC model raises overall target mIoU 7.9→28.2. Per-class pole/traffic-sign/trunk/fence values exist in Table 1, but the PDF text columns could not be reliably aligned to class names, so they are not reported here — [Yang et al. 2025, Table 1](https://arxiv.org/abs/2506.08979) [PDF-T for totals]
- SqueezeSegV2: the validity mask (not intensity) gave the largest small-class gain, cyclist +6.9 IoU — [SqueezeSegV2, Table I](https://arxiv.org/abs/1809.08495) [PDF-T]
- Point-based reference (not range-view), Mangrove3D thin "Stem" class with PointNet++: XYZ 0.401 → +normals/geometric 0.424 → +intensity/range/Z (IRZ) 0.446 IoU (Table 6) — [arXiv 2510.06582](https://arxiv.org/abs/2510.06582) [HTML]
- Toronto-3D (MLS, point-based): the dataset paper argues road markings need colour/intensity. The only ablation I found compares xyz vs xyz+RGB (71.03 mIoU with RGB), not intensity — [Toronto-3D, CVPRW 2020](https://arxiv.org/abs/2003.08284) [ABS]
- Multispectral MLS (raw 2D-scan rasters with intensity, reflectance, echo deviation, range as channels; urban): multispectral input gave a "71% and 28% relative increase" in mIoU (to 43.5) vs single-wavelength references. Multi-scan stacking raised single-wavelength mIoU 45.4 → 62.1 (24 scans). Per-class pole numbers not seen — [Semantic segmentation of raw multispectral laser scanning data from urban environments, ISPRS Open J. 2024](https://www.sciencedirect.com/science/article/pii/S2667393224000048) [ABS; full text 403]

### Inferences
- Calibration of intensity is not uniformly good for poles. Range normalisation removes the near-range brightness pattern, which may itself be a cue for close, vertical, narrow objects (Rellis pole −3.8). In urban SemanticKITTI the effect on pole/sign is within ±2 IoU, probably within run-to-run noise (single runs, no seeds reported).
- For PoleSight's 4 pole classes (which may differ by material: metal, wood, concrete), intensity could be more informative than for the SemanticKITTI "pole vs. not pole" split. No study tests material-discriminating pole subclasses, so this is an untested hypothesis worth an ablation (range-only vs intensity-only vs both).

### Gaps
- No found paper reports SemanticKITTI, nuScenes, SemanticPOSS, Paris-Lille-3D or Toronto-3D **pole / traffic-sign IoU with vs. without intensity** for a range-view network. Sanchez et al. have per-class tables (supplementary Tables 14–25) for no-reflectivity models only; not retrieved.
- No range-view instance-segmentation (AP on poles) channel ablation found anywhere.

---

## Q4. Normalisation / encoding of channels and missing-value handling

### Takeaway
There is solid evidence that **missing-value handling matters**: +2.2 to +3.3 mIoU from filling holes across four range-view nets, and masks worth several IoU points on small classes. Intensity **calibration** helps cross-sensor and off-road but is marginal in urban scenes. **Intensity dropout** (randomly replacing it with occupancy) recovers most of the in-domain loss of dropping intensity while keeping most of the generalisation gain. Standard per-channel z-score or [0,1] scaling is universal but never ablated. I found **no ablation of log-range or inverse-range encodings** for range-view segmentation.

### Cited Findings
- FMVNet: filling missing values with Scan Unfolding++ (ring-index projection) plus range-dependent KNN interpolation (copy the nearest-range neighbour into empty pixels) improves SemanticKITTI val mIoU for RangeNet53++ 61.5→64.4, FIDNet 63.8→66.0, CENet 63.5→66.3 and FMVNet 65.3→68.6 (Table 11). Inputs are z-scored ("normalized to be zero-mean and unit variance") — [Filling Missing Values Matters, arXiv 2405.10175](https://arxiv.org/abs/2405.10175) [HTML]
- Mask channel: SalsaNet SFV vehicle IoU 71.44 → 59.53 when the occupancy mask is removed (Table III) — [SalsaNet](https://arxiv.org/abs/1909.08291) [PDF-T]. SqueezeSegV2 cyclist 25.4 → 32.3 when the mask is added (Table I) — [SqueezeSegV2](https://arxiv.org/abs/1809.08495) [PDF-T]
- SqueezeSegV2's Context Aggregation Module was designed for dropout noise (missing points): "as we increase the dropout probability, the error also increases", while CAM "is much less sensitive". +CAM moved avg IoU 40.5→44.9 (Table I) — [SqueezeSegV2](https://arxiv.org/abs/1809.08495) [PDF-T for table; HTML for quote]
- RIU-Net uses a validity mask m only to exclude empty pixels from the loss (not as input) — [RIU-Net Sec. III](https://arxiv.org/abs/1905.08748) [PDF-T]
- Intensity calibration formula used by the reflectivity paper: I(ρ) = I(R,α,ρ)·R² / (cos α · η(R)), with η(R) a near-range lens-defocus term estimated from annotated data. Gains: +4.0 mIoU Rellis, +0.2 SemanticKITTI, +6.3 cross-sensor — [Reflectivity Is All You Need, Sec. II, Tables I–III](https://arxiv.org/abs/2403.13188) [PDF-T]
- Raw intensity "is influenced by factors such as range, incidence angle, and surface properties", which "can distort the data and limit the precision of segmentation" — [Reflectivity Is All You Need](https://arxiv.org/abs/2403.13188) [HTML]
- Reflectivity dropout: 3DLabelProp trains SRU-Net with 50% of scans using occupancy instead of reflectivity. Results, SK / SK32 / P64 / PFF / SP / Waymo / nuScenes / PL3D: with reflectivity 61.9/57.0/44.7/16.5/45.2/26.0/39.6/30.7; without 58.6/54.0/44.2/22.2/45.3/33.1/42.7/33.1; **dropout 61.4/57.4/43.3/17.2/46.4/32.2/45.4/33.9** (Table VII). This is a voxel model; the authors still chose "without" for simplicity — [3DLabelProp, arXiv 2501.14605](https://arxiv.org/abs/2501.14605) [PDF-T]
- ParisLuco3D: dropping intensity for SemanticKITTI→ParisLuco3D raised SRUNet 27.9→30.7 and Cylinder3D 2.7→23.0 (voxel). With the *same* sensor (nuScenes→ParisLuco3D, both HDL-32), keeping intensity was better (SRUNet 37.4 vs 32.3; C3D 25.5 vs 17.1) (Table V). For detection the authors found intensity hurts even with the same sensor (−2.0 to −10.2 mAP, Table VII), "due to a difference in the intensity distributions of objects" — [ParisLuco3D, arXiv 2310.16542](https://arxiv.org/abs/2310.16542) [PDF-T]
- Sanchez et al. rescaled reflectivity to match SemanticPOSS's range before testing, "a first step of domain adaptation" — [Sanchez et al. 2023, Sec. 3.4](https://arxiv.org/abs/2212.04245) [PDF-T]
- Langer et al. (RangeNet++/LiDAR-bonnetal domain transfer HDL-64→HDL-32) keep remission and render it from a TSDF mesh. They note "semantics and remissions … are not as consistent from different viewing angles as the depth" — [Langer et al. IROS 2020](http://www.ipb.uni-bonn.de/pdfs/langer2020iros.pdf) [PDF-T]
- Value scaling in practice: Dong et al. normalise valid range to [0,1] (32×256 input) — [Dong et al. RAS 2022](https://arxiv.org/abs/2208.07364) [PDF-T]. SalsaNet normalises each BEV channel to [0,1] — [SalsaNet](https://arxiv.org/abs/1909.08291) [PDF-T]. FMVNet z-scores inputs — [arXiv 2405.10175](https://arxiv.org/abs/2405.10175) [HTML]. None ablate the choice.
- Mangrove3D preprocessing (histogram-stretch "contrast enhancement" of intensity and range; inverse Z) is part of its best stacks, but single-channel raw vs enhanced numbers are reported only as a 0.702–0.731 aggregate — [arXiv 2510.06582](https://arxiv.org/abs/2510.06582) [HTML]
- Resolution as a related "encoding" choice: FLARES finds a 512-wide range image (with 3 sub-clouds and multi-cloud hole filling) optimal vs 2048 on SemanticKITTI (Fig. 8b) — [FLARES, arXiv 2502.09274](https://arxiv.org/abs/2502.09274) [HTML]. Yang et al. show adverse-weather baselines do not improve with resolution (avg 12.5 at 512 vs 9.9 at 1024 for SalsaNext) — [arXiv 2506.08979, Table 6 text](https://arxiv.org/abs/2506.08979) [PDF-T]
- Normals derived from the range image: FIDNet computes them "following [41], [42]" (range-image normal estimation) for training stability. Reichert et al. get +1.71 mIoU from normals computed by cross products of image-gradient vectors at 128×2048 — [FIDNet](https://arxiv.org/abs/2109.03787) [PDF-T]; [Reichert et al. 2025](https://arxiv.org/abs/2504.21602) [HTML]

### Inferences
- **PoleSight transfer (instance segmentation, intensity+range only):**
  1. Add an explicit validity mask channel (from no-return pixels). This has the strongest, most consistent evidence for small objects.
  2. Consider filling holes before the network (nearest-smaller-range KNN fill as in FMVNet) while keeping the mask, since masks are tiny (median 71 px²) and holes can break pole silhouettes.
  3. Normals and pseudo-xyz can be computed from range plus the (row = elevation, column = azimuth) grid if the angular spacing is known. Normals gave +1.7 to +2.0 mIoU at 128-beam resolution in one study. xyz alone likely adds little.
  4. Because PoleSight is single-sensor, keep intensity. The generalization penalty applies only if models must transfer to other scanners. If they must, intensity dropout (replace with zeros/occupancy on 50% of samples) is a cheap hedge with published evidence.
  5. Log or inverse range encodings are unvalidated in this literature; treat them as a hyperparameter to test, not a known gain.
- Uncalibrated MLS intensity (Finnish MLS) likely has a range/incidence-angle dependence. Range-normalising it (R² term) may help or hurt poles (Rellis pole fell 3.8 IoU after calibration), so test both raw and calibrated intensity.

### Gaps
- No ablation found of log(range), 1/range, or per-channel mean/std vs min-max scaling for range-view segmentation or detection.
- No study of interpolation vs. mask-only vs. both in an *instance* segmentation setting, or on pole-sized objects specifically.
- No 1024×128 MLS range-image study with channel ablations was found. The closest are Ouster OS-128 automotive (128×2048) and TLS Mangrove3D (540×1440).
