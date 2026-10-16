# LiDAR as a camera: 2D detectors and instance segmenters on single vs fused LiDAR image channels (2018–2026)

Scope note: these notes cover works that render spinning/mobile LiDAR returns into 2D images (signal/intensity, calibrated reflectivity, near-IR/ambient, range/depth) and run 2D detectors or instance segmenters on them. Evidence levels are marked throughout as one of:
- **[paper-read]**: numbers or quotes taken from the full text (PMC/arXiv full text, either through the fetch summariser or from a local `pdftotext` of the PDF).
- **[abstract-level]**: taken from an abstract, dataset page, README or search snippet only.

Note on the fetch summariser: it sometimes garbles details. Where a detail looked suspicious, it is flagged.

## Q1. Which papers compare single-channel vs combined LiDAR channels for 2D detection/segmentation, and what were the numbers?

### Takeaway
Only two works were found that ablate single vs combined same-sensor LiDAR channels for 2D detection or instance segmentation:
- **Casado-Coscolla et al. 2024** (YOLOv8-seg, persons, Ouster OS0-128). Reflectivity is the strongest single channel, range matches or beats it at strict IoU, and the reflectivity+range combination scores highest on average. The exact values are only in a figure.
- **The SnowPole extended evaluation (NTNU, FAIEMA 2025)**. Six YOLO models on Ouster OS2-128 snow poles. Pseudo-colour combinations of NIR/signal/reflectivity beat single channels "across most configurations". I could only get this at abstract level, with no numbers.

Every other "LiDAR-as-camera" detection paper either uses a single channel (Turku Sensors 2023 used signal only, as did the Ouster blog and the Turku UAV paper) or uses a fixed 3-channel stack with no single-channel ablation (LiCAR 2025, UTS "Connecting the Dots").

### Cited Findings

**Yu Xianjia, Salimpour, Peña Queralta, Westerlund (Univ. Turku), Sensors 2023, 23(6):2936; arXiv:2203.04064 (2022)**
- The study focused on one channel only. Quote: "we decided to focus on one of the three types of images provided by the lidar sensors, namely the signal image. In addition to this, the sensors also provided depth, near-infrared and reflectivity images." The other channels "did not perform as well with out-of-the-box DL models without further preprocessing." No per-channel numbers were reported and channels were never stacked. [paper-read] — [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10058223/); [arXiv](https://arxiv.org/abs/2203.04064)
- Models: YOLOv5, YOLOX, Faster R-CNN and Mask R-CNN for detection; HRNet+OCR+SegFix, PointRend and Mask R-CNN for segmentation. All used general-purpose pretrained weights applied out of the box. The summariser found no fine-tuning. [paper-read] — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10058223/)
- Table 3 (detection success proportion):
  - Person: 95.3% with YOLOX indoors, 63.3% with YOLOX outdoors.
  - Car: 89.3% with YOLOv5 outdoors.
  - Chair: 51.5% with YOLOX indoors.
  - Bike: 64.3% with Mask R-CNN outdoors.

  [paper-read] — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10058223/)
- Table 4 (best precision/recall):
  - Person indoors: YOLOX, P = 1.0 / R = 0.953.
  - Car outdoors: Faster R-CNN, P = 0.943 / R = 0.688.

  Throughput: YOLOv5 ≈ 24 Hz; Faster R-CNN and PointRend ≈ 15 Hz. [paper-read] — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10058223/)
- A ResearchGate figure from the paper is titled "YOLOx detection based on images from other channels of Ouster lidar sensors". This is a qualitative illustration only. — [ResearchGate figure](https://www.researchgate.net/figure/YOLOx-detection-based-on-images-from-other-channels-of-Ouster-lidar-sensors_fig3_369138251)

**Casado-Coscolla, Sanchez-Belenguer, Wolfart, Sequeira, "Point-Cloud Instance Segmentation for Spinning Laser Sensors", Journal of Imaging 2024 (PMC11728245)**
- Data: Ouster OS0-128 at 1024×128, mostly indoor with some outdoor sequences, one class (person), about 4,000 training scans. [paper-read] — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11728245/)
- Model: YOLOv8-seg in sizes n/s/m/l/x, starting from pretrained weights and trained for 250 epochs. [paper-read] — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11728245/)
- Ablation: all seven channel subsets (A, D, R, A+D, A+R, D+R, A+D+R; A = ambient, D = range, R = reflectivity) were evaluated. Figure 5 reports AP50 and mAP50:95 for 2D boxes and for masks. Quote: "reflectivity seems to be the most contributing one. However, with high overlaps (mAP50:95), the range component provides similar or even better results. The combination of both is, on average, the one that provides the highest scores." The ambient channel contributes least and is "noisy" indoors. [paper-read; values are in a figure and were not extracted] — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11728245/)
  - Conflict/ambiguity: one summariser pass said A+D+R achieved the highest precision overall, while the quoted text says "the combination of both" (reflectivity + range) is highest on average. The final model in Table 3 uses A+D+R. The exact Figure 5 values need to be read off the figure.
- Table 3 (mask metrics, A+D+R input): Ours-M AP50 94.93%, AP75 92.13%, mAP50:95 81.58%. [paper-read] — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11728245/)
- Table 2 (3D boxes after lifting the 2D masks to 3D): Ours-M AP50 79.01%, AP75 29.92%, mAP 38.03% at 9.88 ms. The paper reports this as outperforming CenterPoint, SECOND, PV-RCNN, PointPillars and PointRCNN. Post-processing improved AP50 by 8.98% and mAP50:95 by 37.03% over the raw CNN output. [paper-read] — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11728245/)

**SnowPole Detection (NTNU; Bavirisetti, Rafiq, Tabassum, Kiss, Lindseth)**
- Dataset paper: Data in Brief 59 (2025) 111403.
  - Sensor: Ouster OS2-128 at 1024×128.
  - Four modalities: Near-IR, Signal, Reflectivity and Range.
  - Colour composite: "Near-IR, Signal, and Reflectivity mapped to the blue, green, and red channels, respectively", with Range excluded.
  - 1,954 images (1,367 train / 390 val / 197 test), YOLO-format labels, no baselines in the dataset paper.

  [paper-read] — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11925094/); [DiVA PDF](http://www.diva-portal.org/smash/get/diva2:1943199/FULLTEXT01.pdf)
- Extended evaluation (FAIEMA 2025; SSRN 5386946):
  - Models: YOLOv5s, YOLOv7-tiny, YOLOv8n, YOLOv9t, YOLOv10n and YOLOv11n.
  - Inputs: single channels (Reflectance, Signal, Near-IR) and "six pseudo-color combinations".
  - Results: pseudo-colour combinations "particularly those fusing Near-Infrared, Signal, and Reflectance channels, outperformed single modalities across most configurations, achieving the highest Rank Scores and mAP metrics". The authors recommend "Combination 4 and Combination 5". YOLOv9t had the highest accuracy and YOLOv11n the best accuracy/speed trade-off.
  - Rank Score is a composite of accuracy and GPU latency.

  [abstract-level; no numbers or per-combination channel mappings were available on the GitHub README or the Mendeley page] — [GitHub](https://github.com/MuhammadIbneRafiq/Extended-evaluation-snowpole-lidar-dataset); [Mendeley v3](https://data.mendeley.com/datasets/tt6rbx7s3h/3); [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5386946)

**LiCAR: pseudo-RGB LiDAR image for CAR segmentation (Páez-Ubieta, Velasco-Sánchez, Puente; Univ. Alicante), arXiv:2501.13960, Jan 2025 (preprint)**
- Input: Ouster OS1-128 pseudo-RGB with calibrated reflectivity → R, NIR → G, signal → B. Range was not used in the image. [paper-read] — [arXiv HTML](https://arxiv.org/html/2501.13960)
- Models: YOLOv5m, YOLOv7 and YOLOv8 n/s/m/l, all with "transfer learning from previously trained models provided by each NN model". Training: YOLOv8 for 100 epochs at batch 256; YOLOv5 for 300 epochs at batch 128. [paper-read] — [arXiv HTML](https://arxiv.org/html/2501.13960)
- Results (car class):

  | Model / output | P | R | mAP50 | mAP50-95 |
  |---|---|---|---|---|
  | YOLOv8-L box | 0.88 | 0.811 | 0.881 | 0.649 |
  | YOLOv8-M box | 0.824 | 0.832 | 0.888 | 0.654 |
  | YOLOv8-L mask | 0.815 | 0.751 | 0.826 | 0.509 |

  YOLOv8-L inference: 28.7 ms. [paper-read; table numbers not captured] — [arXiv HTML](https://arxiv.org/html/2501.13960)
- There is no single-channel vs pseudo-RGB ablation, so the paper gives no evidence that fusion helps. [paper-read] — [arXiv HTML](https://arxiv.org/html/2501.13960)

**Dai, Le Gentil, Vidal-Calleja (UTS), "Connecting the Dots for Real-Time LiDAR-based Object Detection with YOLO"**
- Method: Delaunay-triangulation upsampling of spinning-LiDAR scans into dense 3-channel images. Each channel can be "independently associated with depth, inverse depth, or intensity information". [paper-read, local pdftotext] — [UTS OPUS PDF](https://opus.lib.uts.edu.au/rest/bitstreams/9c4b4e78-897e-452c-ac3f-39ce4ec2aafc/retrieve)
- Channel choice: "After a qualitative analysis of several parameter sets, we opted for the following configuration that converts LiDAR information into dense images without histogram equalisation": R = intensity, G = inverse depth, B = depth. No quantitative channel ablation was reported. [paper-read] — [UTS OPUS PDF](https://opus.lib.uts.edu.au/rest/bitstreams/9c4b4e78-897e-452c-ac3f-39ce4ec2aafc/retrieve)
- Table 2 (KITTI plus their own VLP-16 data, person/vehicle):

  | Metric | LiDAR-only | RGB-only |
  |---|---|---|
  | mAP | 81.27% | 80.67% |
  | Average IoU | 65.11% | 63.15% |
  | F1 | 0.82 | 0.82 |

  [paper-read] — [UTS OPUS PDF](https://opus.lib.uts.edu.au/rest/bitstreams/9c4b4e78-897e-452c-ac3f-39ce4ec2aafc/retrieve)
- Year and venue were not confirmed from the PDF text. It appears to be a ~2018–2019 conference paper. The extracted text cites Asvadi et al. 2018a/b.

**Asvadi, Garrote, Premebida, Peixoto, Nunes, "Multimodal vehicle detection: fusing 3D-LIDAR and color camera data", Pattern Recognition Letters 2018**
- Method: separate YOLO detectors on a colour image (YOLO-C), an upsampled dense depth map (YOLO-D) and an upsampled dense reflectance map (YOLO-R), combined by late fusion (re-scoring plus NMS / a learned ANN). The fusion reportedly "outperforms each individual one" on KITTI. [abstract-level; the PDF returned 403, so no numbers were obtained] — [ResearchGate](https://www.researchgate.net/publication/320089205_Multimodal_vehicle_detection_Fusing_3D-LIDAR_and_color_camera_data); [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167865517303598)
- Note: this fusion includes a camera, so it is not a LiDAR-only channel fusion. It is still the clearest example of a late-fusion / ensemble design over per-channel LiDAR detectors (D and R).

**Melotti, Asvadi, Premebida, "CNN-LIDAR pedestrian classification: combining range and reflectance data", IEEE ICVES 2018**
- Task: classification of pedestrian vs non-pedestrian on LiDAR dense depth and reflectance maps, comparing early and late fusion. The fusion schemes reportedly "greatly increase" accuracy. [abstract-level; numbers not retrieved] — [IEEE Xplore](https://ieeexplore.ieee.org/document/8519497/)

**Practitioner sources (Ouster)**
- Ouster blog: YOLOv5s fine-tuned from COCO weights on the reflectivity channel only. Preprocessing was destagger plus uint8 conversion. Range was used only to compute distance, not for detection. Rationale for choosing reflectivity: signal "varies with range" and near-IR "varies with sunlight levels", whereas reflectivity is consistent. No metrics were given, only a qualitative note that the fine-tuned model detected both people while the COCO model found one person at about 42% confidence. — [Ouster blog](https://ouster.com/insights/blog/object-detection-and-tracking-using-deep-learning-and-ouster-python-sdk)
- Ouster community thread "Running YOLO on 2D LidarScans using the SDK": per the search snippet, YOLO is run twice per frame, on NEAR_IR and on REFLECTIVITY, because "depending on the scene, either channel can outperform the other". This is a late, per-channel approach with no fusion and no numbers. [snippet-level; the page fetch failed] — [Ouster community](https://community.ouster.com/t/running-yolo-on-2d-lidarscans-using-the-sdk/73)

**Other single-channel LiDAR-as-camera detection**
- Sier et al. (Turku), "UAV Tracking with Lidar as a Camera Sensor in GNSS-Denied Environments", 2023: custom YOLOv5 on Ouster **signal** images only, used to initialise point-cloud UAV tracking. The conclusion lists "fusing the Ouster images (depth, signal, and ambient), point clouds, and conventional RGB images" as future work. [paper-read, local pdftotext] — [arXiv:2303.00277](https://arxiv.org/pdf/2303.00277)
- An improved YOLOv10 for ground targets in UAV LiDAR range images (Electronics 2026, 15(1):211) reports "88.96% mAP at 54.2 FPS". [snippet-level; MDPI blocked fetch] — [DOI](https://doi.org/10.3390/electronics15010211)

**Non-detection but directly relevant channel ablation: Tampuu et al., "LiDAR-as-Camera for End-to-End Driving", Sensors 2023, 23(5):2845 (arXiv:2206.15170)**
- Input: Ouster OS1-128 surround image with R = intensity, G = depth, B = ambient (Fig. 3 caption), cropped to 258×66×3. [paper-read, local pdftotext] — [arXiv](https://arxiv.org/pdf/2206.15170)
- Table II (on-policy steering in November):

  | Input | Interventions | Distance driven |
  |---|---|---|
  | 3-channel model | 0 | 8491.6 m |
  | Intensity-only | 2 | 8446.2 m |
  | Depth-only | 22 (run interrupted) | 1679.0 m |
  | Ambient-only | 19 (run interrupted) | 329.5 m |

  The authors write that depth and ambient "may nevertheless still contribute useful information to the 3-channel model". [paper-read] — [arXiv](https://arxiv.org/pdf/2206.15170)

### Inferences
- The evidence that same-sensor LiDAR channel fusion helps 2D detection/segmentation is thin. One controlled ablation exists (Casado-Coscolla 2024: one class, mostly indoor, values only in a figure), plus one abstract-level multi-model study (SnowPole extended evaluation). Both point the same way: combinations ≥ best single channel on average.
- Casado-Coscolla's pattern (reflectivity best at loose IoU, range best or equal at strict IoU, combination best on average) is the closest analogue to PoleSight's intensity-vs-range split. PoleSight's range-only beats intensity-only on 14/15 models, but intensity wins on fence_pole. That suggests the two channels carry complementary cues, as in Casado-Coscolla, and an R+D stack or a per-class ensemble is worth testing.
- "Six pseudo-color combinations" from three channels (SnowPole) equals 3! = 6. This suggests, without confirmation, that the combinations may be permutations of channel-to-RGB slot order. If so, the variation between combinations would measure how slot order interacts with ImageNet/COCO-pretrained first-layer filters. This needs the full paper to confirm.
- Fusion design in this literature is overwhelmingly early fusion by stacking 2–3 channels into the pretrained 3-channel input (Casado-Coscolla, LiCAR, SnowPole, Dai et al., Tampuu). I found no LiDAR-only dual-branch or 4+-channel 2D detector with an ablation. Late fusion appears only with a camera included (Asvadi 2018) or as informal per-channel inference (Ouster forum).

### Gaps
- No numeric values for the SnowPole extended evaluation (per-modality mAP, combination definitions). The full FAIEMA/SSRN paper was not accessible in this session.
- Exact Figure 5 values from Casado-Coscolla 2024 (A, D, R, A+D, A+R, D+R, A+D+R) could not be extracted as text.
- No numbers for Asvadi 2018 (YOLO-D vs YOLO-R vs fusion) or Melotti 2018 because of paywall/403.
- No study found on thin/tiny objects (poles at a few pixels wide) comparing channels.

## Q2. How were channels mapped to the 3 inputs of ImageNet/COCO-pretrained models, how was 16-bit data handled and normalised, and did pretrained weights help?

### Takeaway
The standard practice is to:
1. Compress each 16/32-bit LiDAR field to 8 bits, by histogram equalisation (reflectivity, ambient), fixed-max linear clamping (range), or plain uint8 casting.
2. Stack up to three fields into the RGB slots of a COCO- or ImageNet-pretrained model, which is then fine-tuned.

Channel-to-slot assignment varies between papers and was never justified quantitatively. No study found compared 16-bit vs 8-bit inputs, normalisation schemes, or pretrained vs from-scratch training on LiDAR images.

### Cited Findings
- Casado-Coscolla 2024 mapping and normalisation:
  - "we stack the three channels provided by the sensor (range, reflectivity, and ambient) into the three input channels originally intended for RGB intensities".
  - "both reflectivity and ambient channels are normalized using a histogram equalization, while for the range channel, we clamp and normalize it using a fixed maximum distance".
  - YOLOv8-seg with pretrained weights.

  [paper-read] — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11728245/)
- Tampuu 2023: depth in 0–50 m is "mapped linearly to the values 255 to 0, i.e 20 cm depth resolution. All distances beyond 50 meters are marked as 0." The other channels got no further processing. Slots: R = intensity, G = depth, B = ambient. [paper-read] — [arXiv](https://arxiv.org/pdf/2206.15170)
- Dai et al. (UTS) slots: R = intensity, G = inverse depth, B = depth.
  - The pipeline includes optional "independent histogram equalisation of I's channels", but the chosen configuration was "without histogram equalisation", based on qualitative analysis.
  - YOLO (Darknet) used ImageNet-pretrained weights and was fine-tuned.
  - They argue that transfer learning makes the method "data efficient" and that LiDAR-only matched RGB-only (Table 2: mAP 81.27% vs 80.67%). There was no from-scratch baseline.

  [paper-read] — [UTS OPUS PDF](https://opus.lib.uts.edu.au/rest/bitstreams/9c4b4e78-897e-452c-ac3f-39ce4ec2aafc/retrieve)
- LiCAR 2025 used R = reflectivity, G = NIR, B = signal with transfer learning from each model's provided pretrained weights. The summariser found no explicit normalisation or 16-bit handling described. [paper-read] — [arXiv HTML](https://arxiv.org/html/2501.13960)
- SnowPole used R = reflectivity, G = signal, B = NIR via the Ouster Python SDK. Bit depth and normalisation are not specified in the dataset paper. [paper-read] — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11925094/)
- Ouster blog: destagger, then convert to uint8; YOLOv5s fine-tuned from COCO weights. The COCO model run directly (zero-shot) did worse qualitatively (one person at about 42% confidence vs both people detected after fine-tuning). — [Ouster blog](https://ouster.com/insights/blog/object-detection-and-tracking-using-deep-learning-and-ouster-python-sdk)
- Turku Sensors 2023 preprocessing:
  - Resized images to 1000×300 with linear interpolation and applied a box filter for denoising.
  - The authors attribute low performance partly to "the high distortion in the untraditional image ratio". The summariser listed native sizes of 2048×128 for OS1-64 and 2048×64 for OS0-128. These appear swapped by the extraction, since a 64-beam sensor should give 64 rows; verify against the paper.
  - Pretrained COCO-type models were used without retraining, which shows the zero-shot transfer works on signal images.

  [paper-read] — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10058223/)
- Colour-mapping alternatives: a Remote Sensing 2022 paper (14(3):645) states, per a snippet, that "some studies have shown that the detection performance of converting depth information to a jet colormap is better than that of converting it to HHA", although HHA carries more information. It also notes that many methods encode grayscale as pseudo-colour (jet) for 3-channel networks. [snippet-level] — [MDPI](https://www.mdpi.com/2072-4292/14/3/645)

### Inferences
- For PoleSight's 16-bit 1024×128 images, the literature gives two default recipes:
  - histogram equalisation for intensity-type channels, as in Casado-Coscolla;
  - fixed-max linear clamping for range, as in Casado-Coscolla and Tampuu.

  Neither was compared against alternatives such as percentile scaling or log range. These are untested defaults, not established best practice.
- Pretrained weights are used everywhere and fine-tuning beats zero-shot (Ouster blog, qualitative). No paper isolates the pretraining benefit against from-scratch training on LiDAR images, which leaves a gap PoleSight could fill.

### Gaps
- No quantitative comparison found of:
  - 8-bit vs 16-bit or float input;
  - histogram equalisation vs linear vs log normalisation;
  - grayscale-replicated vs stacked channels;
  - channel-to-slot permutations (the SnowPole numbers might cover this but were not obtained);
  - pretrained vs scratch training for LiDAR images.

## Q3. Are there works detecting infrastructure (poles, signs, trees, road furniture) from LiDAR range/intensity images with 2D detectors, including MLS/panoramic range images?

### Takeaway
Very few were found:
- **SnowPole** (NTNU) is the closest match: YOLO detection of roadside snow poles in Ouster OS2-128 1024×128 LiDAR images, with single-channel vs pseudo-colour comparisons.
- **Dong et al. 2022** segments poles in range images with SalsaNext, but as semantic segmentation for localisation, not a COCO-style detector.

MLS road-asset inventory work instead mostly uses point-cloud methods, intensity thresholding, or camera panoramas with Mask R-CNN/YOLO. I found no MLS work that runs 2D instance segmenters on rendered MLS range/intensity images apart from PoleSight itself.

### Cited Findings
- SnowPole Detection dataset:
  - Labelled LiDAR images for snow pole detection, captured with an OS2-128 on an autonomous vehicle platform in mountainous, open and forested areas.
  - 1,954 images in 1024×128 Ouster image modalities, with YOLO labels.

  [paper-read] — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11925094/)
- SnowPole extended evaluation: six lightweight YOLO models; pseudo-colour NIR/signal/reflectivity combinations beat single modalities. [abstract-level] — [Mendeley v3](https://data.mendeley.com/datasets/tt6rbx7s3h/3); [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5386946)
- Dong, Chen, Särkkä, Stachniss, "Online pole segmentation on range images for long-term LiDAR localization in urban environments", Robotics and Autonomous Systems 2022 (arXiv:2208.07364):
  - A geometric extractor runs on range images (range plus x, y, z per pixel). Its output serves as pseudo-labels to retrain SalsaNext for binary pole/non-pole segmentation on range images.
  - Table 1:

    | Dataset | Method | Precision | Recall | F1 |
    |---|---|---|---|---|
    | NCLT | Learned (Ours-L) | 0.675 | 0.674 | 0.674 |
    | NCLT | Geometric (Ours-G) | 0.765 | 0.657 | 0.706 |
    | NCLT | Schaefer baseline | 0.690 | 0.386 | 0.495 |
    | SemanticKITTI | Learned (Ours-L) | 0.607 | 0.582 | 0.594 |
    | SemanticKITTI | Geometric (Ours-G) | 0.687 | 0.439 | 0.515 |
    | SemanticKITTI | Schaefer baseline | 0.621 | 0.380 | 0.455 |

  - The extracted text showed no input-channel ablation (e.g., with/without remission).

  [paper-read, local pdftotext] — [arXiv](https://arxiv.org/abs/2208.07364); [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0921889022001725)
- MLS road-object reviews and traffic-sign work mainly use intensity thresholding on point clouds, because retroreflective sign sheeting gives high intensity. Another route is camera panoramas plus deep detectors, with results then projected to points. [abstract-level] — [Remote Sensing review 2018](https://doi.org/10.3390/rs10101531); [Traffic signage, Sensors 2023 (PMC9964076)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9964076/)
- WHU-Infra3D (2026) is a roadside-infrastructure benchmark that pairs panoramic imagery with LiDAR. Its "2D detection" task is on panoramic camera imagery according to the abstract. No LiDAR range/intensity-image detection baselines are mentioned. [abstract-level] — [arXiv:2606.09882](https://arxiv.org/abs/2606.09882)
- A hybrid MMS pipeline segments pole-like objects using cubemap-decomposed camera panoramas with Mask R-CNN, then links the results to point clouds. This is a camera-based, not LiDAR-image-based, 2D segmenter. [abstract-level] — [ResearchGate](https://www.researchgate.net/publication/352800858_AUTOMATIC_DETECTION_AND_VECTORIZATION_OF_LINEAR_AND_POINT_OBJECTS_IN_3D_POINT_CLOUD_AND_PANORAMIC_IMAGES_FROM_MOBILE_MAPPING_SYSTEM)

### Inferences
- PoleSight fills a gap: no published MLS range/intensity-image instance-segmentation benchmark for multi-class pole infrastructure was found.
- SnowPole is the nearest comparator, but it is a single class, uses vehicle-mounted Ouster images rather than MLS renderings, and does detection rather than masks.
- Because retroreflective sign plates are strongly visible in intensity, an intensity channel may help sign-bearing classes such as gantry signs. This is an inference from the MLS sign-extraction literature, not tested in a 2D detector.

### Gaps
- No paper found applying YOLO-seg, Mask R-CNN, DETR or SAM to rendered MLS panoramic range/intensity images of road furniture.
- No pole/infrastructure study found with a channel ablation and published numbers.

## Q4. Any results showing a channel combination hurting performance, or no gain?

### Takeaway
No paper was found that reports a same-sensor channel combination being clearly worse than the best single channel for 2D detection. There are weaker signals:
- SnowPole's combinations win only "across most configurations", which implies some model/combination cases where they do not.
- Casado-Coscolla finds the ambient channel adds the least and is noisy.
- Tampuu shows depth and ambient are individually poor.
- Turku found non-signal channels worked poorly out of the box.

Ouster's own guidance avoids fusion and picks one channel, or runs per-channel detectors, because channel quality depends on the scene.

### Cited Findings
- Casado-Coscolla 2024:
  - Ambient "contribut[es] the least" and is "noisy" indoors.
  - Reflectivity+range is described as highest "on average". "On average" leaves open that some metric/model cells favour a single channel or pair; values are in Figure 5.
  - The summariser did not find any combination clearly worse than its constituents.

  [paper-read] — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11728245/)
- SnowPole extended evaluation: combinations outperform single modalities "across most configurations", not all. [abstract-level] — [Mendeley v3](https://data.mendeley.com/datasets/tt6rbx7s3h/3)
- Tampuu 2023 Table II: depth-only and ambient-only runs were stopped for too many interventions (22 in 1679.0 m and 19 in 329.5 m), while intensity-only (2 in 8446.2 m) came close to the 3-channel model (0 in 8491.6 m). The marginal gain from adding depth+ambient to intensity was therefore small in this task. [paper-read] — [arXiv](https://arxiv.org/pdf/2206.15170)
- Turku Sensors 2023: depth, near-IR and reflectivity "did not perform as well with out-of-the-box DL models without further preprocessing", so the authors used signal only. [paper-read] — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10058223/)
- Ouster forum: run YOLO separately on NEAR_IR and REFLECTIVITY because "either channel can outperform the other" depending on the scene. [snippet-level] — [Ouster community](https://community.ouster.com/t/running-yolo-on-2d-lidarscans-using-the-sdk/73)
- Ouster blog: near-IR depends on sunlight and signal depends on range, so reflectivity alone was chosen. — [Ouster blog](https://ouster.com/insights/blog/object-detection-and-tracking-using-deep-learning-and-ouster-python-sdk)

### Inferences
- The likely failure mode of fusion is adding a noisy or illumination-dependent channel (ambient/NIR) into a 3-channel slot. Weak channels add noise and use up pretrained capacity.
- PoleSight's two channels (intensity, range) are both active-sensing and pixel-registered, so they look more like Casado-Coscolla's R+D case (complementary, best on average) than the ambient case.
- Per-class effects are plausible. PoleSight already sees intensity winning on fence_pole, so a fused model could trade some classes against others. Per-class reporting is warranted, because no paper found reports per-class channel effects.

### Gaps
- No paper found that explicitly reports and analyses a negative fusion result with numbers for LiDAR-image 2D detection.
- No per-class channel-effect analysis was found in any reviewed work.
