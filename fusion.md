# Multimodal fusion on PoleSight

Research brief (quick mode). Written 2026-10-04 on branch `fusion`.

## 1. What we have

PoleSight ships the same 1,088 frames in two renderings, `intensity_filtered` and `range_filtered`. Both are 1024×128, single-channel, 16-bit PNG, and they share one label set and one split (870 / 109 / 109). A frame has the same filename in both modalities, and both come from one spherical projection, so the two images are pixel-registered by construction. There is no calibration or alignment problem, which is the hard part of most fusion work.

Facts from the repo that shape the design:

- **The modalities differ in what they help.** Per `results/result.md`, range beats intensity on mask mAP50-95 in 14 of 15 checkpoints (mean +1.28 points, box +3.60). `fence_pole` goes the other way (intensity ahead by 1.82 points, range wins only 2 of 15). That per-class disagreement is the best evidence we have that the two carry some complementary information.
- **Absolute numbers are low.** Mask mAP50-95 sits at 8–12, with 5,988 of 6,121 instances below the COCO small cutoff (median 71 px²). Fusion has to be judged against a weak and noisy baseline.
- **The baselines never saw the data properly.** Ultralytics reads the PNGs with `cv2.IMREAD_COLOR`, so 16-bit values collapse to about 167 levels and are copied into three identical channels. The report already lists "pack intensity and range into separate channels" as the obvious next experiment.
- **One seed per cell.** Per-model deltas are single observations. Any fusion gain below roughly 1 point is not interpretable without repeated seeds.
- **Only two modalities are released.** `rangegen/` can also emit a vertex map (xyz), RGB, classification and index maps, but the source point clouds are not redistributed. Extra channels would have to come from regenerating frames, which the public release cannot do.

## 2. Fusion taxonomy and what it means here

The literature groups multimodal fusion by where the modalities meet: early (inputs or raw features), intermediate (features), and late (decisions) ([Deep Multimodal Data Fusion, ACM CSUR](https://dl.acm.org/doi/full/10.1145/3649447); [multimodal fusion review](https://www.sciencedirect.com/science/article/abs/pii/S0925231225014997)). One biomedical review reports intermediate fusion beating early and late fusion in 24 works ([systematic review of intermediate fusion](https://www.sciencedirect.com/science/article/pii/S0262885625000976)). That evidence comes from biomedical data, so treat it as a prior, not a prediction for range images.

For LiDAR segmentation, range-view methods already feed intensity and range together. RangeNet++ and SalsaNext use a range image whose channels are range, x, y, z and remission ([SalsaNext](https://arxiv.org/pdf/2003.03653); [overview of projection methods](https://arxiv.org/pdf/2202.13377)). So channel-stacking is the standard input for this representation, not an exotic choice. Late fusion in LiDAR segmentation is comparatively under-explored ([AMVNet](https://arxiv.org/pdf/2012.04934)).

## 3. Candidate approaches

Ordered by cost. Each one answers a different question, so run them in this order and stop when the gain disappears into the seed noise.

### A. Early fusion, channel packing (start here)

Build a 3-channel image per frame from the two registered renderings, for example `[intensity, range, intensity]`, `[intensity, range, range]`, or `[intensity, range, |∇range|]`. Train the existing YOLO family unchanged.

- Cost: a loader change only. No architecture work.
- Needs: bypass `IMREAD_COLOR` (write 8-bit packed PNGs to a new `data/fused/` tree, or a custom dataset class) and keep the 16-bit precision question separate by also testing a percentile-normalised 8-bit packing.
- Pretrained weights expect RGB statistics. Pick channel order and normalisation deliberately and report it.
- Set `hsv_h: 0` and `hsv_s: 0`. Once the channels differ, Ultralytics' HSV augmentation remixes them for each sample (see [fusion_sota.md §1.3](fusion_sota.md)).
- Answers: do the two renderings carry complementary or redundant information?

### B. Early fusion, 4+ channel input

Widen the first conv to take intensity, range and optionally derived channels (range gradient, normal estimate, validity mask). Initialise the extra channels from the mean of the pretrained RGB filters.

- Cost: small change to the stem of each model. Ultralytics 8.4.19 reads N-channel TIFFs through `channels: N` in the data YAML, but it does **not** initialise the stem: a first conv whose shape differs from the checkpoint is left random. The stem initialisation has to be written by hand (see [fusion_sota.md §1.2](fusion_sota.md)).
- Derived channels are free to compute from range alone and may help thin vertical structure. This is a hypothesis, not a finding.

### C. Mid-level fusion, dual-stream backbone

Two backbones (one per modality) with cross-modal exchange at several scales, then a shared neck and head. YOLO variants for RGB-IR detection do this and report midway fusion as the stronger placement ([MCF-YOLO](https://doi.org/10.3390/s26123938); [GEM-YOLO](https://doi.org/10.3390/s26072035); [CFT](https://arxiv.org/pdf/2111.00273)). CMX generalises the pattern to RGB-X segmentation, including RGB-LiDAR, with a rectification module and a fusion module between two transformer streams ([CMX](https://arxiv.org/abs/2203.04838)).

- Cost: highest. Roughly double the backbone parameters, new code, and no off-the-shelf Ultralytics path.
- The domain gap is unusual. Both streams here are single-channel renderings of the same sensor, not two sensors. Gains from RGB-thermal papers need not transfer.
- Risk: with 870 training frames, a two-stream model is more likely to overfit than a packed input.

### D. Late fusion, decision level

Train one model per modality (already done: `results/` and `results_range_filtered/`). Combine predictions at test time.

- Boxes: [Weighted Boxes Fusion](https://arxiv.org/abs/1910.13302) averages confidence-weighted boxes from different models and is a drop-in for the existing checkpoints.
- Masks: no WBF equivalent for polygons. Options are per-pixel soft-voting on mask logits or mask NMS across the two models. Needs a small amount of code.
- Cost: lowest if the checkpoints exist. They are not released (1.7 GB), so they must be retrained or this runs only on the author's machine.
- Answers: is there a gain from ensembling alone? Ensemble two intensity models too, otherwise the gain cannot be credited to modality.

### E. Class-aware fusion

The `fence_pole` reversal suggests weighting modality per class. In late fusion that is a per-class weight on each model's scores, tuned on the validation split. In mid fusion it is a gating module. Try this only after D shows any gain, because tuning per-class weights on 109 validation frames is easy to overfit.

### F. Robustness training: modality dropout

If a deployed sensor can lose one channel, randomly zero a whole modality during training so the model does not lean on one input ([ModDrop](https://www.researchgate.net/publication/270454771_ModDrop_Adaptive_Multi-Modal_Gesture_Recognition) and [follow-ups](https://www.emergentmind.com/topics/modality-dropout)). Applies to A–C. Low priority unless missing-modality robustness is a stated goal.

## 4. Recommended plan

> Superseded in detail by [fusion_sota.md](fusion_sota.md) §3–4, which adds a Tier-0 step (native 1024-px resolution and colour augmentation off) before any fusion run, and a ranked set of state-of-the-art fusion architectures.

| Step | Experiment | Why first |
|---|---|---|
| 0 | Seed baseline: `yolo11m-seg` at seeds 0–2, both modalities (about 2.5 h per the existing report) | Sets the noise floor every later claim depends on |
| 1 | Approach A with two or three channel layouts, same model, same seeds | Cheapest test of complementarity |
| 2 | Approach D with WBF and mask voting, plus an intensity+intensity ensemble control | Separates "two modalities" from "two models" |
| 3 | Approach B with derived range channels | Only if step 1 shows a gain |
| 4 | Approach C on one small backbone | Only if 1–3 leave a clear gap |

Use `yolo11m-seg` for the whole ladder. It is the best of the three `m` models on range (10.70 mask mAP50-95) and cheap to train (0.6 h per run), but the choice is for speed, not because it is proven best.

## 5. Evaluation protocol

- Same split, same test frames, same metric (mask mAP50-95) as `results/`, so fused numbers go straight into `make_table.py`.
- Report mean and spread over at least three seeds. Call a gain real only if it clears the seed spread.
- Report per class. A fusion that lifts `traffic_pole` and `light_pole` but loses `fence_pole` is a different result from a uniform gain.
- Stratify by route using `data/metadata.json` (seven routes, 41 to 300 frames each) to check that fusion is not carrying one route.
- Keep controls: single-modality, duplicated-modality (`[I, I, I]` against `[I, R, I]`), and ensemble-of-same-modality.

## 6. Limitations of this brief

- This is a quick-mode scan from web search results, not a PRISMA-style review. Citations were taken from search snippets and arXiv/publisher pages, not read in full. Check each before citing it in a paper.
- The RGB-IR YOLO papers use camera pairs. None of the cited work fuses two renderings of the same LiDAR frame at this resolution (128 rows) and object scale, so expected gains are untested for this setting.
- If both renderings come from one point set, they may share most of their information, in which case early fusion will show little. Step 1 settles that.
- Fusion cannot add signal that neither rendering contains. At 71 px² median mask size, the bottleneck may be resolution and thin-structure geometry. Input-resolution and head-stride experiments belong in a separate track.

## 7. Immediate next steps

1. Write `scripts/make_fused.py` to build the packed-channel dataset and a `data/data_fused.yaml`.
2. Add `configs/train_fused.yaml` as a twin of `train_range_filtered.yaml`, changing only the data YAML and output directory.
3. Run the seed baseline and step 1.

*AI-assisted research tools (web search and drafting) were used to prepare this brief.*
