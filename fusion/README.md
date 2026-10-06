# Intensity–range fusion experiments

Code and results for fusing the two registered PoleSight renderings, `intensity_filtered` (I) and `range_filtered` (R). It implements the plan in [../fusion.md](../fusion.md) and the ranked methods in [../fusion_sota.md](../fusion_sota.md).

**Question.** Does a model that sees both renderings beat the better single-modality model (range), under identical training, resolution and augmentation?

**Status.** Complete for `yolo11m-seg` at 640 px, 3 seeds: Stage 1 (single-modality controls, early fusion, late fusion) and Stage 2 (two-stream mid fusion and its control). Full tables with bootstrap intervals: [results/results_yolo11m-seg_640.md](results/results_yolo11m-seg_640.md) (numbers in [.json](results/results_yolo11m-seg_640.json)). Not yet run: native 1024 px, other model scales.

## Model

All arms use **YOLO11m-seg** (Ultralytics 8.4.19), initialised from the COCO checkpoint `yolo11m-seg.pt`: 22.4 M parameters, 112.9 GFLOPs at 640 px. The two-stream model has 34.3 M parameters. One model scale was chosen for the whole ladder, for speed, as `fusion.md` §4 recommends; it is not claimed to be the best scale.

Training recipe = the published baseline (`configs/train_default.yaml`): 100 epochs, batch 16, `optimizer=auto` (AdamW), stock augmentation, `deterministic=True`, 640 px, `cache=True`. Changes, applied to every arm including the controls:

| Change | Why |
|---|---|
| Data read from packed 8-bit PNGs (`make_fused.py`) | Stock loading reads the 16-bit PNGs with `IMREAD_COLOR`, collapsing them to ~167 levels in three identical channels |
| `hsv_h = hsv_s = hsv_v = 0` | On packed input HSV jitter remixes the modalities per sample ([fusion_sota.md §1.3](../fusion_sota.md)) |
| Seeds 0, 1, 2 | One seed per cell cannot resolve gains below ~1 point |

Pretrained weights are used as they are; no 4-channel input is used, so the stem initialisation problem in fusion_sota.md §1.2 does not arise.

## Methods

| Arm | Fusion level | What it is | Research basis |
|---|---|---|---|
| `III`, `RRR` | none | Single-modality controls, same 8-bit encoding as the fused rows | Tier 0 |
| `IRI` | early | Channels `[I, R, I]` into the unmodified 3-channel model | fusion.md Approach A; E1 |
| `IRG` | early | `[I, R, ∂R/∂col]`: third channel is the horizontal gradient of log-range (edges of thin vertical poles) | fusion_sota.md E2 |
| late-WBF `I+R` | decision | One `III` and one `RRR` model, predictions fused at test time with Weighted Boxes Fusion. WBF has no mask counterpart, so a fused mask is the score-weighted mean of the members' mask probabilities, thresholded at 0.5. At most one detection per model per cluster, so fusing a model with itself returns it | fusion.md Approach D; Tier 2; ProbEn/WBF |
| late-NMS `I+R` | decision | Pool both models' detections, class-wise NMS (IoU 0.7). The plain ensemble | Tier 2 |
| late-WBF class-aware | decision | As late-WBF with a per-class modality weight (0, .25, .5, .75, 1) tuned on the **validation** split | fusion.md Approach E |
| `MIDIR` | mid | Two-stream YOLO11m-seg, see below | fusion_sota.md Tier 3, fusion option 1 |
| `MIDRR` | mid, control | Same architecture, both streams fed range | fusion_sota.md §4, "R ⊕ R" control |

### Encoding (`make_fused.py`)

Both renderings are pixel-registered (one spherical projection), so channels are packed per pixel. Constants are measured on the train split only (`fusion/data/encoding.json`).

- No-return pixels (intensity = 65535, about half of each frame) → 255 in every image channel; 128 in the gradient channel.
- Intensity: linear between the 0.5 / 99.5 percentiles of valid pixels → 0–254.
- Range: log between the same percentiles → 0–254 (spends levels where the poles are: close range).
- Gradient: central difference of log-range along columns, scaled by the train 99.5th percentile of |g|, 128 = flat; 0 where a neighbour is no-return.

### Two-stream mid-level fusion (`dual_stream.py`)

- **Main stream:** the stock YOLO11m-seg (backbone, neck, head), fed the range channel.
- **Aux stream:** a copy of backbone layers 0–10, fed the intensity channel.
- **Fusion:** at P3, P4, P5 (after layers 4, 6, 10) `x ← x + Conv1x1([x, aux])`, zero-initialised. At step 0 the model is exactly the range model (checked: max output difference 0.0).
- **Warm start:** main stream from the `RRR` checkpoint of the same seed, aux backbone from the `III` checkpoint of the same seed. Only the three fusion convolutions are new.
- **Fine-tune:** 50 epochs, AdamW, lr0 0.0005, 1 warm-up epoch, colour augmentation off.
- **Control `MIDRR`:** aux stream = the `RRR` backbone of the *next* seed, both streams fed range. A mid-fusion model has seen 100 + 50 epochs and has 1.5× the parameters, so the claim "two modalities help" is only supported against this control, not against the 100-epoch rows alone.

## Evaluation

- **One scorer.** `fuse_eval.py` scores every row, single, early, late and mid, through Ultralytics' own validator pieces (dataloader, NMS, mask IoU, matching) and keeps per-image TP matrices. Checked against `model.val`: identical to four decimals for single models (5.3372 / 15.986) and for the two-stream checkpoint (10.247); fusing a model with itself returns it for both WBF and NMS.
- **Split and metric.** The 109 test frames, mask mAP50-95 (box mAP and per-class alongside), the same split and metric as `results/`.
- **Statistics.** Mean ± std over seeds. Headline comparisons use a paired bootstrap over the test frames: one resample of the 109 frames is scored for every seed of both arms, and the statistic is the seed-mean difference. The interval covers test-set sampling; training variance is the seed spread.
- **Controls.** `III`/`RRR` at the same encoding; same-modality late ensembles (`R+R`, `I+I`, different seeds); `MIDRR`.

## Files

| File | Purpose |
|---|---|
| `make_fused.py` | Build packed datasets `III, RRR, IRI, IRG` under `fusion/data/` and their YAMLs |
| `train_fused.py` | Train single and early-fusion arms (layouts × seeds × imgsz) |
| `train_mid.py` | Train `MIDIR` and `MIDRR` from the Stage 1 checkpoints |
| `dual_stream.py` | Two-stream model and trainer |
| `fuse_eval.py` | Per-image evaluator, late fusion (WBF with mask averaging, NMS) |
| `analyze.py` | Scores all arms, controls, bootstrap; writes `results/results_<model>_<imgsz>.{md,json}` |
| `run_stage1.sh`, `run_stage2.sh` | The sweeps as run |
| `env.sh` | Environment (see below) |

## Reproduce

```bash
source fusion/env.sh
$PY fusion/make_fused.py
fusion/run_stage1.sh        # 12 runs, ~28 min each on an RTX 4500 Ada
fusion/run_stage2.sh        # 6 runs, after Stage 1
$PY fusion/analyze.py       # writes fusion/results/
```

`env.sh` exists because the repository's `.venv` interpreter symlink is dangling on this machine: it uses a system Python 3.10 with the venv's site-packages on `PYTHONPATH`, and a shim directory supplies the `_bz2` extension that build lacks. On a machine with a working environment, ignore it and run the scripts directly. Generated data, run trees and caches (`fusion/data`, `fusion/runs`, `fusion/cache`) are git-ignored.

## Limitations

- One model scale (`yolo11m-seg`) and, so far, one resolution (640). The native-resolution factor (1024) in fusion_sota.md §1.1 is not run yet, so fusion gains are measured at the published 640 setting and may differ at native width.
- Three seeds, 109 test frames, 78–95 instances per class: per-class results carry wide intervals.
- Both renderings come from the same LiDAR returns, so redundancy is possible; the `IRI` vs `III` and `MIDIR` vs `MIDRR` comparisons are the tests for it.
- The class-aware weights are tuned on 109 validation frames and can overfit.
- No 4-channel (CoordConv) variant and no Tier 4 geometry-prior attention were run.
- Absolute numbers are low (mask mAP50-95 about 10) and not comparable to `results/`, which used a different GPU and the stock 16-bit→8-bit loading.

## Results

`yolo11m-seg`, 640 px, test split (109 frames), seeds 0–2, mean ± std, points of mAP50-95. Δ is the seed-mean difference with a 95% paired-bootstrap interval over test frames.

| Arm | Mask mAP50-95 | Box mAP50-95 | Params | Inference |
|---|---:|---:|---:|---:|
| `III` intensity only | 9.56 ± 0.12 | 23.55 ± 1.28 | 22.4 M | 1.1 ms |
| `RRR` range only (best single modality) | 10.21 ± 0.48 | 26.31 ± 0.42 | 22.4 M | 1.1 ms |
| `IRI` early fusion | 10.41 ± 0.24 | 28.64 ± 0.40 | 22.4 M | 1.0 ms |
| `IRG` early fusion + range gradient | 10.94 ± 0.55 | 28.52 ± 0.81 | 22.4 M | 1.0 ms |
| late WBF `I+R` | 10.96 ± 0.17 | 28.51 ± 0.78 | 2 × 22.4 M | 2 models |
| late WBF `R+R` (control) | 10.41 ± 0.24 | 27.65 ± 0.25 | 2 × 22.4 M | 2 models |
| **`MIDIR` two-stream mid fusion** | **11.40 ± 0.45** | 28.32 ± 0.54 | 34.3 M | 1.6 ms |
| `MIDRR` mid fusion control (range twice) | 10.34 ± 0.09 | 27.19 ± 0.73 | 34.3 M | 1.5 ms |

Inference time is the Ultralytics validator's per-image model time on an RTX 4500 Ada at batch 16.

### Findings

1. **Mid fusion is the only arm with a mask gain that holds up.** `MIDIR − RRR` = **+1.19 [+0.47, +1.87]** and `MIDIR − MIDRR` = **+1.05 [+0.44, +1.82]**, higher at all three seeds (11.04, 11.25, 11.90 against 10.45, 10.30, 10.28 for the control). The control matters: the fused model has 1.5× the parameters and 150 epochs of training, and `MIDRR − RRR` is +0.13 [−0.54, +0.75], so extra capacity and epochs alone do not produce it. It costs about 1.5× the inference time of the single model.
2. **Early and late fusion do not show a mask gain over range alone.** `IRI − RRR` +0.20 [−0.72, +1.34], `IRG − RRR` +0.73 [−0.23, +1.92], late WBF `I+R − RRR` +0.75 [−0.18, +1.61]. All intervals include zero. `MIDIR` is not significantly better than late WBF (+0.44 [−0.29, +1.11]) or `IRG` (+0.46 [−0.59, +1.36]), so the evidence ranks mid fusion first but does not separate it from these.
3. **Box mAP improves with any fusion, about +2 points over range** (`IRI` +2.33 [+0.90, +3.87], `MIDIR` +2.01 [+0.81, +3.21]), also for the single-model early fusion that costs nothing at inference. Part of the late-fusion box gain is ensembling: two range models already give +1.34.
4. **The modalities complement each other on `fence_pole`, where range alone is weakest.** `fence_pole` mask AP: intensity 11.4, range 8.4. `MIDIR` reaches 12.5 (Δ vs range +4.09 [+2.12, +6.12]) and late WBF 11.7 (+3.28 [+0.96, +6.09]). `MIDIR` also gains on `light_pole` (+1.15 [+0.20, +2.16]) and is flat on `gantry_sign_pole` and `traffic_pole`. This agrees with the per-class reversal in `results/result.md`.
5. **Per-class modality weights tuned on validation did not help** (mask Δ vs range 0.00, box +0.58): 109 validation frames are too few, as `fusion.md` warned.
6. **Two modalities that share their returns still carry something extra.** The same-modality ensembles (`I+I`, `R+R`) gain 0.0–0.2 mask points, while the mid-fusion `MIDIR − MIDRR` gap is +1.05, so the gain is not an ensembling or capacity effect.

### How far to trust this

- Intervals cover sampling of the 109 test frames. Training variance is only the 3-seed spread, so a different set of seeds could move a mean by a few tenths.
- About 20 comparisons are reported without a multiplicity correction. The `MIDIR` comparisons were the pre-specified Tier 3 test; the others are supporting.
- The mid-fusion recipe (50 epochs, lr0 0.0005, zero-initialised fusion at P3–P5, one fusion operator) was fixed before the runs and not tuned. Neither the warm start nor the fusion operator was ablated, and the other Tier 3 fusion modules (CMX-style, cross-attention) were not tried.
- Absolute mask mAP is about 11, so a 1-point gain is about 10% relative but small in absolute terms.
- Only one model scale and the published 640 px setting were run. If native-resolution training changes how much each modality contributes, the ranking may move.
