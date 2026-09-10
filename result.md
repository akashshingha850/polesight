# intensity_filtered vs range_filtered

Paired comparison of the two modality sweeps. Every figure is generated from the evaluation records in `results/` and `results_range_filtered/` by `compare_modalities.py`, which reads them through `make_table.load_cells` — the same loader behind the published table.

## What was compared

| | intensity_filtered | range_filtered |
|---|---|---|
| data.yaml | ./data/data.yaml | ./data/data_range_filtered.yaml |
| modality | intensity_filtered | range_filtered |
| train / val / test | 870 / 109 / 109 | 870 / 109 / 109 |
| epochs | 100 | 100 |
| seed | 0 | 0 |
| imgsz | 640 | 640 |
| optimizer | AdamW | AdamW |
| ultralytics | 8.4.19 | 8.4.19 |
| torch | 2.10.0+cu128 | 2.10.0+cu128 |
| GPU | NVIDIA GeForce RTX 2080 Ti | NVIDIA GeForce RTX 2080 Ti |

The two sweeps come from configs that differ in three keys — the data yaml, the run directory and the W&B project. Everything that touches what the model learns is identical, so the modality is the only variable.

## Training-condition parity

All 15 paired runs completed the same number of epochs at the same realised batch size with the same optimizer. The OOM backoff resolved identically on both modalities, so every row below is a like-for-like comparison.

Total training time: 10.7 h (intensity_filtered), 10.5 h (range_filtered).

## Headline: mask mAP50-95 across the 15 checkpoints

The unit of analysis is the trained checkpoint, not the table row. YOLO26 is scored through two heads from one checkpoint, so counting all 20 rows would double-weight it; these 15 rows are independent runs.

- **range_filtered wins 14 of 15** paired comparisons (1 loss)
- Mean delta **+1.28** points, median **+1.14**
- Sign test p < 0.001; Wilcoxon signed-rank p < 0.001
- Box mAP50-95 moves the same way and further: **range_filtered wins 15 of 15**, mean **+3.60** points (p < 0.001)

| Model | intensity_filtered | range_filtered | Δ |
|---|---:|---:|---:|
| `yolo11n-seg` | 8.54 | 11.16 | +2.62 |
| `yolo11x-seg` | 8.63 | 10.97 | +2.34 |
| `yolo11s-seg` | 8.79 | 11.11 | +2.32 |
| `yolo11l-seg` | 9.75 | 11.93 | +2.18 |
| `yolov8n-seg` | 8.09 | 10.26 | +2.18 |
| `yolo11m-seg` | 9.31 | 10.70 | +1.39 |
| `yolo26l-seg` | 7.93 | 9.10 | +1.16 |
| `yolo26m-seg` | 8.16 | 9.30 | +1.14 |
| `yolo26s-seg` | 8.37 | 9.49 | +1.12 |
| `yolov8s-seg` | 9.16 | 10.10 | +0.94 |
| `yolo26n-seg` | 7.82 | 8.62 | +0.80 |
| `yolov8m-seg` | 10.08 | 10.55 | +0.47 |
| `yolov8l-seg` | 10.43 | 10.79 | +0.35 |
| `yolov8x-seg` | 10.30 | 10.65 | +0.35 |
| `yolo26x-seg` | 8.08 | 7.96 | -0.12 |

## Every evaluated head

| Group | Variant | Box mAP50-95 | | Δ | Mask mAP50-95 | | Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| | | intensity_filtered | range_filtered | | intensity_filtered | range_filtered | |
| YOLOv8-seg (NMS) | n | 19.26 | 23.91 | +4.64 | 8.09 | 10.26 | +2.18 |
| YOLOv8-seg (NMS) | s | 22.91 | 26.47 | +3.56 | 9.16 | 10.10 | +0.94 |
| YOLOv8-seg (NMS) | m | 22.82 | 26.30 | +3.47 | 10.08 | 10.55 | +0.47 |
| YOLOv8-seg (NMS) | l | 23.37 | 26.54 | +3.17 | 10.43 | 10.79 | +0.35 |
| YOLOv8-seg (NMS) | x | 23.41 | 26.56 | +3.15 | 10.30 | 10.65 | +0.35 |
| YOLO11-seg (NMS) | n | 20.12 | 24.18 | +4.06 | 8.54 | 11.16 | +2.62 |
| YOLO11-seg (NMS) | s | 22.37 | 25.26 | +2.89 | 8.79 | 11.11 | +2.32 |
| YOLO11-seg (NMS) | m | 23.15 | 26.34 | +3.19 | 9.31 | 10.70 | +1.39 |
| YOLO11-seg (NMS) | l | 24.09 | 26.18 | +2.10 | 9.75 | 11.93 | +2.18 |
| YOLO11-seg (NMS) | x | 23.05 | 27.95 | +4.90 | 8.63 | 10.97 | +2.34 |
| YOLO26-seg (NMS) | n | 20.73 | 24.95 | +4.23 | 6.95 | 9.45 | +2.50 |
| YOLO26-seg (NMS) | s | 23.94 | 28.87 | +4.93 | 7.77 | 9.52 | +1.75 |
| YOLO26-seg (NMS) | m | 25.81 | 29.02 | +3.21 | 8.09 | 9.17 | +1.09 |
| YOLO26-seg (NMS) | l | 25.96 | 28.49 | +2.52 | 8.03 | 9.59 | +1.56 |
| YOLO26-seg (NMS) | x | 25.42 | 29.21 | +3.79 | 7.66 | 9.18 | +1.52 |
| YOLO26-seg (NMS-free) | n | 19.37 | 24.33 | +4.96 | 7.82 | 8.62 | +0.80 |
| YOLO26-seg (NMS-free) | s | 23.25 | 28.69 | +5.45 | 8.37 | 9.49 | +1.12 |
| YOLO26-seg (NMS-free) | m | 25.85 | 28.40 | +2.55 | 8.16 | 9.30 | +1.14 |
| YOLO26-seg (NMS-free) | l | 25.28 | 27.52 | +2.24 | 7.93 | 9.10 | +1.16 |
| YOLO26-seg (NMS-free) | x | 24.06 | 27.74 | +3.68 | 8.08 | 7.96 | -0.12 |

## Per class, mask mAP50-95

Averaged over the 15 checkpoints. The classes differ sharply in mask size, so a modality that helps thin targets need not help everywhere.

| Class | intensity_filtered | range_filtered | Δ | Wins |
|---|---:|---:|---:|---:|
| `fence_pole` | 8.91 | 7.09 | -1.82 | 2/15 |
| `gantry_sign_pole` | 9.98 | 12.54 | +2.56 | 14/15 |
| `light_pole` | 12.68 | 14.46 | +1.78 | 14/15 |
| `traffic_pole` | 4.01 | 6.63 | +2.62 | 15/15 |

**This class moves against the overall result:** `fence_pole` (-1.82, 2/15). The aggregate delta is not uniform across classes, so a single headline number hides a real disagreement — worth reporting per class rather than only in the mean.

## How much these numbers can carry

- **One seed per cell.** Both sweeps are seed 0. A per-model delta is a single paired observation and is not resolved against run-to-run variance. The 15-way sign test is the claim that survives; an individual row is not.
- **Untuned by design.** Both sweeps take the installed Ultralytics version's defaults, so this compares modalities under a fixed recipe, not the best each modality could reach.
- **8-bit input.** The frames are 16-bit PNGs, but Ultralytics reads them with `cv2.IMREAD_COLOR`: 18,754 distinct values collapse to 167, replicated across three identical channels. Both modalities lose the same way, so the comparison holds, but the quantization falls on absolute distance in one rendering and on return strength in the other. The gap here survives that collapse, so quantization is not what separates the two — if anything the full-precision gap would be at least this large.
- **Same labels, same frames, same split.** Verified byte-identical, so nothing in the annotation or the split assignment can account for a delta.

## What would strengthen this

1. **Bound the seed noise.** Rerun one mid-size model (`yolo11m-seg`, 0.6 h per run) at seeds 1 and 2 on both modalities. Four runs, about 2.5 h, and the per-model deltas stop being single observations.
2. **Explain the class that disagrees** rather than averaging over it. The per-class split above is the part of this result most likely to be asked about in review.
3. **Fuse the two.** The three input channels currently hold one rendering three times. Packing intensity and range into separate channels costs nothing and needs no code change, and it is the experiment that says whether the two carry complementary information or the same information twice.

