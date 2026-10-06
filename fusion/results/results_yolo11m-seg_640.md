# Fusion results: yolo11m-seg @ 640 px

Test split, 109 frames, seeds [0, 1, 2]. Mask and box mAP50-95 in points (mean ± std over seeds). Every row is scored through one evaluator, `fusion/fuse_eval.py`.

## Arms

| Arm | Mask mAP50-95 | Mask mAP50 | Box mAP50-95 | per-seed mask |
|---|---:|---:|---:|---|
| III | 9.56 ± 0.12 | 24.78 ± 0.91 | 23.55 ± 1.28 | 9.49, 9.69, 9.49 |
| RRR | 10.21 ± 0.48 | 28.51 ± 0.61 | 26.31 ± 0.42 | 9.77, 10.14, 10.72 |
| IRI | 10.41 ± 0.24 | 28.87 ± 0.67 | 28.64 ± 0.40 | 10.56, 10.14, 10.53 |
| IRG | 10.94 ± 0.55 | 30.62 ± 1.06 | 28.52 ± 0.81 | 11.54, 10.84, 10.45 |
| MIDIR | 11.40 ± 0.45 | 31.10 ± 1.80 | 28.32 ± 0.54 | 11.04, 11.25, 11.90 |
| MIDRR | 10.34 ± 0.09 | 29.46 ± 1.12 | 27.19 ± 0.73 | 10.45, 10.30, 10.28 |
| late-wbf I+R | 10.96 ± 0.17 | 29.85 ± 0.14 | 28.51 ± 0.78 | 11.10, 10.77, 10.99 |
| late-wbf R+R (ctrl) | 10.41 ± 0.24 | 29.35 ± 0.69 | 27.65 ± 0.25 | 10.15, 10.62, 10.45 |
| late-wbf I+I (ctrl) | 9.60 ± 0.11 | 24.93 ± 0.50 | 24.91 ± 0.52 | 9.48, 9.61, 9.71 |
| late-nms I+R | 10.69 ± 0.44 | 28.02 ± 0.28 | 27.29 ± 0.75 | 10.29, 10.62, 11.17 |
| late-nms R+R (ctrl) | 10.40 ± 0.45 | 29.02 ± 0.76 | 26.70 ± 0.10 | 9.87, 10.62, 10.69 |
| late-nms I+I (ctrl) | 9.58 ± 0.22 | 24.26 ± 0.96 | 23.55 ± 0.71 | 9.82, 9.39, 9.55 |
| late-wbf I+R class-aware (val-tuned) | 10.21 ± 0.20 | 28.58 ± 1.08 | 26.89 ± 0.41 | 10.05, 10.13, 10.44 |

## Per class, mask mAP50-95

| Arm | fence_pole | gantry_sign_pole | light_pole | traffic_pole |
|---|---:|---:|---:|---:|
| III | 11.43 ± 0.42 | 9.90 ± 1.37 | 12.82 ± 0.41 | 4.07 ± 0.80 |
| RRR | 8.42 ± 2.01 | 10.60 ± 0.44 | 13.99 ± 0.27 | 7.82 ± 0.38 |
| IRI | 8.62 ± 0.38 | 10.32 ± 0.53 | 14.62 ± 0.30 | 8.08 ± 1.22 |
| IRG | 10.15 ± 1.61 | 10.81 ± 0.29 | 14.63 ± 0.88 | 8.17 ± 1.44 |
| MIDIR | 12.51 ± 0.85 | 10.58 ± 0.77 | 15.14 ± 0.31 | 7.36 ± 1.10 |
| MIDRR | 7.88 ± 0.41 | 10.34 ± 0.72 | 14.59 ± 0.48 | 8.55 ± 0.82 |
| late-wbf I+R | 11.69 ± 0.59 | 10.81 ± 0.86 | 14.09 ± 0.48 | 7.23 ± 0.64 |
| late-wbf R+R (ctrl) | 9.25 ± 0.97 | 10.74 ± 0.72 | 13.64 ± 0.42 | 8.00 ± 0.08 |
| late-wbf I+I (ctrl) | 11.57 ± 0.14 | 9.54 ± 0.18 | 12.57 ± 0.61 | 4.71 ± 0.11 |
| late-nms I+R | 11.56 ± 1.97 | 10.87 ± 0.51 | 13.44 ± 0.36 | 6.90 ± 0.17 |
| late-nms R+R (ctrl) | 8.93 ± 1.33 | 11.08 ± 0.90 | 13.69 ± 0.22 | 7.87 ± 0.38 |
| late-nms I+I (ctrl) | 11.77 ± 0.17 | 9.73 ± 0.63 | 12.61 ± 0.25 | 4.23 ± 0.36 |
| late-wbf I+R class-aware (val-tuned) | 9.37 ± 1.13 | 10.46 ± 0.48 | 13.82 ± 0.25 | 7.17 ± 0.34 |

## Paired comparisons (seed-mean Δ in mask mAP50-95, 95% bootstrap CI over test frames)

| A − B | mask Δ | 95% CI | P(Δ>0) | box Δ | box 95% CI |
|---|---:|---|---:|---:|---|
| IRI − RRR | +0.20 | [-0.72, +1.34] | 0.710 | +2.33 | [+0.90, +3.87] |
| IRI − III | +0.85 | [-0.47, +2.28] | 0.920 | +5.09 | [+3.31, +7.17] |
| IRG − RRR | +0.73 | [-0.23, +1.92] | 0.920 | +2.22 | [+0.73, +3.68] |
| IRG − III | +1.38 | [+0.21, +2.54] | 0.987 | +4.97 | [+3.14, +6.98] |
| MIDIR − RRR | +1.19 | [+0.47, +1.87] | 0.997 | +2.01 | [+0.81, +3.21] |
| MIDIR − III | +1.84 | [+0.66, +2.93] | 0.997 | +4.77 | [+2.88, +7.19] |
| MIDRR − RRR | +0.13 | [-0.54, +0.75] | 0.603 | +0.88 | [+0.18, +1.49] |
| MIDRR − III | +0.79 | [-0.76, +2.11] | 0.850 | +3.64 | [+1.64, +5.93] |
| late-wbf I+R − RRR | +0.75 | [-0.18, +1.61] | 0.947 | +2.20 | [+1.01, +3.26] |
| late-wbf I+R − III | +1.40 | [+0.48, +2.25] | 1.000 | +4.96 | [+3.59, +6.58] |
| late-wbf R+R (ctrl) − RRR | +0.20 | [-0.14, +0.49] | 0.863 | +1.34 | [+1.03, +1.75] |
| late-wbf R+R (ctrl) − III | +0.85 | [-0.60, +2.23] | 0.873 | +4.10 | [+1.89, +6.66] |
| late-wbf I+I (ctrl) − RRR | -0.61 | [-2.22, +0.94] | 0.200 | -1.39 | [-3.76, +0.71] |
| late-wbf I+I (ctrl) − III | +0.04 | [-0.55, +0.41] | 0.457 | +1.36 | [+1.07, +1.69] |
| late-nms I+R − RRR | +0.48 | [-0.27, +1.30] | 0.907 | +0.98 | [+0.01, +2.14] |
| late-nms I+R − III | +1.14 | [+0.23, +2.14] | 0.987 | +3.74 | [+2.05, +5.72] |
| late-nms R+R (ctrl) − RRR | +0.19 | [-0.14, +0.56] | 0.847 | +0.39 | [+0.04, +0.81] |
| late-nms R+R (ctrl) − III | +0.84 | [-0.82, +2.46] | 0.847 | +3.14 | [+0.87, +5.64] |
| late-nms I+I (ctrl) − RRR | -0.62 | [-2.10, +0.97] | 0.210 | -2.76 | [-4.99, -0.68] |
| late-nms I+I (ctrl) − III | +0.03 | [-0.40, +0.38] | 0.510 | -0.00 | [-0.32, +0.45] |
| late-wbf I+R class-aware (val-tuned) − RRR | -0.00 | [-0.37, +0.38] | 0.480 | +0.58 | [+0.10, +1.12] |
| late-wbf I+R class-aware (val-tuned) − III | +0.65 | [-0.69, +1.91] | 0.827 | +3.33 | [+1.41, +5.33] |
| RRR − III | +0.65 | [-0.83, +2.03] | 0.813 | +2.76 | [+0.57, +5.07] |
| late-wbf I+R − late-wbf R+R (ctrl) | +0.55 | [-0.28, +1.41] | 0.897 | +0.86 | [-0.48, +1.94] |
| late-nms I+R − late-nms R+R (ctrl) | +0.30 | [-0.66, +1.24] | 0.797 | +0.59 | [-0.56, +1.85] |
| MIDIR − MIDRR | +1.05 | [+0.44, +1.82] | 1.000 | +1.13 | [+0.18, +2.25] |
| late-wbf I+R − IRG | +0.02 | [-1.01, +0.89] | 0.450 | -0.01 | [-1.37, +1.14] |
| MIDIR − late-wbf I+R | +0.44 | [-0.29, +1.11] | 0.920 | -0.19 | [-1.28, +1.05] |
| MIDIR − IRG | +0.46 | [-0.59, +1.36] | 0.807 | -0.21 | [-1.46, +1.01] |

## Per-class paired deltas

| A − B | class | Δ | 95% CI |
|---|---|---:|---|
| IRI − RRR | fence_pole | +0.20 | [-2.27, +3.90] |
| IRI − RRR | gantry_sign_pole | -0.29 | [-2.47, +1.81] |
| IRI − RRR | light_pole | +0.62 | [-0.61, +1.83] |
| IRI − RRR | traffic_pole | +0.26 | [-2.03, +1.58] |
| IRI − III | fence_pole | -2.82 | [-5.67, +0.71] |
| IRI − III | gantry_sign_pole | +0.41 | [-2.09, +3.20] |
| IRI − III | light_pole | +1.79 | [+0.36, +3.36] |
| IRI − III | traffic_pole | +4.02 | [+1.62, +6.65] |
| IRG − RRR | fence_pole | +1.73 | [-1.17, +5.22] |
| IRG − RRR | gantry_sign_pole | +0.21 | [-1.75, +2.42] |
| IRG − RRR | light_pole | +0.64 | [-0.54, +1.85] |
| IRG − RRR | traffic_pole | +0.35 | [-1.78, +1.71] |
| IRG − III | fence_pole | -1.29 | [-4.06, +1.99] |
| IRG − III | gantry_sign_pole | +0.91 | [-1.35, +3.50] |
| IRG − III | light_pole | +1.81 | [+0.50, +3.40] |
| IRG − III | traffic_pole | +4.11 | [+2.01, +6.61] |
| late-wbf I+R − RRR | fence_pole | +3.28 | [+0.96, +6.09] |
| late-wbf I+R − RRR | gantry_sign_pole | +0.20 | [-1.25, +1.79] |
| late-wbf I+R − RRR | light_pole | +0.10 | [-0.80, +0.82] |
| late-wbf I+R − RRR | traffic_pole | -0.59 | [-2.00, +0.63] |
| late-wbf I+R − III | fence_pole | +0.26 | [-1.71, +2.07] |
| late-wbf I+R − III | gantry_sign_pole | +0.90 | [-1.18, +2.34] |
| late-wbf I+R − III | light_pole | +1.27 | [+0.30, +2.45] |
| late-wbf I+R − III | traffic_pole | +3.16 | [+1.74, +5.46] |
| late-nms I+R − RRR | fence_pole | +3.14 | [+0.47, +5.87] |
| late-nms I+R − RRR | gantry_sign_pole | +0.27 | [-1.07, +1.96] |
| late-nms I+R − RRR | light_pole | -0.55 | [-1.55, +0.25] |
| late-nms I+R − RRR | traffic_pole | -0.92 | [-1.89, -0.08] |
| late-nms I+R − III | fence_pole | +0.12 | [-2.07, +2.06] |
| late-nms I+R − III | gantry_sign_pole | +0.97 | [-0.81, +2.86] |
| late-nms I+R − III | light_pole | +0.62 | [-0.18, +1.41] |
| late-nms I+R − III | traffic_pole | +2.84 | [+1.11, +5.43] |
| MIDIR − RRR | fence_pole | +4.09 | [+2.12, +6.12] |
| MIDIR − RRR | gantry_sign_pole | -0.02 | [-1.75, +1.59] |
| MIDIR − RRR | light_pole | +1.15 | [+0.20, +2.16] |
| MIDIR − RRR | traffic_pole | -0.46 | [-2.13, +0.52] |
| MIDIR − III | fence_pole | +1.07 | [-2.23, +4.48] |
| MIDIR − III | gantry_sign_pole | +0.68 | [-2.09, +3.00] |
| MIDIR − III | light_pole | +2.32 | [+1.22, +3.53] |
| MIDIR − III | traffic_pole | +3.29 | [+1.66, +5.60] |
| MIDRR − RRR | fence_pole | -0.54 | [-2.37, +1.07] |
| MIDRR − RRR | gantry_sign_pole | -0.26 | [-1.57, +0.93] |
| MIDRR − RRR | light_pole | +0.60 | [-0.29, +1.45] |
| MIDRR − RRR | traffic_pole | +0.73 | [-0.68, +2.05] |
| MIDRR − III | fence_pole | -3.55 | [-8.06, +0.12] |
| MIDRR − III | gantry_sign_pole | +0.44 | [-2.43, +2.99] |
| MIDRR − III | light_pole | +1.77 | [+0.51, +3.36] |
| MIDRR − III | traffic_pole | +4.49 | [+2.58, +6.92] |
| RRR − III | fence_pole | -3.02 | [-7.34, +0.83] |
| RRR − III | gantry_sign_pole | +0.70 | [-2.00, +3.52] |
| RRR − III | light_pole | +1.17 | [-0.10, +2.75] |
| RRR − III | traffic_pole | +3.76 | [+1.61, +7.24] |

Class-aware weights (weight on the intensity model, 0 = range only, 1 = intensity only; order fence, gantry, light, traffic), per seed, tuned on validation: [0.25, 0.0, 0.75, 0.25]; [0.0, 0.25, 0.75, 0.25]; [0.0, 0.0, 0.25, 0.0]
