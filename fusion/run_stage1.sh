#!/usr/bin/env bash
# Stage 1: single-modality controls and early fusion at the published 640 px setting,
# 4 layouts x 3 seeds (seed-major, so a partial sweep is still a balanced comparison).
# Finished runs are skipped, so this is safe to re-run after an interruption.
cd "$(dirname "$0")/.." && source fusion/env.sh
$PY fusion/train_fused.py --layouts III RRR IRI IRG --seeds 0 1 2 --imgsz 640 \
  >> fusion/logs/stage1.log 2>&1
