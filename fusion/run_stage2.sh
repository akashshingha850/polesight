#!/usr/bin/env bash
# Stage 2: two-stream mid-level fusion and its R+R control, warm-started from the Stage 1
# checkpoints. Needs RRR and III for every seed. Finished runs are skipped.
cd "$(dirname "$0")/.." && source fusion/env.sh
$PY fusion/train_mid.py --modes IR RR --seeds 0 1 2 >> fusion/logs/stage2.log 2>&1
