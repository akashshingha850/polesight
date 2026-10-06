#!/usr/bin/env python3
"""Train the two-stream mid-level fusion model (dual_stream.py) and its control.

    IR   main stream = range model (RRR, seed s), aux stream = intensity backbone (III, seed s),
         fed the packed IRI images: main reads channel 1 (range), aux reads channel 0 (intensity)
    RR   control: the same architecture, but the aux stream is the range backbone of a *different*
         seed (RRR, seed s+1) and both streams read range. Separates "two modalities" from
         "twice the parameters, extra epochs and a second range model" (fusion_sota.md section 4).

Both warm-start from their Tier-0 checkpoints (train_fused.py), so the fusion convolutions are
the only new parameters, and both fine-tune with the same recipe: AdamW, lr0 0.0005, 1 warm-up
epoch, 50 epochs, close_mosaic 10, colour augmentation off, batch 16. A fused model has therefore
seen 100 + 50 epochs; compare it against RR (same budget), not only against the 100-epoch rows.

    python fusion/train_mid.py --modes IR RR --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "disabled")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ultralytics import YOLO, settings  # noqa: E402

import dual_stream  # noqa: E402  (must be imported as `dual_stream`: checkpoints pickle it by name)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "fusion" / "data"
RUNS = ROOT / "fusion" / "runs"
N_SEEDS = 3


def best(model: str, layout: str, imgsz: int, seed: int) -> Path:
    return RUNS / f"{model}_{layout}_{imgsz}_s{seed}" / "train" / "weights" / "best.pt"


def run_one(mode: str, seed: int, model: str, imgsz: int, epochs: int, lr0: float, device: str, seeds: list[int]) -> None:
    name = f"{model}_MID{mode}_{imgsz}_s{seed}"
    out = RUNS / name
    if (out / "eval.json").exists():
        print(f"[skip] {name}")
        return
    main_ckpt = best(model, "RRR", imgsz, seed)
    if mode == "IR":
        aux_ckpt, layout, chans = best(model, "III", imgsz, seed), "IRI", (1, 0)
    else:
        aux_ckpt, layout, chans = best(model, "RRR", imgsz, seeds[(seeds.index(seed) + 1) % len(seeds)]), "RRR", (0, 1)
    for p in (main_ckpt, aux_ckpt):
        if not p.exists():
            print(f"[wait] {name}: missing {p}")
            return
    dual_stream.DualStreamTrainer.main_ckpt = str(main_ckpt)
    dual_stream.DualStreamTrainer.aux_ckpt = str(aux_ckpt)
    dual_stream.DualStreamTrainer.chans = chans
    data = str(DATA / f"data_{layout}.yaml")
    start = time.time()
    if not (out / "train" / "weights" / "best.pt").exists():
        trainer = dual_stream.DualStreamTrainer(overrides=dict(
            model=f"{model}.yaml", data=data, task="segment", epochs=epochs, patience=epochs, batch=16,
            imgsz=imgsz, seed=seed, deterministic=True, workers=6, cache=True, plots=False, val=True,
            device=device, optimizer="AdamW", lr0=lr0, warmup_epochs=1, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
            project=str(out), name="train", exist_ok=True))
        trainer.train()
    train_h = (time.time() - start) / 3600
    w = out / "train" / "weights" / "best.pt"
    val = YOLO(str(w)).val(data=data, split="test", imgsz=imgsz, batch=16, half=False, plots=False,
                           verbose=False, device=device, project=str(out), name="test", exist_ok=True)
    rec = {"run": name, "model": model, "layout": layout, "arm": f"MID{mode}", "imgsz": imgsz, "seed": seed,
           "epochs": epochs, "lr0": lr0, "main_ckpt": str(main_ckpt), "aux_ckpt": str(aux_ckpt),
           "train_hours": round(train_h, 3),
           "box": {"mAP50-95": float(val.box.map), "mAP50": float(val.box.map50)},
           "mask": {"mAP50-95": float(val.seg.map), "mAP50": float(val.seg.map50),
                    "per_class_mAP50-95": [float(x) for x in val.seg.maps]},
           "speed_ms": {k: float(v) for k, v in val.speed.items()}, "weights": str(w)}
    (out / "eval.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(f"[done] {name}: mask mAP50-95={rec['mask']['mAP50-95'] * 100:.2f} ({train_h:.2f} h)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", nargs="+", default=["IR", "RR"], choices=["IR", "RR"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--model", default="yolo11m-seg")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr0", type=float, default=0.0005)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    settings.update({"wandb": False})
    for seed in a.seeds:
        for mode in a.modes:
            run_one(mode, seed, a.model, a.imgsz, a.epochs, a.lr0, a.device, a.seeds)


if __name__ == "__main__":
    main()
