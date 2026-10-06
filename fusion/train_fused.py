#!/usr/bin/env python3
"""Train one model per (layout, seed, imgsz) on the packed datasets.

The recipe is the published baseline's (scripts/train.py with train_default.yaml:
AdamW via optimizer=auto, 100 epochs, batch 16, stock augmentation, seed-controlled,
deterministic) with exactly the changes the research notes call for:

    * data            the packed layout (make_fused.py), not the 16-bit PNG
    * hsv_h/s/v = 0   a packed image has distinct channels, so HSV jitter would remix
                      the modalities (fusion_sota.md 1.3). Applied to every arm,
                      controls included, so augmentation is not a confound.
    * imgsz           640 (as published) or 1024 (native width, fusion_sota.md 1.1)

After training, the best checkpoint is scored on the test split with Ultralytics'
own validator and the result is written to <run>/eval.json.

    python fusion/train_fused.py --layouts III RRR IRI IRG --seeds 0 1 2
    python fusion/train_fused.py --layouts III RRR --seeds 0 --imgsz 1024
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "disabled")

from ultralytics import YOLO, settings  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "fusion" / "data"
RUNS = ROOT / "fusion" / "runs"


def run_name(model: str, layout: str, imgsz: int, seed: int) -> str:
    return f"{model}_{layout}_{imgsz}_s{seed}"


def metrics_block(m) -> dict:
    return {
        "mAP50-95": float(m.map),
        "mAP50": float(m.map50),
        "precision": float(m.mp),
        "recall": float(m.mr),
        "per_class_mAP50-95": [float(x) for x in m.maps],
    }


def train_one(model: str, layout: str, imgsz: int, seed: int, epochs: int, batch: int, device: str) -> None:
    name = run_name(model, layout, imgsz, seed)
    out = RUNS / name
    best = out / "train" / "weights" / "best.pt"
    if (out / "eval.json").exists():
        print(f"[skip] {name}: already done")
        return
    start = time.time()
    if not best.exists():
        yolo = YOLO(str(ROOT / f"{model}.pt"))
        yolo.train(
            data=str(DATA / f"data_{layout}.yaml"),
            task="segment",
            epochs=epochs,
            patience=epochs,
            batch=batch,
            imgsz=imgsz,
            seed=seed,
            deterministic=True,
            workers=6,
            cache=True,
            plots=False,
            val=True,
            device=device,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            project=str(out),
            name="train",
            exist_ok=True,
        )
    train_h = (time.time() - start) / 3600
    val = YOLO(str(best)).val(
        data=str(DATA / f"data_{layout}.yaml"),
        split="test",
        imgsz=imgsz,
        batch=16,
        half=False,
        plots=False,
        verbose=False,
        device=device,
        project=str(out),
        name="test",
        exist_ok=True,
    )
    record = {
        "run": name,
        "model": model,
        "layout": layout,
        "imgsz": imgsz,
        "seed": seed,
        "epochs": epochs,
        "batch": batch,
        "train_hours": round(train_h, 3),
        "box": metrics_block(val.box),
        "mask": metrics_block(val.seg),
        "speed_ms": {k: float(v) for k, v in val.speed.items()},
        "weights": str(best),
    }
    (out / "eval.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"[done] {name}: mask mAP50-95={record['mask']['mAP50-95'] * 100:.2f} "
          f"box mAP50-95={record['box']['mAP50-95'] * 100:.2f} ({train_h:.2f} h)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layouts", nargs="+", default=["III", "RRR", "IRI", "IRG"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--model", default="yolo11m-seg")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    settings.update({"wandb": False})
    for seed in a.seeds:
        for layout in a.layouts:
            train_one(a.model, layout, a.imgsz, seed, a.epochs, a.batch, a.device)


if __name__ == "__main__":
    main()
