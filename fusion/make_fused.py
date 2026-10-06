#!/usr/bin/env python3
"""Build packed 3-channel 8-bit datasets from the registered intensity/range frames.

PoleSight ships ``intensity_filtered`` and ``range_filtered`` as two renderings of
one spherical projection, so frame ``NNNN.png`` is pixel-registered in both. The
stock Ultralytics loader reads the 16-bit PNGs with ``cv2.IMREAD_COLOR``, which
collapses them to ~167 levels and copies one rendering into three identical
channels. This script replaces that with a *deliberate* 8-bit encoding per
modality and packs the two renderings into separate channels (fusion.md, Approach
A; fusion_sota.md, Tier 1).

Layouts written (channel order is the order stored in the PNG, i.e. BGR to cv2):

    III   [I, I, I]        single-modality control, same encoding as the fused rows
    RRR   [R, R, R]        single-modality control, same encoding as the fused rows
    IRI   [I, R, I]        E1: early fusion, plain
    IRG   [I, R, dR/dcol]  E2: early fusion + horizontal range gradient (pole edges)

Encodings (constants are measured on the *train* split only, saved to
``encoding.json``):

    no-return pixels   (I == 65535, i.e. range >= 64536)          -> 255 in every channel
                                                                      except the gradient (128)
    intensity          linear between the train 0.5th / 99.5th percentile of valid pixels -> 0..254
    range              log between the same percentiles of valid pixels                    -> 0..254
    range gradient     central difference of log-range along columns, scaled by the train
                       99.5th percentile of |g|, 128 = flat; 0 where a neighbour is no-return

Each layout is a complete Ultralytics dataset: ``fusion/data/<layout>/{train,valid,test}/
{images,labels}`` plus ``fusion/data/data_<layout>.yaml``. Labels are copied from the
released intensity_filtered tree (byte-identical to range_filtered).

    python fusion/make_fused.py
"""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data"
OUT = ROOT / "fusion" / "data"
SPLITS = ["train", "valid", "test"]
NAMES = ["fence_pole", "gantry_sign_pole", "light_pole", "traffic_pole"]
LAYOUTS = {"III": "III", "RRR": "RRR", "IRI": "IRI", "IRG": "IRG"}
SKY_I = 65535


def read_pair(split: str, name: str) -> tuple[np.ndarray, np.ndarray]:
    i = cv2.imread(str(SRC / split / "intensity_filtered" / "images" / name), cv2.IMREAD_UNCHANGED)
    r = cv2.imread(str(SRC / split / "range_filtered" / "images" / name), cv2.IMREAD_UNCHANGED)
    if i is None or r is None or i.dtype != np.uint16 or i.shape != r.shape:
        raise RuntimeError(f"unexpected pair for {split}/{name}")
    return i, r


def valid_mask(i: np.ndarray) -> np.ndarray:
    return i != SKY_I


def _gradient(r: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Central difference of log-range along columns, and where it is defined."""
    lr = np.log(np.maximum(r.astype(np.float32), 1.0))
    g = np.zeros_like(lr)
    g[:, 1:-1] = (lr[:, 2:] - lr[:, :-2]) * 0.5
    ok = np.zeros_like(valid)
    ok[:, 1:-1] = valid[:, 2:] & valid[:, :-2] & valid[:, 1:-1]
    return np.where(ok, g, 0.0), ok


def _stats_one(args: tuple[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split, name = args
    i, r = read_pair(split, name)
    v = valid_mask(i)
    g, ok = _gradient(r, v)
    return i[v][::3], r[v][::3], np.abs(g[ok])[::3]


def fit_encoding(names: list[str], workers: int) -> dict:
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(_stats_one, [("train", n) for n in names], chunksize=16))
    vi = np.concatenate([p[0] for p in parts]).astype(np.float64)
    vr = np.concatenate([p[1] for p in parts]).astype(np.float64)
    ag = np.concatenate([p[2] for p in parts]).astype(np.float64)
    return {
        "intensity": [float(np.percentile(vi, 0.5)), float(np.percentile(vi, 99.5))],
        "range": [float(np.percentile(vr, 0.5)), float(np.percentile(vr, 99.5))],
        "grad_scale": float(np.percentile(ag, 99.5)),
        "valid_fraction_train": float(len(vi) * 3 / (len(names) * 128 * 1024)),
        "note": "intensity linear, range log, both clipped to the train 0.5/99.5 percentiles of "
        "valid pixels; no-return pixels = 255; gradient = central difference of log-range",
    }


def encode(i: np.ndarray, r: np.ndarray, enc: dict) -> dict[str, np.ndarray]:
    v = valid_mask(i)
    lo, hi = enc["intensity"]
    ie = np.clip((i.astype(np.float32) - lo) / (hi - lo), 0, 1) * 254.0
    lo, hi = enc["range"]
    re = np.clip((np.log(np.maximum(r.astype(np.float32), 1.0)) - np.log(lo)) / (np.log(hi) - np.log(lo)), 0, 1) * 254.0
    ie = np.where(v, np.rint(ie), 255).astype(np.uint8)
    re = np.where(v, np.rint(re), 255).astype(np.uint8)
    g, ok = _gradient(r, v)
    ge = np.where(ok, 128 + 127 * np.clip(g / enc["grad_scale"], -1, 1), 128)
    ge = np.rint(ge).astype(np.uint8)
    return {
        "III": np.dstack([ie, ie, ie]),
        "RRR": np.dstack([re, re, re]),
        "IRI": np.dstack([ie, re, ie]),
        "IRG": np.dstack([ie, re, ge]),
    }


def _write_one(args: tuple[str, str, dict]) -> None:
    split, name, enc = args
    i, r = read_pair(split, name)
    for layout, img in encode(i, r, enc).items():
        path = OUT / layout / split / "images" / name
        if not cv2.imwrite(str(path), img):
            raise RuntimeError(f"write failed: {path}")


def main() -> None:
    workers = 16
    train_names = sorted(p.name for p in (SRC / "train" / "intensity_filtered" / "images").glob("*.png"))
    enc = fit_encoding(train_names, workers)
    print(json.dumps(enc, indent=2))

    jobs = []
    for layout in LAYOUTS:
        for split in SPLITS:
            (OUT / layout / split / "images").mkdir(parents=True, exist_ok=True)
            labels = OUT / layout / split / "labels"
            if labels.exists():
                shutil.rmtree(labels)
            shutil.copytree(SRC / split / "intensity_filtered" / "labels", labels)
    for split in SPLITS:
        # The two modalities must name the same frames, or the pairing is wrong.
        ni = sorted(p.name for p in (SRC / split / "intensity_filtered" / "images").glob("*.png"))
        nr = sorted(p.name for p in (SRC / split / "range_filtered" / "images").glob("*.png"))
        if ni != nr:
            raise SystemExit(f"{split}: intensity and range frame names differ")
        jobs += [(split, n, enc) for n in ni]
    with ProcessPoolExecutor(workers) as pool:
        list(pool.map(_write_one, jobs, chunksize=16))

    for layout in LAYOUTS:
        yaml = (
            f"# Generated by fusion/make_fused.py - packed layout {layout}\n"
            f"path: {(OUT / layout).as_posix()}\n"
            "train: train/images\nval: valid/images\ntest: test/images\n"
            f"nc: {len(NAMES)}\nnames: {NAMES}\n"
        )
        (OUT / f"data_{layout}.yaml").write_text(yaml, encoding="utf-8")
    (OUT / "encoding.json").write_text(json.dumps(enc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(jobs)} frames x {len(LAYOUTS)} layouts to {OUT}")


if __name__ == "__main__":
    main()
