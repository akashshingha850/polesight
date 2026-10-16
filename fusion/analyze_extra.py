#!/usr/bin/env python3
"""Second-pass analysis of the fusion study: where the gains come from and what caps them.

``analyze.py`` answers "which arm wins". This script asks why, with the same runs, the
same evaluator pieces and the same 109 test frames:

    1. IoU profile       mask AP at each IoU threshold 0.50 ... 0.95 (analyze.py cache)
    2. routes            mask mAP50-95 per survey route (data/metadata.json)
    3. objects           one record per ground-truth instance: native size, size on the
                         scoring grid, best mask IoU per run, found or missed
    4. complementarity   objects found by the I model only, the R model only, both, neither,
                         against the same split for two R (or two I) models of different seeds
    5. scoring grid      Ultralytics scores masks at imgsz/4 (160 x 24 cells at 640 px).
                         Re-score every single, early and mid run at the full letterboxed
                         resolution (GT rasterised with mask_ratio 1, predicted masks
                         upsampled as in ``ops.process_mask_native``) and redo the headline
                         paired bootstraps there
    6. training curves   validation mask mAP per epoch, from each run's results.csv

    python fusion/analyze_extra.py            # writes fusion/results/extra_<model>_<imgsz>.{json,md}
                                              # and fusion/results/figures/*.png
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as an  # noqa: E402
import fuse_eval as fe  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.nn.autobackend import AutoBackend  # noqa: E402
from ultralytics.utils import ops  # noqa: E402
from ultralytics.utils.metrics import SegmentMetrics, box_iou, mask_iou  # noqa: E402
from ultralytics.utils.torch_utils import select_device  # noqa: E402

ROOT = an.ROOT
OUT = an.OUT
FIG = OUT / "figures"
CLASSES = an.CLASSES
NATIVE_WH = (1024, 128)
CONF = 0.25  # operating point for "found": Ultralytics' predict default
SIZE_BINS = [0, 40, 80, 160, np.inf]  # native mask area, px^2
SINGLE_ARMS = ["III", "RRR", "IRI", "IRG", "MIDIR", "MIDRR"]


# ------------------------------------------------------------------ per-object records


class ObjValidator(fe.FusedValidator):
    """FusedValidator that also keeps, per image, the GT-by-prediction mask and box IoU matrices."""

    native = False  # score at the full letterboxed resolution instead of imgsz/4

    def init_metrics(self, model):
        super().init_metrics(model)
        if self.native:
            self.process = ops.process_mask_native
        self.records = []

    def postprocess(self, preds):
        if self.native and len(preds) > 1:
            raise NotImplementedError("late fusion builds masks at proto resolution; native grid is single-model only")
        return super().postprocess(preds)

    def _process_batch(self, preds, batch):
        tp = super()._process_batch(preds, batch)
        g, p = batch["cls"].shape[0], preds["cls"].shape[0]
        miou = np.zeros((g, p), np.float32)
        biou = np.zeros((g, p), np.float32)
        if g and p:
            miou = mask_iou(batch["masks"].flatten(1).float(), preds["masks"].flatten(1).float()).cpu().numpy()
            biou = box_iou(batch["bboxes"], preds["bboxes"]).cpu().numpy()
        gt_box = ops.scale_boxes(batch["imgsz"], batch["bboxes"].clone(), batch["ori_shape"],
                                 ratio_pad=batch["ratio_pad"]).cpu().numpy() if g else np.zeros((0, 4))
        self.records.append({
            "file": Path(batch["im_file"]).name,
            "gt_cls": batch["cls"].cpu().numpy().astype(int),
            "gt_px": batch["masks"].flatten(1).sum(1).cpu().numpy() if g else np.zeros(0),
            "grid": tuple(batch["masks"].shape[-2:]) if g else None,
            "gt_box": gt_box,
            "pred_cls": preds["cls"].cpu().numpy().astype(int),
            "conf": preds["conf"].cpu().numpy(),
            "miou": miou.astype(np.float16),
            "biou": biou.astype(np.float16),
        })
        return tp


def run_objects(members, data_yaml, imgsz=640, native=False, fuse="wbf", device="0", batch=16):
    """As fe.evaluate, plus per-image IoU records. Returns (stats, summary, records)."""
    over = dict(mask_ratio=1) if native else {}
    v = ObjValidator(args=dict(data=str(data_yaml), split="test", imgsz=imgsz, batch=batch, half=False, plots=False,
                               task="segment", rect=True, workers=4, conf=0.001, iou=0.7, max_det=300,
                               device=device, verbose=False, **over))
    v.fuse, v.native = fuse, native
    models = [AutoBackend(str(w), device=select_device(device), fp16=False).eval() for w, _ in members]
    v.device, v.stride = models[0].device, models[0].stride
    v.data = check_det_dataset(str(data_yaml))
    v.dataloader = v.get_dataloader(v.data["test"], batch)
    v.training = False
    v.init_metrics(models[0])
    v.seen = 0
    with torch.no_grad():
        for b in v.dataloader:
            b = v.preprocess(b)
            outs = [m(b["img"] if ch is None else b["img"][:, [ch, ch, ch]]) for m, (_, ch) in zip(models, members)]
            v.update_metrics(v.postprocess(outs), b)
    stats = {k: list(val) for k, val in v.metrics.stats.items()}
    summary = fe.score(stats, v.names)
    summary["files"] = [r["file"] for r in v.records]
    return stats, summary, v.records


def objects_single(model, layout, imgsz, seed, native=False):
    w = an.best_pt(model, layout, imgsz, seed)
    data = an.DATA / an.DATASET_OF.get(layout, layout) / "data.yaml"
    tag = "native" if native else "grid"
    return an.cached(f"obj_{tag}_{model}_{layout}_{imgsz}_s{seed}",
                     lambda: run_objects([(w, None)], data, imgsz, native=native))


def objects_pair(model, imgsz, a, b, fuse, tag):
    members = [(an.best_pt(model, la, imgsz, sa), ch) for la, sa, ch in (a, b)]
    return an.cached(f"obj_grid_{model}_{imgsz}_{tag}_{fuse}",
                     lambda: run_objects(members, an.DATA / "IRI" / "data.yaml", imgsz, fuse=fuse))


# ------------------------------------------------------------------ ground truth geometry


def native_gt(files: list[str], records: list[dict]) -> list[dict]:
    """Native-resolution area and box of every GT in record order, matched to the label polygons by box IoU."""
    W, H = NATIVE_WH
    lab_dir = an.DATA / "IRI" / "test" / "labels"
    out = []
    for f, r in zip(files, records):
        polys = []
        for line in (lab_dir / f"{Path(f).stem}.txt").read_text().split("\n"):
            if line.strip():
                v = line.split()
                xy = np.array(v[1:], float).reshape(-1, 2) * [W, H]
                x, y = xy[:, 0], xy[:, 1]
                area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                polys.append((int(v[0]), area, np.array([x.min(), y.min(), x.max(), y.max()])))
        for c, box in zip(r["gt_cls"], r["gt_box"]):
            cand = [(fe._box_iou_np(box, pb[None])[0], a, pb) for pc, a, pb in polys if pc == c]
            iou, area, pb = max(cand, key=lambda t: t[0])
            out.append({"file": f, "cls": int(c), "area": float(area), "w": float(pb[2] - pb[0]),
                        "h": float(pb[3] - pb[1]), "match_iou": float(iou)})
    return out


def best_iou(records: list[dict], conf: float = CONF) -> np.ndarray:
    """Per GT (record order): best mask IoU with a same-class prediction at confidence >= conf."""
    out = []
    for r in records:
        for gi, c in enumerate(r["gt_cls"]):
            ok = (r["pred_cls"] == c) & (r["conf"] >= conf)
            out.append(float(r["miou"][gi, ok].max()) if ok.any() else 0.0)
    return np.array(out)


def gt_px(records: list[dict]) -> np.ndarray:
    return np.concatenate([r["gt_px"] for r in records])


# ------------------------------------------------------------------ cached-stat analyses


def ap_by_iou(stats: dict) -> tuple[np.ndarray, np.ndarray]:
    """Class-mean mask and box AP at each of the ten IoU thresholds."""
    m = SegmentMetrics(names=an.NAMES)
    m.stats = {k: list(v) for k, v in stats.items()}
    m.process(save_dir=Path("."), plot=False)
    return m.seg.all_ap.mean(0), m.box.all_ap.mean(0)


def route_of() -> dict[str, str]:
    meta = json.loads((ROOT / "data" / "metadata.json").read_text())["metadata"]
    return {v["image"]: v["route"] for v in meta.values() if v["split"] == "test"}


def curves(model: str, layout: str, imgsz: int, seeds: list[int]) -> np.ndarray:
    """Validation mask mAP50-95 per epoch, (seeds, epochs)."""
    rows = []
    for s in seeds:
        with open(an.RUNS / f"{model}_{layout}_{imgsz}_s{s}" / "train" / "results.csv") as fh:
            rows.append([float(r["metrics/mAP50-95(M)"]) for r in csv.DictReader(fh)])
    n = min(map(len, rows))
    return np.array([r[:n] for r in rows]) * 100


# ------------------------------------------------------------------ figures


# Reference palette (dataviz skill, light mode): slots 1-3 and a neutral for controls.
BLUE, ORANGE, AQUA, YELLOW, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"
INK, INK2 = "#0b0b0b", "#52514e"


def _style(ax, title, ylabel=None, xlabel=None):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c3")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color="#e7e6e2", lw=0.8)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)


def figures(res: dict, curv: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    colours = {"RRR": GREY, "III": YELLOW, "IRG": AQUA, "late-wbf I+R": ORANGE, "MIDIR": BLUE}
    label = {"RRR": "R only", "III": "I only", "IRG": "early IRG", "late-wbf I+R": "late WBF I+R", "MIDIR": "mid I+R"}

    # 1. AP by IoU threshold, difference to range only
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=160)
    thr = np.linspace(0.5, 0.95, 10)
    base = np.array(res["iou_profile"]["RRR"]["mask"]).mean(0)
    for k in ("III", "IRG", "late-wbf I+R", "MIDIR"):
        d = np.array(res["iou_profile"][k]["mask"]).mean(0) - base
        ax.plot(thr, d, lw=2, color=colours[k], marker="o", ms=4)
        ax.annotate(label[k], (thr[0], d[0]), xytext=(-6, 0), textcoords="offset points", ha="right", va="center",
                    fontsize=8, color=INK2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlim(0.38, 0.97)
    _style(ax, "Mask AP gain over range only, by IoU threshold", "Δ mask AP (points)", "IoU threshold")
    fig.tight_layout()
    fig.savefig(FIG / "ap_by_iou.png")
    plt.close(fig)

    # 2. recall by native object size
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=160)
    keys = ["III", "RRR", "IRG", "late-wbf I+R", "MIDIR"]
    nb = len(SIZE_BINS) - 1
    wbar = 0.16
    for i, k in enumerate(keys):
        r = np.array(res["objects"]["recall_by_size"][k]["mean"]) * 100
        x = np.arange(nb) + (i - 2) * wbar
        ax.bar(x, r, wbar * 0.88, color=colours[k], label=label[k])
    n = res["objects"]["size_bin_counts"]
    ax.set_xticks(np.arange(nb), [(f"{a:g}–{b:g}" if np.isfinite(b) else f"≥{a:g}") + f"\nn = {c}" for a, b, c in
                                  zip(SIZE_BINS[:-1], SIZE_BINS[1:], n)])
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper left")
    _style(ax, f"Recall (mask IoU ≥ 0.5, conf ≥ {CONF}) by native mask area", "recall (%)", "mask area at native 1024×128 (px²)")
    fig.tight_layout()
    fig.savefig(FIG / "recall_by_size.png")
    plt.close(fig)

    # 3. complementarity
    fig, ax = plt.subplots(figsize=(6.4, 2.6), dpi=160)
    rows = [("I + R", "IR"), ("R + R (ctrl)", "RR"), ("I + I (ctrl)", "II")]
    segs = [("both", "#86b6ef"), ("A only", BLUE), ("B only", ORANGE), ("neither", "#e7e6e2")]
    for yi, (name, key) in enumerate(rows):
        v = res["objects"]["complementarity"][key]
        left = 0
        for sk, col in segs:
            val = v[sk.replace(" ", "_")] * 100
            ax.barh(yi, val, left=left, color=col, edgecolor="white", lw=1.5, height=0.6)
            if val > 4:
                ax.text(left + val / 2, yi, f"{val:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if col in (BLUE, ORANGE) else INK)
            left += val
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    from matplotlib.patches import Patch
    ax.legend([Patch(color=c) for _, c in segs], ["found by both", "first model only (I, R, I)", "second model only (R, R, I)", "neither"],
              frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    _style(ax, "Which test objects two single models find (share of all objects, %)")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIG / "complementarity.png")
    plt.close(fig)

    # 4. training curves
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.2), dpi=160, gridspec_kw={"width_ratios": [2, 1]})
    for k, col in (("III", YELLOW), ("RRR", GREY), ("IRI", "#86b6ef"), ("IRG", AQUA)):
        c = curv[k]
        axs[0].plot(np.arange(1, c.shape[1] + 1), c.mean(0), lw=2, color=col, label=k)
    axs[0].legend(frameon=False, fontsize=8, ncol=4)
    _style(axs[0], "Stage 1: validation mask mAP50-95", "points", "epoch")
    for k, col in (("MIDIR", BLUE), ("MIDRR", GREY)):
        c = curv[k]
        axs[1].plot(np.arange(1, c.shape[1] + 1), c.mean(0), lw=2, color=col, label=k)
    axs[1].legend(frameon=False, fontsize=8)
    _style(axs[1], "Stage 2: fine-tune", None, "epoch")
    fig.tight_layout()
    fig.savefig(FIG / "training_curves.png")
    plt.close(fig)

    # 5. grid vs native scoring
    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=160)
    arms = [a for a in SINGLE_ARMS if a in res["scoring_grid"]["arms"]]
    g = [res["scoring_grid"]["arms"][a]["grid"][0] for a in arms]
    nt = [res["scoring_grid"]["arms"][a]["native"][0] for a in arms]
    x = np.arange(len(arms))
    ax.bar(x - 0.2, g, 0.36, color="#86b6ef", label="default: imgsz/4 grid")
    ax.bar(x + 0.2, nt, 0.36, color=BLUE, label="full letterbox resolution")
    ax.set_xticks(x, arms)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax, "Mask mAP50-95 under two scoring grids (seed mean)", "points")
    fig.tight_layout()
    fig.savefig(FIG / "scoring_grid.png")
    plt.close(fig)


# ------------------------------------------------------------------ main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo11m-seg")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--draws", type=int, default=300)
    a = ap.parse_args()
    m, sz, seeds = a.model, a.imgsz, a.seeds
    n = len(seeds)
    I = lambda s: ("III", s, 0)  # noqa: E731
    R = lambda s: ("RRR", s, 1)  # noqa: E731
    nxt = {s: seeds[(i + 1) % n] for i, s in enumerate(seeds)}
    res: dict = {"model": m, "imgsz": sz, "seeds": seeds, "conf_found": CONF, "size_bins_px2": SIZE_BINS[:-1]}

    # ---- 1, 2: from the analyze.py cache
    arms = {k: [an.single(m, k, sz, s) for s in seeds] for k in SINGLE_ARMS}
    arms["late-wbf I+R"] = [an.pair(m, sz, I(s), R(s), "wbf", f"IR_s{s}") for s in seeds]
    arms["late-wbf R+R (ctrl)"] = [an.pair(m, sz, R(s), R(nxt[s]), "wbf", f"RR_s{s}_s{nxt[s]}") for s in seeds]
    res["iou_profile"] = {}
    for k, runs in arms.items():
        prof = [ap_by_iou(st) for st, _ in runs]
        res["iou_profile"][k] = {"mask": [list(p[0] * 100) for p in prof], "box": [list(p[1] * 100) for p in prof]}

    files = arms["RRR"][0][1]["files"]
    rmap = route_of()
    routes = sorted({rmap[f] for f in files}, key=lambda r: -sum(rmap[f] == r for f in files))
    res["routes"] = {}
    for r in routes:
        ix = [i for i, f in enumerate(files) if rmap[f] == r]
        row = {"frames": len(ix), "instances": int(sum(len(arms["RRR"][0][0]["target_cls"][i]) for i in ix))}
        for k in ("III", "RRR", "IRG", "late-wbf I+R", "MIDIR", "MIDRR"):
            row[k] = an.ms([fe.score(an.resampled(st, ix), an.NAMES)["mask_map"] * 100 for st, _ in arms[k]])
        res["routes"][r] = row

    # ---- 3, 4: per-object records at the default scoring grid
    obj = {k: [objects_single(m, k, sz, s) for s in seeds] for k in SINGLE_ARMS}
    obj["late-wbf I+R"] = [objects_pair(m, sz, I(s), R(s), "wbf", f"IR_s{s}") for s in seeds]
    for k in SINGLE_ARMS:  # the hooked validator must reproduce the cached scores exactly
        for (st, sm, _), (_, ref) in zip(obj[k], arms[k]):
            assert abs(sm["mask_map"] - ref["mask_map"]) < 1e-9, (k, sm["mask_map"], ref["mask_map"])
            assert sm["files"] == files
    recs0 = obj["RRR"][0][2]
    gt = native_gt(files, recs0)
    area = np.array([g["area"] for g in gt])
    cls = np.array([g["cls"] for g in gt])
    px = gt_px(recs0)
    grid = next(r["grid"] for r in recs0 if r["grid"])
    bins = np.digitize(area, SIZE_BINS[1:-1])
    res["objects"] = {
        "n_gt": int(len(gt)),
        "gt_match_iou_min": float(min(g["match_iou"] for g in gt)),
        "native_area_px2": {"median": float(np.median(area)), "p10": float(np.percentile(area, 10)),
                            "p90": float(np.percentile(area, 90))},
        "native_width_px": {"median": float(np.median([g["w"] for g in gt]))},
        "scoring_grid_hw": list(grid),
        "gt_cells_on_grid": {"median": float(np.median(px)), "zero": int((px == 0).sum()),
                             "le_3": int((px <= 3).sum()), "le_10": int((px <= 10).sum())},
        "size_bin_counts": [int((bins == b).sum()) for b in range(len(SIZE_BINS) - 1)],
        "recall_by_size": {}, "recall_by_class": {}, "best_iou_found": {},
    }
    found = {k: np.array([best_iou(rec) >= 0.5 for _, _, rec in obj[k]]) for k in obj}  # (seeds, n_gt)
    biou = {k: np.array([best_iou(rec) for _, _, rec in obj[k]]) for k in obj}
    for k in obj:
        f = found[k]
        res["objects"]["recall_by_size"][k] = {
            "mean": [float(f[:, bins == b].mean()) for b in range(len(SIZE_BINS) - 1)],
            "std": [float(f[:, bins == b].mean(1).std(ddof=1)) for b in range(len(SIZE_BINS) - 1)]}
        res["objects"]["recall_by_class"][k] = [float(f[:, cls == c].mean()) for c in range(4)]
        q = biou[k]
        res["objects"]["best_iou_found"][k] = float(q[q >= 0.5].mean())
        res["objects"].setdefault("recall", {})[k] = float(f.mean())
    # zero-cell objects can never be matched; how many of them does any run find?
    res["objects"]["zero_cell_found_any"] = int(np.any(np.concatenate([found[k] for k in obj]), 0)[px == 0].sum())

    def split(fa: np.ndarray, fb: np.ndarray) -> dict:
        return {"both": float((fa & fb).mean()), "A_only": float((fa & ~fb).mean()),
                "B_only": float((~fa & fb).mean()), "neither": float((~fa & ~fb).mean()),
                "oracle_recall": float((fa | fb).mean())}

    comp = {}
    for key, A, B in (("IR", "III", "RRR"), ("RR", "RRR", "RRR"), ("II", "III", "III")):
        rows = []
        for i, s in enumerate(seeds):
            j = i if A != B else seeds.index(nxt[s])
            rows.append(split(found[A][i], found[B][j]))
        comp[key] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        if key == "IR":
            comp[key]["per_class"] = {c: split(found[A][:, cls == ci].ravel(), found[B][:, cls == ci].ravel())
                                      for ci, c in enumerate(CLASSES)}
    # of the I-only objects (I finds, R misses), how many do the fused arms recover?
    rec_I_only = {}
    for k in ("IRI", "IRG", "late-wbf I+R", "MIDIR", "MIDRR"):
        hits, tot = 0, 0
        for i in range(n):
            sel = found["III"][i] & ~found["RRR"][i]
            hits += int(found[k][i][sel].sum())
            tot += int(sel.sum())
        rec_I_only[k] = hits / max(tot, 1)
    comp["fused_recover_I_only"] = rec_I_only
    res["objects"]["complementarity"] = comp

    # ---- 5: re-score at full letterbox resolution, with paired bootstraps
    nat = {k: [objects_single(m, k, sz, s, native=True) for s in seeds] for k in SINGLE_ARMS}
    idxs = an.boot_idx(len(files), a.draws)
    sg = {"arms": {}, "comparisons": []}
    for k in SINGLE_ARMS:
        sg["arms"][k] = {"grid": an.ms([r[1]["mask_map"] * 100 for r in arms[k]]),
                         "native": an.ms([r[1]["mask_map"] * 100 for r in nat[k]]),
                         "native_mask50": an.ms([r[1]["mask_map50"] * 100 for r in nat[k]]),
                         "native_per_class": [an.ms(np.array([r[1]["mask_per_class"][c] for r in nat[k]]) * 100)
                                              for c in range(4)]}
    nruns = {k: [(st, sm) for st, sm, _ in nat[k]] for k in nat}
    boot = {k: an.boot_scores(nruns[k], idxs) for k in nruns}
    point = {k: np.array([[sm["mask_map"], *sm["mask_per_class"], sm["box_map"]] for _, sm in nruns[k]]) for k in nruns}
    for x, y in (("MIDIR", "RRR"), ("MIDIR", "MIDRR"), ("IRG", "RRR"), ("IRI", "RRR"), ("RRR", "III"),
                 ("MIDRR", "RRR"), ("MIDIR", "IRG")):
        sg["comparisons"].append({"a": x, "b": y, **an.delta_ci(boot[x], boot[y], 0, point[x], point[y]),
                                  "fence_pole": an.delta_ci(boot[x], boot[y], 1, point[x], point[y])})
    sg["native_grid_hw"] = list(next(r["grid"] for r in nat["RRR"][0][2] if r["grid"]))
    res["scoring_grid"] = sg

    # ---- 6: training curves
    curv = {k: curves(m, k, sz, seeds) for k in SINGLE_ARMS}
    res["training"] = {k: {"epochs": int(c.shape[1]), "best_epoch_mean": float(np.mean(c.argmax(1) + 1)),
                           "max": an.ms(c.max(1)), "last10": an.ms(c[:, -10:].mean(1)),
                           "first_epoch": an.ms(c[:, 0])} for k, c in curv.items()}

    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{m}_{sz}"
    (OUT / f"extra_{tag}.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    figures(res, curv)
    print(json.dumps({k: res[k] for k in ("objects", "scoring_grid", "training")}, indent=1, default=float)[:6000])
    print(f"wrote {OUT}/extra_{tag}.json and {FIG}/")


if __name__ == "__main__":
    main()
