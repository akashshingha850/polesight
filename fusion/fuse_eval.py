#!/usr/bin/env python3
"""Per-image evaluation of single models and decision-level (late) fusion.

Late fusion (fusion.md, Approach D; fusion_sota.md, Tier 2) combines the predictions
of an intensity model and a range model at test time. Weighted Boxes Fusion handles
boxes but has no mask counterpart, so this module adds the missing step: after
clustering, the mask of a fused instance is the confidence-weighted mean of its
members' mask probabilities, thresholded at 0.5.

Everything is scored through Ultralytics' own ``SegmentationValidator`` pieces
(dataloader, NMS, mask IoU, TP matching), so that

  * a single model scores the same here as under ``model.val`` (checked by
    ``--check``), and
  * single, early-fused and late-fused rows are all scored by one code path,
  * the per-image TP matrices are kept, which is what a paired bootstrap over test
    images needs (``analyze.py``).

Fusion methods for ``--members`` with two or more checkpoints:

    wbf    weighted-boxes-fusion clustering (IoU > 0.55, same class, at most one detection
           per model in a cluster, so fusing a model with itself returns it), fused score =
           mean member score * min(#members, #models) / #models, fused box =
           score-weighted mean, fused mask = score-weighted mean of mask probabilities
    nms    pool all detections and run class-wise NMS (IoU 0.7): the plain ensemble

Usage (as a library, see ``analyze.py``; as a script for a quick look):

    python fusion/fuse_eval.py --check fusion/runs/yolo11m-seg_RRR_640_s0
    python fusion/fuse_eval.py --members fusion/runs/yolo11m-seg_III_640_s0:0 \
                                         fusion/runs/yolo11m-seg_RRR_640_s0:1 --fuse wbf
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "disabled")

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import ops
from ultralytics.utils.metrics import SegmentMetrics
from ultralytics.utils.torch_utils import select_device

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dual_stream  # noqa: E402,F401  (checkpoints of the two-stream model pickle this module by name)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "fusion" / "data"


# --------------------------------------------------------------------------- fusion


def _mask_probs(proto: torch.Tensor, det: dict, imgsz: tuple[int, int]) -> torch.Tensor:
    """Cropped mask probabilities at proto resolution: sigmoid(coef @ proto), 0 outside the box."""
    c, mh, mw = proto.shape
    n = det["bboxes"].shape[0]
    if n == 0:
        return torch.zeros((0, mh, mw), device=proto.device)
    logits = (det["extra"] @ proto.float().view(c, -1)).view(n, mh, mw)
    ratios = torch.tensor([[mw / imgsz[1], mh / imgsz[0]] * 2], device=proto.device)
    logits = ops.crop_mask(logits, det["bboxes"] * ratios)  # zero outside the box
    inside = ops.crop_mask(torch.ones_like(logits), det["bboxes"] * ratios)
    return torch.sigmoid(logits) * inside


def _box_iou_np(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a + b - inter + 1e-9)


def wbf_fuse(members: list[dict], probs: list[torch.Tensor], thr: float = 0.55, max_det: int = 300,
             class_w: np.ndarray | None = None) -> dict:
    """Cluster the pooled detections of all models and fuse boxes, scores and mask probabilities.

    ``class_w`` is an optional (n_models, n_classes) multiplier on each model's scores
    before clustering (the class-aware variant, fusion.md Approach E).
    """
    n_models = len(members)
    dev = probs[0].device
    boxes = torch.cat([m["bboxes"] for m in members]).cpu().numpy()
    conf = torch.cat([m["conf"] for m in members]).cpu().numpy().astype(np.float64)
    cls = torch.cat([m["cls"] for m in members]).cpu().numpy().astype(int)
    mid = np.concatenate([np.full(len(m["conf"]), k) for k, m in enumerate(members)])
    if class_w is not None and len(conf):
        conf = conf * class_w[mid, cls]
    P = torch.cat(probs)  # (N, mh, mw)
    mh, mw = P.shape[1:]
    if len(conf) == 0:
        return {"bboxes": torch.zeros((0, 4), device=dev), "conf": torch.zeros(0, device=dev),
                "cls": torch.zeros(0, device=dev), "masks": torch.zeros((0, mh, mw), dtype=torch.uint8, device=dev)}

    order = np.argsort(-conf)
    fused_box: list[np.ndarray] = []
    clusters: list[list[int]] = []
    cl_cls: list[int] = []
    cl_models: list[set] = []  # at most one detection per model per cluster
    for idx in order:
        best, best_iou = -1, thr
        if clusters:
            same = [j for j, c in enumerate(cl_cls) if c == cls[idx] and mid[idx] not in cl_models[j]]
            if same:
                ious = _box_iou_np(boxes[idx], np.stack([fused_box[j] for j in same]))
                k = int(np.argmax(ious))
                if ious[k] > best_iou:
                    best = same[k]
        if best < 0:
            clusters.append([idx])
            fused_box.append(boxes[idx].copy())
            cl_cls.append(int(cls[idx]))
            cl_models.append({int(mid[idx])})
        else:
            clusters[best].append(idx)
            cl_models[best].add(int(mid[idx]))
            w = conf[clusters[best]]
            fused_box[best] = (boxes[clusters[best]] * w[:, None]).sum(0) / w.sum()

    C = len(clusters)
    W = torch.zeros((C, len(conf)), device=dev)
    out_conf = np.zeros(C)
    for j, members_idx in enumerate(clusters):
        w = conf[members_idx]
        W[j, members_idx] = torch.as_tensor(w, dtype=torch.float32, device=dev)
        out_conf[j] = w.mean() * min(len(members_idx), n_models) / n_models
    masks = (W @ P.view(len(conf), -1)) / W.sum(1, keepdim=True).clamp_min(1e-9)
    masks = (masks.view(C, mh, mw) > 0.5).to(torch.uint8)

    keep = np.argsort(-out_conf)[:max_det]
    return {
        "bboxes": torch.as_tensor(np.stack(fused_box)[keep], dtype=torch.float32, device=dev),
        "conf": torch.as_tensor(out_conf[keep], dtype=torch.float32, device=dev),
        "cls": torch.as_tensor(np.array(cl_cls)[keep], dtype=torch.float32, device=dev),
        "masks": masks[torch.as_tensor(keep, device=dev)],
    }


def nms_fuse(members: list[dict], masks: list[torch.Tensor], iou: float = 0.7, max_det: int = 300) -> dict:
    """Pool detections from all models and keep the class-wise NMS survivors."""
    boxes = torch.cat([m["bboxes"] for m in members])
    conf = torch.cat([m["conf"] for m in members])
    cls = torch.cat([m["cls"] for m in members])
    M = torch.cat(masks)
    keep = torchvision.ops.batched_nms(boxes, conf, cls, iou)[:max_det]
    return {"bboxes": boxes[keep], "conf": conf[keep], "cls": cls[keep], "masks": M[keep]}


# ------------------------------------------------------------------------- validator


class FusedValidator(SegmentationValidator):
    """SegmentationValidator whose ``postprocess`` takes one raw output per member model."""

    fuse = "wbf"
    class_w: np.ndarray | None = None

    def postprocess(self, preds):  # preds: list of AutoBackend outputs, one per member
        if len(preds) == 1:
            return SegmentationValidator.postprocess(self, preds[0])
        per_model = []
        for out in preds:
            # Same unpacking as SegmentationValidator.postprocess.
            proto = out[0][1] if isinstance(out[0], tuple) else out[1]
            dets = DetectionValidator.postprocess(self, out[0])
            imgsz = [4 * x for x in proto.shape[2:]]
            per_model.append((dets, proto, imgsz))
        results = []
        for i in range(len(per_model[0][0])):
            members = [pm[0][i] for pm in per_model]
            probs = [_mask_probs(pm[1][i], pm[0][i], pm[2]) for pm in per_model]
            if self.fuse == "wbf":
                results.append(wbf_fuse(members, probs, class_w=self.class_w))
            elif self.fuse == "nms":
                results.append(nms_fuse(members, [(p > 0.5).to(torch.uint8) for p in probs]))
            else:
                raise ValueError(self.fuse)
        return results


def evaluate(members: list[tuple[str, int | None]], data_yaml: str | Path, imgsz: int = 640,
             split: str = "test", fuse: str = "wbf", class_w: np.ndarray | None = None,
             device: str = "0", batch: int = 16) -> tuple[dict, dict]:
    """Score one or more checkpoints on ``split``.

    ``members`` is a list of ``(weights, channel)``. ``channel`` picks one tensor channel of
    the dataset image and replicates it to three (so a model trained on layout ``III`` or
    ``RRR`` can be fed from the packed ``IRI`` dataset); ``None`` feeds the image as is.
    Returns (per-image stats dict of lists, summary incl. the frame order).
    """
    v = FusedValidator(args=dict(data=str(data_yaml), split=split, imgsz=imgsz, batch=batch, half=False,
                                 plots=False, task="segment", rect=True, workers=4, conf=0.001, iou=0.7,
                                 max_det=300, device=device, verbose=False))
    v.fuse, v.class_w = fuse, class_w
    models = []
    for w, _ in members:
        m = AutoBackend(str(w), device=select_device(device), fp16=False)
        m.eval()
        models.append(m)
    v.device = models[0].device
    v.args.half = False
    v.stride = models[0].stride
    v.data = check_det_dataset(str(data_yaml))
    v.dataloader = v.get_dataloader(v.data[split], batch)
    v.training = False
    v.init_metrics(models[0])
    v.seen = 0
    files: list[str] = []
    with torch.no_grad():
        for b in v.dataloader:
            files += [Path(f).name for f in b["im_file"]]
            b = v.preprocess(b)
            outs = []
            for m, (_, ch) in zip(models, members):
                x = b["img"] if ch is None else b["img"][:, [ch, ch, ch]]
                outs.append(m(x))
            v.update_metrics(v.postprocess(outs), b)
    stats = {k: list(val) for k, val in v.metrics.stats.items()}
    summary = score(stats, v.names)
    summary["files"] = files  # per-image stats follow this order; analyze.py checks it is shared
    return stats, summary


def score(stats: dict, names: dict) -> dict:
    """mAP summary from (a subset of) per-image stats, via Ultralytics' own metric code."""
    m = SegmentMetrics(names=names)
    m.stats = {k: list(v) for k, v in stats.items()}
    m.process(save_dir=Path("."), plot=False)
    return {
        "mask_map": float(m.seg.map), "mask_map50": float(m.seg.map50),
        "box_map": float(m.box.map), "box_map50": float(m.box.map50),
        "mask_per_class": [float(x) for x in m.seg.maps], "box_per_class": [float(x) for x in m.box.maps],
    }


# ----------------------------------------------------------------------------- CLI


def _parse_member(s: str) -> tuple[str, int | None]:
    path, _, ch = s.partition(":")
    p = Path(path)
    if p.is_dir():
        p = p / "train" / "weights" / "best.pt"
    return str(p), (int(ch) if ch else None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", metavar="RUN", help="score a run here and compare with its eval.json")
    ap.add_argument("--members", nargs="+", help="run_dir[:channel] ... (channel picks I=0 / R=1 of the IRI set)")
    ap.add_argument("--fuse", default="wbf", choices=["wbf", "nms"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    if a.check:
        import json
        run = Path(a.check)
        rec = json.loads((run / "eval.json").read_text())
        _, s = evaluate([_parse_member(str(run))], DATA / f"data_{rec['layout']}.yaml", rec["imgsz"], a.split)
        print(f"here : mask {s['mask_map'] * 100:.3f}  box {s['box_map'] * 100:.3f}")
        print(f"val(): mask {rec['mask']['mAP50-95'] * 100:.3f}  box {rec['box']['mAP50-95'] * 100:.3f}")
    if a.members:
        mem = [_parse_member(m) for m in a.members]
        _, s = evaluate(mem, DATA / "data_IRI.yaml", a.imgsz, a.split, fuse=a.fuse)
        print(f"{a.fuse}: mask {s['mask_map'] * 100:.3f}  box {s['box_map'] * 100:.3f}")


if __name__ == "__main__":
    main()
