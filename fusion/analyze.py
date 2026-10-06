#!/usr/bin/env python3
"""Score every fusion arm, run the controls, and write the findings.

Reads the trained runs in ``fusion/runs/``, scores them (and the late-fusion
combinations built from them) through ``fuse_eval.py`` on the test split, and writes

    fusion/results/results.json   every number below, per run and per arm
    fusion/results/results.md     the tables and the verdicts

Arms (all ``yolo11m-seg``, mask mAP50-95 on the 109 test frames, seeds 0-2):

    single-modality controls   III, RRR            (same 8-bit encoding as the fused rows)
    early fusion               IRI, IRG            (fusion.md Approach A / fusion_sota.md E1, E2)
    late fusion                I+R   WBF / NMS     (Approach D / Tier 2), seed-matched pairs
    ensemble controls          R+R, I+I            different-seed pairs of the same modality
    class-aware late fusion    I+R   WBF, per-class weights tuned on the validation split (Approach E)

Statistics (fusion_sota.md section 4): mean and standard deviation over seeds, and for each
headline comparison a paired bootstrap over test images. One bootstrap draw resamples the
109 frames once and recomputes mAP for every seed of both arms on that sample; the
statistic is the seed-mean difference, so the interval covers test-set sampling but not
training variance, which the seed spread reports.

    python fusion/analyze.py                      # uses fusion/runs, imgsz from --imgsz
    python fusion/analyze.py --imgsz 1024
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fuse_eval as fe  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "fusion" / "runs"
CACHE = ROOT / "fusion" / "cache"
OUT = ROOT / "fusion" / "results"
DATA = ROOT / "fusion" / "data"
NAMES = {0: "fence_pole", 1: "gantry_sign_pole", 2: "light_pole", 3: "traffic_pole"}
CLASSES = list(NAMES.values())


# ------------------------------------------------------------------------- scoring


def best_pt(model: str, layout: str, imgsz: int, seed: int) -> Path:
    return RUNS / f"{model}_{layout}_{imgsz}_s{seed}" / "train" / "weights" / "best.pt"


def cached(key: str, fn):
    """Per-image stats are expensive to recompute and the runs are immutable; cache on disk."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{key}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    value = fn()
    path.write_bytes(pickle.dumps(value))
    return value


DATASET_OF = {"MIDIR": "IRI", "MIDRR": "RRR"}  # two-stream runs read the packed set they were trained on


def single(model, layout, imgsz, seed):
    w = best_pt(model, layout, imgsz, seed)
    data = DATA / f"data_{DATASET_OF.get(layout, layout)}.yaml"
    return cached(f"{model}_{layout}_{imgsz}_s{seed}", lambda: fe.evaluate([(w, None)], data, imgsz))


def pair(model, imgsz, a, b, fuse, tag, class_w=None, split="test"):
    """Late fusion of two checkpoints, each fed its own channel of the packed IRI set.

    a, b are (layout, seed, channel). Layout III reads channel 0 (I), RRR channel 1 (R).
    """
    members = [(best_pt(model, la, imgsz, sa), ch) for la, sa, ch in (a, b)]
    return cached(f"{model}_{imgsz}_{tag}_{fuse}_{split}", lambda: fe.evaluate(
        members, DATA / "data_IRI.yaml", imgsz, split, fuse=fuse, class_w=class_w))


def tune_class_weights(model, imgsz, a, b, tag):
    """Per-class modality weight alpha_c in {0, .25, .5, .75, 1}, chosen on the *validation* split.

    Fusion clusters per class, so classes are independent and each is tuned on its own
    mask AP50-95. The two models' scores are scaled by 2*alpha_c and 2*(1-alpha_c).
    """
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    best = np.zeros(4)
    best_ap = np.full(4, -1.0)
    for alpha in grid:
        cw = np.array([[2 * alpha] * 4, [2 * (1 - alpha)] * 4])
        _, s = pair(model, imgsz, a, b, "wbf", f"{tag}_a{alpha}", class_w=cw, split="val")
        for c in range(4):
            if s["mask_per_class"][c] > best_ap[c] + 1e-12:
                best_ap[c], best[c] = s["mask_per_class"][c], alpha
    return best


# ---------------------------------------------------------------------- statistics


def boot_idx(n: int, draws: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(draws, n))


def resampled(stats: dict, idx: np.ndarray) -> dict:
    return {k: [v[i] for i in idx] for k, v in stats.items()}


def boot_scores(runs: list, idxs: np.ndarray) -> np.ndarray:
    """Mask mAP50-95 (col 0), its four per-class values (cols 1-4) and box mAP50-95 (col 5), per seed and draw."""
    out = np.zeros((len(runs), len(idxs), 6))
    for si, (stats, _) in enumerate(runs):
        for di, ix in enumerate(idxs):
            s = fe.score(resampled(stats, ix), NAMES)
            out[si, di] = [s["mask_map"], *s["mask_per_class"], s["box_map"]]
    return out


def delta_ci(a: np.ndarray, b: np.ndarray, col: int, point_a: np.ndarray, point_b: np.ndarray) -> dict:
    """Seed-mean difference with a paired-bootstrap interval. a, b: (seeds, draws, 6); points: (seeds, 6)."""
    d = (a[:, :, col] - b[:, :, col]).mean(0) * 100
    return {"delta": float((point_a[:, col] - point_b[:, col]).mean() * 100),
            "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)), "p_gt0": float((d > 0).mean())}


def ms(values) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


# ---------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo11m-seg")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--layouts", nargs="+", default=["III", "RRR", "IRI", "IRG"])
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--no-late", action="store_true", help="skip decision-level fusion")
    a = ap.parse_args()
    m, sz, seeds = a.model, a.imgsz, a.seeds
    seeds = [s for s in seeds if all(best_pt(m, lay, sz, s).exists() and
                                     (RUNS / f"{m}_{lay}_{sz}_s{s}" / "eval.json").exists() for lay in a.layouts)]
    print(f"complete seeds: {seeds}")
    if not seeds:
        raise SystemExit("no seed has every layout finished")

    arms: dict[str, list] = {}  # arm -> [(stats, summary) per seed]
    for lay in a.layouts:
        arms[lay] = [single(m, lay, sz, s) for s in seeds]
    files = arms[a.layouts[0]][0][1]["files"]
    for lay in a.layouts:
        for r in arms[lay]:
            assert r[1]["files"] == files, "frame order differs between runs"

    for mid in ("MIDIR", "MIDRR"):
        if all((RUNS / f"{m}_{mid}_{sz}_s{s}" / "eval.json").exists() for s in seeds):
            arms[mid] = [single(m, mid, sz, s) for s in seeds]
            assert all(r[1]["files"] == files for r in arms[mid]), "frame order differs"
    if not a.no_late and "III" in arms and "RRR" in arms and len(seeds) >= 2:
        n = len(seeds)
        I = lambda s: ("III", s, 0)  # noqa: E731
        R = lambda s: ("RRR", s, 1)  # noqa: E731
        for fuse in ("wbf", "nms"):
            arms[f"late-{fuse} I+R"] = [pair(m, sz, I(s), R(s), fuse, f"IR_s{s}") for s in seeds]
            arms[f"late-{fuse} R+R (ctrl)"] = [
                pair(m, sz, R(s), R(seeds[(i + 1) % n]), fuse, f"RR_s{s}_s{seeds[(i + 1) % n]}") for i, s in enumerate(seeds)]
            arms[f"late-{fuse} I+I (ctrl)"] = [
                pair(m, sz, I(s), I(seeds[(i + 1) % n]), fuse, f"II_s{s}_s{seeds[(i + 1) % n]}") for i, s in enumerate(seeds)]
        alphas, pc = [], []
        for s in seeds:
            al = tune_class_weights(m, sz, I(s), R(s), f"IR_s{s}")
            alphas.append(al.tolist())
            cw = np.array([2 * al, 2 * (1 - al)])
            pc.append(pair(m, sz, I(s), R(s), "wbf", f"IR_s{s}_pc", class_w=cw))
        arms["late-wbf I+R class-aware (val-tuned)"] = pc

    n_img = len(files)
    idxs = boot_idx(n_img, a.draws)

    # ------------------------------------------------------------------ per-arm table
    rows = {}
    for name, runs in arms.items():
        mk = [r[1]["mask_map"] * 100 for r in runs]
        bx = [r[1]["box_map"] * 100 for r in runs]
        m50 = [r[1]["mask_map50"] * 100 for r in runs]
        pcm = np.array([r[1]["mask_per_class"] for r in runs]) * 100
        rows[name] = {
            "mask_mAP50-95": ms(mk), "box_mAP50-95": ms(bx), "mask_mAP50": ms(m50),
            "per_seed_mask": mk,
            "per_class_mask": {c: ms(pcm[:, i]) for i, c in enumerate(CLASSES)},
        }

    point = {k: np.array([[r[1]["mask_map"], *r[1]["mask_per_class"], r[1]["box_map"]] for r in runs]) for k, runs in arms.items()}
    boot = {k: boot_scores(runs, idxs) for k, runs in arms.items()}
    comparisons, perclass = [], []
    refs = [k for k in ("RRR", "III") if k in arms]
    pairs = [(x, y) for x in arms for y in refs if x != y and {x, y} != {"III", "RRR"}]
    pairs.append(("RRR", "III")) if {"RRR", "III"} <= set(arms) else None
    # Each fused arm against its own control: the comparison that isolates the modality.
    for x, y in (("late-wbf I+R", "late-wbf R+R (ctrl)"), ("late-nms I+R", "late-nms R+R (ctrl)"),
                 ("MIDIR", "MIDRR"), ("late-wbf I+R", "IRG"), ("MIDIR", "late-wbf I+R"), ("MIDIR", "IRG")):
        if x in arms and y in arms:
            pairs.append((x, y))
    for x, y in pairs:
        comparisons.append({"a": x, "b": y, **delta_ci(boot[x], boot[y], 0, point[x], point[y]),
                            "box": delta_ci(boot[x], boot[y], 5, point[x], point[y])})
    for x in [k for k in ("IRI", "IRG", "late-wbf I+R", "late-nms I+R", "MIDIR", "MIDRR", "RRR") if k in arms]:
        for y in [k for k in ("RRR", "III") if k in arms and k != x]:
            for ci, c in enumerate(CLASSES):
                perclass.append({"a": x, "b": y, "class": c, **delta_ci(boot[x], boot[y], ci + 1, point[x], point[y])})

    result = {"model": m, "imgsz": sz, "seeds": seeds, "n_test_images": n_img, "arms": rows,
              "comparisons": comparisons, "per_class_comparisons": perclass,
              "class_alphas_I_weight_per_seed": alphas if not a.no_late and "III" in arms else None,
              "bootstrap_draws": a.draws}
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{m}_{sz}"
    (OUT / f"results_{tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    write_md(result, OUT / f"results_{tag}.md")
    print(f"wrote {OUT}/results_{tag}.{{json,md}}")


def write_md(r: dict, path: Path) -> None:
    L = [f"# Fusion results: {r['model']} @ {r['imgsz']} px",
         "",
         f"Test split, {r['n_test_images']} frames, seeds {r['seeds']}. Mask and box mAP50-95 in points "
         "(mean ± std over seeds). Every row is scored through one evaluator, `fusion/fuse_eval.py`.",
         "", "## Arms", "",
         "| Arm | Mask mAP50-95 | Mask mAP50 | Box mAP50-95 | per-seed mask |", "|---|---:|---:|---:|---|"]
    for name, v in r["arms"].items():
        f = lambda t: f"{t[0]:.2f} ± {t[1]:.2f}"  # noqa: E731
        L.append(f"| {name} | {f(v['mask_mAP50-95'])} | {f(v['mask_mAP50'])} | {f(v['box_mAP50-95'])} | "
                 + ", ".join(f"{x:.2f}" for x in v["per_seed_mask"]) + " |")
    L += ["", "## Per class, mask mAP50-95", "",
          "| Arm | " + " | ".join(CLASSES) + " |", "|---|" + "---:|" * len(CLASSES)]
    for name, v in r["arms"].items():
        L.append(f"| {name} | " + " | ".join(f"{v['per_class_mask'][c][0]:.2f} ± {v['per_class_mask'][c][1]:.2f}"
                                             for c in CLASSES) + " |")
    L += ["", "## Paired comparisons (seed-mean Δ in mask mAP50-95, 95% bootstrap CI over test frames)", "",
          "| A − B | mask Δ | 95% CI | P(Δ>0) | box Δ | box 95% CI |", "|---|---:|---|---:|---:|---|"]
    for c in r["comparisons"]:
        b = c["box"]
        L.append(f"| {c['a']} − {c['b']} | {c['delta']:+.2f} | [{c['lo']:+.2f}, {c['hi']:+.2f}] | {c['p_gt0']:.3f} | "
                 f"{b['delta']:+.2f} | [{b['lo']:+.2f}, {b['hi']:+.2f}] |")
    L += ["", "## Per-class paired deltas", "",
          "| A − B | class | Δ | 95% CI |", "|---|---|---:|---|"]
    for c in r["per_class_comparisons"]:
        L.append(f"| {c['a']} − {c['b']} | {c['class']} | {c['delta']:+.2f} | [{c['lo']:+.2f}, {c['hi']:+.2f}] |")
    if r.get("class_alphas_I_weight_per_seed"):
        L += ["", "Class-aware weights (weight on the intensity model, 0 = range only, 1 = intensity only; "
              "order fence, gantry, light, traffic), per seed, tuned on validation: "
              + "; ".join(str(x) for x in r["class_alphas_I_weight_per_seed"])]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
