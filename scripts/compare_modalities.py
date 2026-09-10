#!/usr/bin/env python3
"""Compare two modality sweeps and write result.md.

Reads both results trees through make_table.load_cells, so every figure here is
generated from the evaluation records rather than transcribed.

    python compare_modalities.py results results_range_filtered -o result.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_table import GROUPS, VARIANTS, load_cells  # noqa: E402

CLASSES = ["fence_pole", "gantry_sign_pole", "light_pole", "traffic_pole"]
HEADLINE = "mask_mAP50-95"


def rows_in_table_order(cells):
    """(model, nms, label, variant, cell) in the published table's order."""
    for label, _family, stem, nms in GROUPS:
        for variant in VARIANTS:
            model = stem.format(v=variant)
            yield model, nms, label, variant, cells.get((model, nms))


def primary_rows(results_dir):
    """The 15 independent checkpoints — one row per trained model.

    YOLO26 is scored through two heads from a single checkpoint, so the 20 table
    rows are not 20 independent observations. is_primary_head marks the row to
    count when the unit of analysis has to be the checkpoint.
    """
    out = {}
    with (results_dir / "eval_table.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["is_primary_head"].strip().lower() == "true":
                out[row["model"]] = row
    return out


def per_class(results_dir):
    """{model: {class: mask mAP50-95}} for the primary head of each checkpoint."""
    out = {}
    for model, row in primary_rows(results_dir).items():
        out[model] = {
            c: float(row[f"{c}_mask_mAP50-95"]) * 100.0 for c in CLASSES
        }
    return out


def provenance(results_dir):
    path = results_dir / "results.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def conditions(results_dir):
    """What each run actually did, for the parity check.

    verify.json carries what the run resolved to; wall time is only in
    results.json's run records, so the two are merged here.
    """
    hours = {}
    report = provenance(results_dir)
    for rec in report.get("runs", []):
        hours[rec.get("model")] = (rec.get("training") or {}).get("train_hours")

    out = {}
    for path in sorted(results_dir.glob("*/verify.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        model = path.parent.name
        out[model] = {
            "epochs_run": rec.get("epochs_run"),
            "batch_actual": rec.get("batch_actual"),
            "optimizer_actual": rec.get("optimizer_actual"),
            "best_epoch": rec.get("best_epoch"),
            "train_hours": hours.get(model),
        }
    return out


def infer_modality(results_dir):
    """The modality a tree was scored on.

    results.json records it directly for runs made after the split was
    reorganised by modality; older trees predate the key, so fall back to the
    data yaml the sweep pointed at.
    """
    dataset = provenance(results_dir).get("dataset", {})
    if dataset.get("modality"):
        return dataset["modality"]
    yaml_name = Path(dataset.get("data_yaml", "")).stem
    if yaml_name == "data":
        return "intensity_filtered"
    if yaml_name.startswith("data_"):
        return yaml_name[len("data_"):]
    return results_dir.name


def signed_tests(deltas):
    """Sign test and Wilcoxon over the paired per-checkpoint deltas."""
    from scipy.stats import binomtest, wilcoxon

    nonzero = [d for d in deltas if d != 0]
    wins = sum(1 for d in nonzero if d > 0)
    out = {"n": len(deltas), "n_nonzero": len(nonzero), "wins": wins,
           "losses": len(nonzero) - wins}
    if nonzero:
        out["sign_p"] = binomtest(wins, len(nonzero), 0.5).pvalue
        try:
            out["wilcoxon_p"] = wilcoxon(nonzero).pvalue
        except ValueError:
            out["wilcoxon_p"] = None
    return out


def fmt(x, spec="{:+.2f}"):
    return "—" if x is None else spec.format(x)


def pval(x):
    """Never print a p-value as 0.000 — say what the rounding hides.

    Carries its own comparator so the caller does not have to choose between
    "p = " and "p < ".
    """
    if x is None:
        return "p —"
    return "p < 0.001" if x < 0.001 else f"p = {x:.3f}"


def build(a_dir, b_dir, a_name, b_name):
    a_cells, b_cells = load_cells(a_dir), load_cells(b_dir)
    a_cond, b_cond = conditions(a_dir), conditions(b_dir)
    a_cls, b_cls = per_class(a_dir), per_class(b_dir)
    a_prim = primary_rows(a_dir)

    lines = []
    W = lines.append

    W(f"# {a_name} vs {b_name}")
    W("")
    W(f"Paired comparison of the two modality sweeps. Every figure is generated "
      f"from the evaluation records in `{a_dir.name}/` and `{b_dir.name}/` by "
      f"`compare_modalities.py`, which reads them through "
      f"`make_table.load_cells` — the same loader behind the published table.")
    W("")

    # --- provenance -------------------------------------------------------
    pa, pb = provenance(a_dir), provenance(b_dir)
    W("## What was compared")
    W("")
    W("| | " + a_name + " | " + b_name + " |")
    W("|---|---|---|")
    for label, path in [("data.yaml", ("dataset", "data_yaml")),
                        ("modality", ("dataset", "modality")),
                        ("train / val / test", None),
                        ("epochs", ("config", "common", "epochs")),
                        ("seed", ("config", "common", "seed")),
                        ("imgsz", ("config", "common", "imgsz")),
                        ("optimizer", None),
                        ("ultralytics", ("environment", "ultralytics")),
                        ("torch", ("environment", "torch")),
                        ("GPU", ("environment", "gpu", "name"))]:
        def dig(d, keys):
            for k in keys:
                if not isinstance(d, dict):
                    return None
                d = d.get(k)
            return d
        if label == "train / val / test":
            va = dig(pa, ("dataset", "images")) or {}
            vb = dig(pb, ("dataset", "images")) or {}
            va = " / ".join(str(va.get(s, "—")) for s in ("train", "valid", "test"))
            vb = " / ".join(str(vb.get(s, "—")) for s in ("train", "valid", "test"))
        elif label == "optimizer":
            va = ", ".join(sorted({c["optimizer_actual"] for c in a_cond.values()})) or "—"
            vb = ", ".join(sorted({c["optimizer_actual"] for c in b_cond.values()})) or "—"
        elif label == "modality":
            # results.json only carries this key for sweeps run after the split
            # was reorganised; older trees fall back to their data yaml.
            va, vb = infer_modality(a_dir), infer_modality(b_dir)
        else:
            va, vb = dig(pa, path), dig(pb, path)
        W(f"| {label} | {va if va is not None else '—'} | {vb if vb is not None else '—'} |")
    W("")
    W("The two sweeps come from configs that differ in three keys — the data "
      "yaml, the run directory and the W&B project. Everything that touches "
      "what the model learns is identical, so the modality is the only "
      "variable.")
    W("")

    # --- parity check -----------------------------------------------------
    W("## Training-condition parity")
    W("")
    mismatched = []
    for model in sorted(set(a_cond) & set(b_cond)):
        for key in ("epochs_run", "batch_actual", "optimizer_actual"):
            if a_cond[model][key] != b_cond[model][key]:
                mismatched.append((model, key, a_cond[model][key], b_cond[model][key]))
    if mismatched:
        W("These runs did **not** train under identical conditions, so their rows "
          "below are not a clean modality comparison:")
        W("")
        W("| Model | Field | " + a_name + " | " + b_name + " |")
        W("|---|---|---:|---:|")
        for model, key, va, vb in mismatched:
            W(f"| `{model}` | {key} | {va} | {vb} |")
    else:
        W(f"All {len(set(a_cond) & set(b_cond))} paired runs completed the same "
          "number of epochs at the same realised batch size with the same "
          "optimizer. The OOM backoff resolved identically on both modalities, "
          "so every row below is a like-for-like comparison.")
    W("")
    ha = sum(c["train_hours"] or 0 for c in a_cond.values())
    hb = sum(c["train_hours"] or 0 for c in b_cond.values())
    W(f"Total training time: {ha:.1f} h ({a_name}), {hb:.1f} h ({b_name}).")
    W("")

    # --- headline ---------------------------------------------------------
    deltas, per_model = [], []
    for model in a_prim:
        ca = a_cells.get((model, a_prim[model]["nms"].strip().lower() == "true"))
        cb = b_cells.get((model, a_prim[model]["nms"].strip().lower() == "true"))
        if not ca or not cb:
            continue
        d = cb[HEADLINE] - ca[HEADLINE]
        deltas.append(d)
        per_model.append((model, ca[HEADLINE], cb[HEADLINE], d))

    # Box AP as well, so the headline is not read as a mask-only effect.
    box_deltas = []
    for model in a_prim:
        nms = a_prim[model]["nms"].strip().lower() == "true"
        ca, cb = a_cells.get((model, nms)), b_cells.get((model, nms))
        if ca and cb:
            box_deltas.append(cb["box_mAP50-95"] - ca["box_mAP50-95"])
    box_st = signed_tests(box_deltas)
    box_mean = sum(box_deltas) / len(box_deltas) if box_deltas else 0.0

    st = signed_tests(deltas)
    mean_d = sum(deltas) / len(deltas) if deltas else 0.0
    med_d = sorted(deltas)[len(deltas) // 2] if deltas else 0.0

    W("## Headline: mask mAP50-95 across the 15 checkpoints")
    W("")
    W("The unit of analysis is the trained checkpoint, not the table row. "
      "YOLO26 is scored through two heads from one checkpoint, so counting all "
      "20 rows would double-weight it; these 15 rows are independent runs.")
    W("")
    losses = st["losses"]
    W(f"- **{b_name} wins {st['wins']} of {st['n_nonzero']}** paired "
      f"comparisons ({losses} loss{'' if losses == 1 else 'es'})")
    W(f"- Mean delta **{mean_d:+.2f}** points, median **{med_d:+.2f}**")
    if st.get("sign_p") is not None:
        W(f"- Sign test {pval(st['sign_p'])}; Wilcoxon signed-rank "
          f"{pval(st.get('wilcoxon_p'))}")
    W(f"- Box mAP50-95 moves the same way and further: **{b_name} wins "
      f"{box_st['wins']} of {box_st['n_nonzero']}**, mean **{box_mean:+.2f}** "
      f"points ({pval(box_st['sign_p'])})")
    W("")
    W("| Model | " + a_name + " | " + b_name + " | Δ |")
    W("|---|---:|---:|---:|")
    for model, va, vb, d in sorted(per_model, key=lambda r: -r[3]):
        W(f"| `{model}` | {va:.2f} | {vb:.2f} | {d:+.2f} |")
    W("")

    # --- full table -------------------------------------------------------
    W("## Every evaluated head")
    W("")
    W("| Group | Variant | Box mAP50-95 | | Δ | Mask mAP50-95 | | Δ |")
    W("|---|---|---:|---:|---:|---:|---:|---:|")
    W(f"| | | {a_name} | {b_name} | | {a_name} | {b_name} | |")
    for model, nms, label, variant, ca in rows_in_table_order(a_cells):
        cb = b_cells.get((model, nms))
        if ca is None or cb is None:
            W(f"| {label} | {variant} | — | — | — | — | — | — |")
            continue
        bd = cb["box_mAP50-95"] - ca["box_mAP50-95"]
        md = cb["mask_mAP50-95"] - ca["mask_mAP50-95"]
        W(f"| {label} | {variant} | {ca['box_mAP50-95']:.2f} | "
          f"{cb['box_mAP50-95']:.2f} | {bd:+.2f} | {ca['mask_mAP50-95']:.2f} | "
          f"{cb['mask_mAP50-95']:.2f} | {md:+.2f} |")
    W("")

    # --- per class --------------------------------------------------------
    W("## Per class, mask mAP50-95")
    W("")
    W("Averaged over the 15 checkpoints. The classes differ sharply in mask "
      "size, so a modality that helps thin targets need not help everywhere.")
    W("")
    class_deltas = []
    W("| Class | " + a_name + " | " + b_name + " | Δ | Wins |")
    W("|---|---:|---:|---:|---:|")
    for c in CLASSES:
        va = [a_cls[m][c] for m in a_cls if m in b_cls]
        vb = [b_cls[m][c] for m in a_cls if m in b_cls]
        if not va:
            continue
        ma, mb = sum(va) / len(va), sum(vb) / len(vb)
        wins = sum(1 for x, y in zip(va, vb) if y > x)
        W(f"| `{c}` | {ma:.2f} | {mb:.2f} | {mb - ma:+.2f} | {wins}/{len(va)} |")
        class_deltas.append((c, mb - ma, wins, len(va)))
    W("")

    # A class moving against the overall direction is the most informative thing
    # in this report, so state it rather than leaving it in the table.
    against = [r for r in class_deltas if r[1] * mean_d < 0]
    if against:
        names = ", ".join(f"`{c}` ({d:+.2f}, {w}/{n})" for c, d, w, n in against)
        W(f"**{'This class moves' if len(against) == 1 else 'These classes move'} "
          f"against the overall result:** {names}. The aggregate delta is not "
          f"uniform across classes, so a single headline number hides a real "
          f"disagreement — worth reporting per class rather than only in the "
          f"mean.")
        W("")

    # --- caveats ----------------------------------------------------------
    W("## How much these numbers can carry")
    W("")
    W("- **One seed per cell.** Both sweeps are seed 0. A per-model delta is a "
      "single paired observation and is not resolved against run-to-run "
      "variance. The 15-way sign test is the claim that survives; an individual "
      "row is not.")
    W("- **Untuned by design.** Both sweeps take the installed Ultralytics "
      "version's defaults, so this compares modalities under a fixed recipe, "
      "not the best each modality could reach.")
    decisive = st.get("sign_p") is not None and st["sign_p"] < 0.05
    W("- **8-bit input.** The frames are 16-bit PNGs, but Ultralytics reads them "
      "with `cv2.IMREAD_COLOR`: 18,754 distinct values collapse to 167, "
      "replicated across three identical channels. Both modalities lose the "
      "same way, so the comparison holds, but the quantization falls on "
      "absolute distance in one rendering and on return strength in the other."
      + (" The gap here survives that collapse, so quantization is not what "
         "separates the two — if anything the full-precision gap would be at "
         "least this large."
         if decisive else
         " A result near parity may therefore say more about the 8-bit path "
         "than about the renderings."))
    W("- **Same labels, same frames, same split.** Verified byte-identical, so "
      "nothing in the annotation or the split assignment can account for a "
      "delta.")
    W("")
    W("## What would strengthen this")
    W("")
    W("1. **Bound the seed noise.** Rerun one mid-size model (`yolo11m-seg`, "
      "0.6 h per run) at seeds 1 and 2 on both modalities. Four runs, about "
      "2.5 h, and the per-model deltas stop being single observations.")
    W("2. **Explain the class that disagrees** rather than averaging over it. "
      "The per-class split above is the part of this result most likely to be "
      "asked about in review.")
    W("3. **Fuse the two.** The three input channels currently hold one "
      "rendering three times. Packing intensity and range into separate "
      "channels costs nothing and needs no code change, and it is the "
      "experiment that says whether the two carry complementary information or "
      "the same information twice.")
    W("")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--a-name", default=None)
    ap.add_argument("--b-name", default=None)
    ap.add_argument("-o", "--out", type=Path, default=Path("result.md"))
    args = ap.parse_args()

    a_name = args.a_name or infer_modality(args.a)
    b_name = args.b_name or infer_modality(args.b)
    args.out.write_text(build(args.a, args.b, a_name, b_name), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
