#!/usr/bin/env python3
"""Rebuild the baseline figures (paper Table II) from the released run artifacts.

Reads ``results/`` and emits the instance-segmentation baselines as CSV or JSON,
so the published numbers are generated from the evaluation records rather than
transcribed by hand.

Sources, per model:
    results/eval_table.csv          metrics and inference latency, one row per
                                    evaluated head
    results/<model>/eval.json       parameter count and GFLOPs
    results/<model>/verify.json     the batch size the run actually used, after
                                    train.py's OOM backoff

Usage:
    python scripts/make_table.py                 # CSV to stdout
    python scripts/make_table.py --format json   # the same figures as JSON
    python scripts/make_table.py --check paper/sec/4_dataset.tex
                                                 # verify a manuscript's table
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"

VARIANTS = ["n", "s", "m", "l", "x"]

# One column group per table header. ``nms`` picks the evaluated head: YOLO26
# trains a single checkpoint and is scored twice, through the one2many (NMS)
# and one2one (NMS-free) heads, which is why its capacity rows repeat.
GROUPS = [
    ("YOLOv8-seg (NMS)", "yolov8", "yolov8{v}-seg", True),
    ("YOLO11-seg (NMS)", "yolo11", "yolo11{v}-seg", True),
    ("YOLO26-seg (NMS)", "yolo26", "yolo26{v}-seg", True),
    ("YOLO26-seg (NMS-free)", "yolo26", "yolo26{v}-seg", False),
]

# (label, source key, formatter). Metrics are stored as fractions and reported
# as percentages, matching the caption.
ROWS = [
    ("Params (M)", "params_m", "{:.2f}"),
    ("GFLOPs", "gflops", "{:.1f}"),
    ("Batch", "batch", "{:.0f}"),
    (None, None, None),  # rule
    ("Box mAP$_{50}$", "box_mAP50", "{:.2f}"),
    ("Box mAP$_{50:95}$", "box_mAP50-95", "{:.2f}"),
    ("Box Precision", "box_precision", "{:.2f}"),
    ("Box Recall", "box_recall", "{:.2f}"),
    (None, None, None),
    ("Mask mAP$_{50}$", "mask_mAP50", "{:.2f}"),
    ("Mask mAP$_{50:95}$", "mask_mAP50-95", "{:.2f}"),
    ("Mask Precision", "mask_precision", "{:.2f}"),
    ("Mask Recall", "mask_recall", "{:.2f}"),
    (None, None, None),
    ("Latency (ms)", "inference_ms", "{:.1f}"),
]

PERCENT_KEYS = {
    "box_mAP50", "box_mAP50-95", "box_precision", "box_recall",
    "mask_mAP50", "mask_mAP50-95", "mask_precision", "mask_recall",
}


def load_cells(results_dir: Path) -> dict[tuple[str, bool], dict]:
    """Collect one record per (model, nms) evaluated head."""
    table = results_dir / "eval_table.csv"
    if not table.exists():
        raise SystemExit(f"missing {table} — run train.py --eval first")

    cells: dict[tuple[str, bool], dict] = {}
    with table.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            model = row["model"]
            nms = row["nms"].strip().lower() == "true"
            cell = {"inference_ms": float(row["inference_ms"])}
            for key in PERCENT_KEYS:
                cell[key] = float(row[key]) * 100.0
            cells[(model, nms)] = cell

    # Capacity and the realised batch size live beside each run, not in the CSV.
    for (model, _nms), cell in cells.items():
        eval_path = results_dir / model / "eval.json"
        if eval_path.exists():
            record = json.loads(eval_path.read_text(encoding="utf-8"))
            cell["params_m"] = record["parameters"] / 1e6
            cell["gflops"] = record["gflops"]
        verify_path = results_dir / model / "verify.json"
        if verify_path.exists():
            record = json.loads(verify_path.read_text(encoding="utf-8"))
            cell["batch"] = record["batch_actual"]
    return cells


def build_grid(cells: dict[tuple[str, bool], dict]) -> list[list[dict | None]]:
    """Lay the records out in table order: group-major, then variant."""
    grid = []
    for _label, _family, stem, nms in GROUPS:
        grid.append([cells.get((stem.format(v=v), nms)) for v in VARIANTS])
    return grid


def render_csv(grid: list[list[dict | None]]) -> str:
    out = ["group,variant," + ",".join(key for _l, key, _f in ROWS if key)]
    for (label, _family, stem, _nms), row in zip(GROUPS, grid):
        for variant, cell in zip(VARIANTS, row):
            values = [
                "" if cell is None or cell.get(key) is None else fmt.format(cell[key])
                for _l, key, fmt in ROWS
                if key
            ]
            out.append(f'"{label}",{variant},' + ",".join(values))
    return "\n".join(out)


def render_json(grid: list[list[dict | None]]) -> str:
    """The same figures as CSV, as one record per evaluated model."""
    records = []
    for (label, family, stem, nms), row in zip(GROUPS, grid):
        for variant, cell in zip(VARIANTS, row):
            record = {
                "group": label,
                "family": family,
                "model": stem.format(v=variant),
                "variant": variant,
                "nms": nms,
            }
            for _label, key, fmt in ROWS:
                if not key:
                    continue
                value = None if cell is None else cell.get(key)
                # Round to the reported precision so JSON and CSV agree, and
                # keep whole-number fields (batch) as integers.
                if value is None:
                    record[key] = None
                else:
                    text = fmt.format(value)
                    record[key] = int(text) if fmt.endswith(".0f}") else float(text)
            records.append(record)
    return json.dumps(records, indent=2)


def check(grid: list[list[dict | None]], tex_path: Path) -> int:
    """Compare every generated figure against the numbers in a manuscript."""
    source = tex_path.read_text(encoding="utf-8")
    mismatches = 0
    for label, key, fmt in ROWS:
        if label is None:
            continue
        # Cells may contain \textbf{...}, so match lazily up to the row
        # terminator at end of line rather than excluding backslashes.
        pattern = re.escape(label) + r"\s*&(.*?)\\\\\s*$"
        found = re.search(pattern, source, re.MULTILINE)
        if not found:
            print(f"  ?  {label}: row not found in {tex_path.name}")
            continue
        printed = [
            c.strip().replace(r"\textbf{", "").replace("}", "").strip()
            for c in found.group(1).split("&")
        ]
        expected = [
            "--" if cell is None or cell.get(key) is None else fmt.format(cell[key])
            for row in grid
            for cell in row
        ]
        for i, (want, got) in enumerate(zip(expected, printed)):
            if want != got:
                group = GROUPS[i // 5][0]
                variant = VARIANTS[i % 5]
                print(f"  X  {label} [{group} {variant}]: paper={got} results={want}")
                mismatches += 1
    print("all figures match" if not mismatches else f"{mismatches} mismatch(es)")
    return 1 if mismatches else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--format", choices=["csv", "json"], default="csv")
    parser.add_argument(
        "--check",
        type=Path,
        help="a .tex file to verify the generated figures against",
    )
    args = parser.parse_args(argv)

    grid = build_grid(load_cells(args.results))
    if args.check:
        return check(grid, args.check)
    print(render_csv(grid) if args.format == "csv" else render_json(grid))
    return 0


if __name__ == "__main__":
    sys.exit(main())
