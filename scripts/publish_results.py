#!/usr/bin/env python3
"""Assemble a publishable results tree from a train.py sweep directory.

A sweep writes a deep tree under its project dir, with absolute paths baked into
args.yaml and results.json. The released form is flat — one directory per model
holding four files — with every path rewritten repository-relative, so the
records mean the same thing in someone else's clone.

    python publish_results.py run/segment/polesight-range-filtered results_range_filtered

Idempotent: re-running overwrites the destination from the sweep.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# eval.json sits at <project>/<model>/; the rest one level deeper, in the
# Ultralytics run directory itself.
AT_MODEL = ["eval.json"]
AT_RUN = ["args.yaml", "results.csv", "verify.json"]
AT_ROOT = ["eval_table.csv", "results.json"]


def rewrite(text: str, project: Path, dest_name: str) -> str:
    """Absolute sweep paths -> repository-relative published paths."""
    root = str(ROOT).rstrip("/") + "/"
    project_rel = str(project.resolve().relative_to(ROOT))
    # Longest first: the project path contains the root prefix.
    text = text.replace(root + project_rel, dest_name)
    text = text.replace(project_rel, dest_name)
    text = text.replace(root, "")
    # The published tree writes the data yaml with a leading ./ — match it.
    text = re.sub(r"(?<![\w./])data/data(_[\w]+)?\.yaml", r"./data/data\1.yaml", text)
    text = text.replace("././", "./")
    return text


def clean_log(text: str) -> str:
    """Collapse progress bars to their finished frame.

    Ultralytics redraws each bar many times per epoch; captured to a file rather
    than a terminal, every redraw lands as its own line and the log balloons
    ~20x. The published results/sweep.log keeps only completed frames, so match
    it: drop bar lines that are not at 100%, keep every other line untouched.
    """
    out = []
    for line in text.splitlines():
        if "\u2501" in line and "100%" not in line:
            continue
        out.append(line)
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("project", type=Path, help="sweep project dir")
    ap.add_argument("dest", type=Path, help="published results dir")
    ap.add_argument("--log", type=Path, help="sweep log to copy in as sweep.log")
    args = ap.parse_args()

    project, dest = args.project, args.dest
    dest_name = str(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    models = sorted(p.name for p in project.iterdir()
                    if p.is_dir() and (p / "eval.json").exists())
    if not models:
        raise SystemExit(f"no model directories with eval.json under {project}")

    for model in models:
        out = dest / model
        out.mkdir()
        for name in AT_MODEL:
            src = project / model / name
            out.joinpath(name).write_text(
                rewrite(src.read_text(encoding="utf-8"), project, dest_name),
                encoding="utf-8")
        run_dir = project / model / model
        for name in AT_RUN:
            src = run_dir / name
            if not src.exists():
                print(f"  ! {model}: missing {name}")
                continue
            if name == "results.csv":       # pure numbers, no paths
                shutil.copy2(src, out / name)
            else:
                out.joinpath(name).write_text(
                    rewrite(src.read_text(encoding="utf-8"), project, dest_name),
                    encoding="utf-8")
        print(f"  {model}: {len(list(out.iterdir()))} files")

    for name in AT_ROOT:
        src = project / name
        if not src.exists():
            raise SystemExit(f"missing {src} — run train.py --eval first")
        dest.joinpath(name).write_text(
            rewrite(src.read_text(encoding="utf-8"), project, dest_name),
            encoding="utf-8")

    if args.log and args.log.exists():
        # The log quotes the paths it trained on, so it needs the same rewrite
        # as the records — the published results/sweep.log carries none either.
        dest.joinpath("sweep.log").write_text(
            clean_log(rewrite(args.log.read_text(encoding="utf-8", errors="replace"),
                              project, dest_name)),
            encoding="utf-8")

    leftover = [p for p in dest.rglob("*")
                if p.is_file() and str(ROOT) in p.read_text(errors="ignore")[:200000]]
    if leftover:
        raise SystemExit("absolute paths survived in: "
                         + ", ".join(str(p) for p in leftover[:5]))

    report = json.loads((dest / "results.json").read_text(encoding="utf-8"))
    print(f"\n{dest}: {len(models)} models, "
          f"{len(report.get('runs', []))} run records, "
          f"modality={report.get('dataset', {}).get('modality')}")


if __name__ == "__main__":
    main()
