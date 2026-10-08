#!/usr/bin/env python3
"""Render a short animated overview of the paired PoleSight dataset."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "valid", "test")
MODALITIES = ("intensity_filtered", "range_filtered")
CLASSES = ("fence_pole", "gantry_sign_pole", "light_pole", "traffic_pole")
CLASS_COLORS = ("#CC79A7", "#E69F00", "#0072B2", "#009E73")
BG = "#FFFFFF"
MUTED = "#526474"
WHITE = "#172B3A"
ACCENT = "#00756A"
FAMILY_COLORS = {"yolov8": "#4A3AA7", "yolo11": "#2A78D6", "yolo26": "#EB6834"}
FAMILY_LABELS = {"yolov8": "YOLOv8", "yolo11": "YOLO11", "yolo26": "YOLO26"}
SPLIT_COLORS = {
    "images": {"train": "#2675D2", "valid": "#438CD2", "test": "#174F91"},
    "instances": {"train": "#4A3AA7", "valid": "#7162BD", "test": "#30216F"},
}


def parse_instances(label_path: Path) -> list[tuple[int, np.ndarray]]:
    """Read YOLO polygons and boxes, returning class IDs and pixel points."""
    found = []
    if not label_path.exists():
        return found
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        try:
            class_id = int(float(values[0]))
            coords = np.asarray([float(v) for v in values[1:]], dtype=float)
        except ValueError:
            continue
        if coords.size >= 6 and coords.size % 2 == 0:
            found.append((class_id, coords.reshape(-1, 2) * (1024, 128)))
        elif coords.size == 4:
            cx, cy, width, height = coords
            x0, x1 = (cx - width / 2) * 1024, (cx + width / 2) * 1024
            y0, y1 = (cy - height / 2) * 128, (cy + height / 2) * 128
            found.append((class_id, np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])))
    return found


def read_display_image(path: Path) -> np.ndarray:
    """Contrast-stretch a 16-bit frame for a compact screen-size preview."""
    with Image.open(path) as source:
        values = np.asarray(source, dtype=np.float32)
    low, high = np.percentile(values, (1, 99.5))
    if high <= low:
        high = low + 1
    return np.clip((values - low) * (255.0 / (high - low)), 0, 255)


def dataset_summary(data_dir: Path):
    split_counts = {}
    class_counts = Counter()
    class_by_split = {}
    images = {}
    for split in SPLITS:
        image_dir = data_dir / split / MODALITIES[0] / "images"
        label_dir = data_dir / split / MODALITIES[0] / "labels"
        split_images = sorted(image_dir.glob("*.png"))
        split_counts[split] = len(split_images)
        class_by_split[split] = Counter()
        for label in label_dir.glob("*.txt"):
            for line in label.read_text(encoding="utf-8").splitlines():
                try:
                    class_id = int(float(line.split()[0]))
                    class_counts[class_id] += 1
                    class_by_split[split][class_id] += 1
                except (ValueError, IndexError):
                    continue
        images[split] = split_images
    return split_counts, class_counts, class_by_split, images


def choose_examples(data_dir: Path, split_counts, images):
    """Pick clear, annotated frames, preferring examples from every class."""
    candidates = []
    for split in SPLITS:
        for image_path in images[split]:
            label = data_dir / split / MODALITIES[0] / "labels" / f"{image_path.stem}.txt"
            objects = parse_instances(label)
            class_ids = {cid for cid, _ in objects}
            if objects:
                candidates.append((image_path, split, objects, class_ids))
    chosen, covered = [], set()
    while candidates and len(chosen) < 4:
        best = max(candidates, key=lambda row: (len(row[3] - covered), min(len(row[2]), 6)))
        chosen.append(best)
        covered.update(best[3])
        candidates.remove(best)
    return chosen


def read_primary_records(table_path: Path) -> dict[str, dict]:
    """Read test metrics once per independently trained checkpoint."""
    records = {}
    with table_path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row.get("is_primary_head", "").strip().lower() != "true":
                continue
            records[row["model"]] = {
                "family": row["family"],
                "mask_ap": 100 * float(row["mask_mAP50-95"]),
                "box_ap": 100 * float(row["box_mAP50-95"]),
                "latency_ms": float(row["inference_ms"]) + float(row["postprocess_ms"]),
            }
    return records


def load_paired_results(results_dir: Path):
    intensity_dir = next((results_dir / name for name in ("intensity_filtered", "intesity_filtered")
                          if (results_dir / name / "eval_table.csv").exists()), None)
    range_dir = next((results_dir / name for name in ("results_range_filtered", "range_filtered")
                      if (results_dir / name / "eval_table.csv").exists()), None)
    if intensity_dir is None or range_dir is None:
        raise SystemExit(f"Expected paired eval_table.csv files under {results_dir}")
    intensity = read_primary_records(intensity_dir / "eval_table.csv")
    ranges = read_primary_records(range_dir / "eval_table.csv")
    common = set(intensity) & set(ranges)
    if not common:
        raise SystemExit(f"No matching model checkpoints found under {results_dir}")

    family_order = {"yolov8": 0, "yolo11": 1, "yolo26": 2}
    scale_order = {"n": 0, "s": 1, "m": 2, "l": 3, "x": 4}

    def model_key(model):
        family = next((f for f in family_order if model.startswith(f)), "")
        variant = model[len(family):len(family) + 1]
        return family_order.get(family, 9), scale_order.get(variant, 9)

    models = sorted(common, key=model_key)
    mask_deltas = [ranges[model]["mask_ap"] - intensity[model]["mask_ap"] for model in models]
    box_deltas = [ranges[model]["box_ap"] - intensity[model]["box_ap"] for model in models]
    return models, mask_deltas, box_deltas, intensity, ranges


def make_gif(data_dir: Path, results_dir: Path, output: Path):
    split_counts, class_counts, class_by_split, images = dataset_summary(data_dir)
    total_images = sum(split_counts.values())
    total_instances = sum(class_counts.values())
    examples = choose_examples(data_dir, split_counts, images)
    if not total_images or not examples:
        raise SystemExit(f"No annotated dataset images found under {data_dir}")
    models, mask_deltas, box_deltas, intensity, ranges = load_paired_results(results_dir)

    fig = plt.figure(figsize=(12, 7), dpi=100, facecolor=BG)
    image_title = fig.text(0.05, 0.955, "Annotated panoramic examples", color=WHITE,
                           fontsize=18, fontweight="bold", va="top")
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.925, 0.925], color="#D5DEE5", lw=0.8,
                              transform=fig.transFigure))

    # Each example uses two full-width rows so the panorama keeps its wide form.
    image_axes = [
        fig.add_axes([0.055, 0.44, 0.935, 0.35], facecolor="#F3F6F8"),
        fig.add_axes([0.055, 0.02, 0.935, 0.35], facecolor="#F3F6F8"),
    ]
    intensity_label = fig.text(0.025, 0.615, "Intensity", color=MUTED,
                               fontsize=11.5, fontweight="bold", va="center",
                               ha="center", rotation=90)
    range_label = fig.text(0.025, 0.195, "Range", color=ACCENT,
                           fontsize=11.5, fontweight="bold", va="center",
                           ha="center", rotation=90)
    frame_label = fig.text(0.055, 0.845, "", color=MUTED, fontsize=9.5,
                           ha="left", va="center")

    # Color family denotes measure; shade denotes train/validation/test.
    stats_title = fig.text(0.05, 0.955, "Dataset split statistics", color=WHITE,
                           fontsize=18, fontweight="bold", va="top", visible=False)
    stats_subtitle = fig.text(0.05, 0.885,
                              "Split composition within each category · segment labels show counts",
                              color=MUTED, fontsize=10, visible=False)
    stats_ax = fig.add_axes([0.035, 0.02, 0.93, 0.88], facecolor=BG)
    stats_ax.set_axis_off()
    stats_ax.set_xlim(0, 100)
    stats_ax.set_ylim(0, 100)

    def key(x, y, label, fill):
        stats_ax.add_patch(Rectangle((x, y - 2.2), 4.0, 4.0,
                                     facecolor=fill, edgecolor=fill, linewidth=0.4))
        stats_ax.text(x + 5.3, y, label, ha="left", va="center", fontsize=9, color=MUTED)

    legend_y = 91
    stats_ax.text(0, legend_y, "Images", ha="left", va="center", fontsize=9,
                  fontweight="bold", color=MUTED)
    for x, split in zip((8, 20, 32), SPLITS):
        key(x, legend_y, split.capitalize(), SPLIT_COLORS["images"][split])
    stats_ax.text(46, legend_y, "Instances", ha="left", va="center", fontsize=9,
                  fontweight="bold", color=MUTED)
    for x, split in zip((57, 71, 85), SPLITS):
        key(x, legend_y, split.capitalize(), SPLIT_COLORS["instances"][split])
    stats_ax.text(98, legend_y, "Total", ha="right", va="center", fontsize=9, color=MUTED)

    def stacked_row(y, label, values, measure, bar_height=8.5):
        x_start, bar_width = 10.0, 81.0
        total = sum(values)
        x = x_start
        stats_ax.text(0, y + bar_height / 2, label, ha="left", va="center",
                      fontsize=10, fontweight="bold", color=WHITE)
        for split, value in zip(SPLITS, values):
            width = bar_width * value / max(total, 1)
            stats_ax.add_patch(Rectangle((x, y), width, bar_height,
                                         facecolor=SPLIT_COLORS[measure][split],
                                         edgecolor=BG, linewidth=0.5))
            if width >= 4.2:
                stats_ax.text(x + width / 2, y + bar_height / 2, f"{value:,}",
                              ha="center", va="center", fontsize=9,
                              fontweight="bold", color="white")
            else:
                stats_ax.text(x + width / 2, y + bar_height + 1.0, f"{value:,}",
                              ha="center", va="bottom", fontsize=8, color=WHITE)
            x += width
        stats_ax.text(98, y + bar_height / 2, f"{total:,}", ha="right", va="center",
                      fontsize=11, fontweight="bold", color=WHITE)

    stacked_row(70, "Images", [split_counts[s] for s in SPLITS], "images")
    class_row_y = (54, 37, 20, 3)
    for class_id, y in enumerate(class_row_y):
        class_label = CLASSES[class_id].replace("_pole", "").replace("_", " ")
        values = [class_by_split[s][class_id] for s in SPLITS]
        stacked_row(y, class_label, values, "instances", bar_height=8.5)

    # Paired speed/accuracy view: each arrow moves one checkpoint from intensity to range.
    result_title = fig.text(0.05, 0.955, "Range input lifts mask AP across YOLO checkpoints",
                            color=WHITE, fontsize=18, fontweight="bold", va="top",
                            visible=False)
    mask_wins = sum(delta > 0 for delta in mask_deltas)
    mask_mean = float(np.mean(mask_deltas))
    result_summary = fig.text(
        0.05, 0.885,
        f"Test split · seed 0 · {mask_wins}/{len(mask_deltas)} checkpoints improve · "
        f"mean +{mask_mean:.2f} pp mask mAP$_{{50:95}}$ · each arrow: intensity → range",
        color=MUTED, fontsize=12, visible=False,
    )
    all_latencies = [row["latency_ms"] for records in (intensity, ranges)
                     for row in records.values()]
    all_scores = [row["mask_ap"] for records in (intensity, ranges)
                  for row in records.values()]
    result_ax = fig.add_axes([0.07, 0.10, 0.60, 0.68], facecolor=BG)
    gain_ax = fig.add_axes([0.765, 0.10, 0.205, 0.68], facecolor=BG)
    for ax in (result_ax, gain_ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["left"].set_color("#AAB7C2")
        ax.spines["bottom"].set_color("#AAB7C2")
        ax.tick_params(axis="both", colors=MUTED, labelsize=15)
        ax.set_axisbelow(True)
    result_ax.grid(True, which="major", color="#E5EAEE", linewidth=0.65, zorder=0)
    result_ax.set_xscale("log")
    result_ax.set_xlim(min(all_latencies) * 0.90, max(all_latencies) * 1.16)
    result_ax.set_ylim(min(all_scores) - 0.35, max(all_scores) + 0.35)
    result_ax.set_xlabel("Latency (ms/image, log scale)", color=MUTED, fontsize=16, labelpad=3)
    result_ax.set_ylabel("Mask mAP$_{50:95}$ (%)", color=MUTED, fontsize=16, labelpad=4)
    result_ax.set_xticks([1.5, 2, 3, 4, 5, 6])
    result_ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    result_ax.set_xticks([], minor=True)

    family_order = ("yolov8", "yolo11", "yolo26")
    family_short = {"yolov8": "v8", "yolo11": "11", "yolo26": "26"}
    marker_area = 80

    def short_name(model):
        family = intensity[model]["family"]
        return f"{family_short[family]}{model.removesuffix('-seg')[-1]}"

    for model in models:
        color = FAMILY_COLORS[intensity[model]["family"]]
        start = (intensity[model]["latency_ms"], intensity[model]["mask_ap"])
        end = (ranges[model]["latency_ms"], ranges[model]["mask_ap"])
        # Skip the arrow when the pair nearly overlaps; the shrink would invert it.
        p0, p1 = result_ax.transData.transform([start, end])
        if np.hypot(*(p1 - p0)) > 18:
            result_ax.annotate(
                "", xy=end, xytext=start, zorder=3,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, alpha=0.75,
                                mutation_scale=11, shrinkA=5, shrinkB=6),
            )
        result_ax.scatter(*start, s=marker_area, marker="o", facecolor="white",
                          edgecolor=color, linewidth=1.6, zorder=4)
        result_ax.scatter(*end, s=marker_area * 1.3, marker="o", color=color,
                          edgecolor="white", linewidth=1.2, zorder=5)
        result_ax.annotate(short_name(model), end, xytext=(7, 0),
                           textcoords="offset points", ha="left", va="center",
                           fontsize=14.5, color=WHITE, zorder=6)

    # Exact per-checkpoint gain, grouped by family in the same colors.
    gain_y, gain_labels, y = [], [], 0.0
    for family in family_order:
        for model in [m for m in models if intensity[m]["family"] == family]:
            delta = ranges[model]["mask_ap"] - intensity[model]["mask_ap"]
            gain_ax.barh(y, delta, height=0.72, color=FAMILY_COLORS[family],
                         edgecolor=BG, linewidth=0.5, zorder=3)
            gain_ax.text(max(delta, 0) + 0.06, y, f"{delta:+.2f}", ha="left", va="center",
                         fontsize=14, color=MUTED, zorder=4,
                         bbox=dict(facecolor=BG, edgecolor="none", pad=0.8))
            gain_y.append(y)
            gain_labels.append(short_name(model))
            y += 1
        y += 0.6
    gain_ax.set_yticks(gain_y, gain_labels)
    gain_ax.tick_params(axis="y", length=0, labelsize=14.5, labelcolor=WHITE)
    gain_ax.set_ylim(y - 0.6 + 0.2, -0.8)
    gain_ax.set_xlim(min(0.0, min(mask_deltas)) - 0.15, max(mask_deltas) + 1.05)
    gain_ax.axvline(0, color="#7B8A97", linewidth=1.0, zorder=2)
    gain_ax.axvline(mask_mean, color=MUTED, linewidth=1.0, linestyle=(0, (3, 2)), zorder=2)
    gain_ax.text(mask_mean, -0.75, f"mean +{mask_mean:.2f}", ha="center", va="bottom",
                 fontsize=14, color=MUTED)
    gain_ax.grid(True, axis="x", color="#E5EAEE", linewidth=0.65, zorder=0)
    gain_ax.set_xlabel("Range − intensity (pp)", color=MUTED, fontsize=16, labelpad=3)
    gain_ax.spines["left"].set_visible(False)

    family_handles = [Line2D([], [], color=FAMILY_COLORS[family], marker="o",
                             linestyle="none", markersize=12, label=FAMILY_LABELS[family])
                      for family in family_order]
    modality_handles = [
        Line2D([], [], color=MUTED, marker="o", markerfacecolor="white",
               markeredgewidth=1.6, linestyle="none", markersize=12, label="Intensity"),
        Line2D([], [], color=MUTED, marker="o", linestyle="none", markersize=12,
               label="Range"),
    ]
    result_legend = fig.legend(
        handles=family_handles + modality_handles, loc="upper left",
        bbox_to_anchor=(0.045, 0.855), ncol=5, frameon=False, fontsize=15.5,
        handlelength=1.0, columnspacing=1.4, labelcolor=MUTED,
    )
    dataset_meta = fig.text(
        0.97, 0.955,
        f"{total_images:,} images · {total_instances:,} instances · 1024 × 128 px",
        color=MUTED, fontsize=9.5, ha="right", va="top",
    )

    total_frames = len(examples[:4]) + 2

    def draw_pair(ax, sample, split, objects, col):
        ax.clear()
        modality = MODALITIES[col]
        path = data_dir / split / modality / "images" / sample.name
        ax.imshow(read_display_image(path), cmap="viridis" if col else "gray",
                  aspect="auto", interpolation="nearest")
        for class_id, points in objects:
            ax.add_patch(Polygon(points, closed=True, fill=False,
                                 edgecolor=CLASS_COLORS[class_id % len(CLASS_COLORS)], linewidth=1.6))
        ax.set_xlim(0, 1024)
        ax.set_ylim(128, 0)
        ax.set_axis_off()

    def update(frame):
        image_scene = frame < len(examples[:4])
        stats_scene = frame == len(examples[:4])
        result_scene = frame == len(examples[:4]) + 1
        if image_scene:
            sample, split, objects, _ = examples[frame]
        else:
            sample, split, objects, _ = examples[0]
        for col, ax in enumerate(image_axes):
            ax.set_visible(image_scene)
            if image_scene:
                draw_pair(ax, sample, split, objects, col)
        image_title.set_visible(image_scene)
        intensity_label.set_visible(image_scene)
        range_label.set_visible(image_scene)
        frame_label.set_visible(image_scene)
        if image_scene:
            frame_label.set_text(f"{split.capitalize()} · {sample.stem}")
        stats_title.set_visible(stats_scene)
        stats_subtitle.set_visible(stats_scene)
        stats_ax.set_visible(stats_scene)
        result_title.set_visible(result_scene)
        result_summary.set_visible(result_scene)
        result_ax.set_visible(result_scene)
        gain_ax.set_visible(result_scene)
        result_legend.set_visible(result_scene)
        dataset_meta.set_visible(True)
        return []

    animation = FuncAnimation(fig, update, frames=total_frames, interval=2000, blit=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = output.parent / f"{output.stem}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_names = [
        f"{index + 1:02d}_sample_{split}_{sample.stem}.png"
        for index, (sample, split, _, _) in enumerate(examples[:4])
    ] + ["05_dataset_split_statistics.png", "06_speed_accuracy_results.png"]
    for frame, name in enumerate(frame_names):
        update(frame)
        fig.savefig(frames_dir / name, dpi=100, facecolor=BG)
    animation.save(output, writer=PillowWriter(fps=0.5), dpi=100)
    plt.close(fig)
    print(f"Saved {output} and {len(frame_names)} PNG frames under {frames_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data",
                        help="dataset directory containing train/valid/test")
    parser.add_argument("--results-dir", type=Path,
                        help="results directory containing paired modality eval_table.csv files")
    parser.add_argument("--output", type=Path, default=ROOT / "assets" / "dataset_overview.gif",
                        help="destination GIF path")
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve() if args.results_dir else data_dir.parent / "results"
    make_gif(data_dir, results_dir, args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
