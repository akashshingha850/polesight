#!/usr/bin/env python3
"""
Comprehensive Dataset Analysis Script

Builds a fully structured statistical analysis of the object-detection dataset
(schema, instances, split distribution, image file sizes and bounding-box
statistics) and writes it as a machine-readable JSON report. It also renders a
set of publication-quality figures into ``data/.figure/``, each accompanied by
a JSON file holding the exact data that figure plots.
"""

import os
import json
import yaml
import numpy as np
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
FIGURE_DIR = ROOT_DIR / ".figure"

SPLIT_ALIASES = {
    "train": ("train",),
    "val": ("valid", "val"),
    "valid": ("valid", "val"),
    "test": ("test",),
}

SPLITS = ["train", "valid", "test"]


def resolve_split_dir(split_name):
    for candidate in SPLIT_ALIASES.get(split_name, (split_name,)):
        candidate_dir = DATA_DIR / candidate
        if candidate_dir.exists():
            return candidate_dir
    return DATA_DIR / split_name


def load_classes_from_yaml():
    """Load class names from data.yaml."""
    yaml_file = DATA_DIR / "data.yaml"
    if yaml_file.exists():
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)
            if data and "names" in data:
                return data["names"]
    return []


TRAIN_DIR = resolve_split_dir("train")
VAL_DIR = resolve_split_dir("valid")
TEST_DIR = resolve_split_dir("test")

# Class names and a fixed, CVD-safe categorical palette (visual reference
# palette slots 1-5, in fixed class-id order — never cycled/re-assigned).
CLASS_NAMES = load_classes_from_yaml()
CLASS_PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7"]


def class_name(class_id):
    if 0 <= class_id < len(CLASS_NAMES):
        return CLASS_NAMES[class_id]
    return f"class_{class_id}"


def class_color(class_id):
    return CLASS_PALETTE[class_id % len(CLASS_PALETTE)]


# ===== FILE / SCHEMA HELPERS =====

def count_files(directory, ext=None):
    """Counts files with a given extension in a directory and its subdirectories."""
    count = 0
    for _, _, files in os.walk(directory):
        if ext:
            count += len([f for f in files if f.endswith(ext)])
        else:
            count += len(files)
    return count


def count_annotated_images(images_dir, labels_dir):
    """Count images that have a corresponding non-empty annotation file."""
    annotated_count = 0
    for root, _, files in os.walk(images_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                label_file = os.path.splitext(file)[0] + ".txt"
                rel_dir = os.path.relpath(root, images_dir)
                rel_dir = "" if rel_dir == "." else rel_dir
                label_path = (
                    os.path.join(labels_dir, rel_dir, label_file)
                    if rel_dir
                    else os.path.join(labels_dir, label_file)
                )
                if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                    annotated_count += 1
    return annotated_count


def analyze_set(images_dir, labels_dir):
    """Analyze a specific dataset split (images/labels/annotation coverage)."""
    if not (os.path.exists(images_dir) and os.path.exists(labels_dir)):
        return None
    image_count = count_files(images_dir, ".jpg") + count_files(images_dir, ".png")
    label_count = count_files(labels_dir, ".txt")
    annotated_count = count_annotated_images(images_dir, labels_dir)
    return {
        "images": image_count,
        "labels": label_count,
        "annotated": annotated_count,
        "unannotated": image_count - annotated_count,
    }


def analyze_image_file_sizes():
    """Analyze file sizes (KB) of all images in the dataset."""
    all_sizes = []
    for split in SPLITS:
        images_dir = resolve_split_dir(split) / "images"
        if images_dir.exists():
            for filename in os.listdir(images_dir):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    all_sizes.append(os.path.getsize(images_dir / filename) / 1024)
    if not all_sizes:
        return {"min": 0.0, "max": 0.0, "avg": 0.0, "total_files": 0, "sizes_kb": []}
    return {
        "min": float(np.min(all_sizes)),
        "max": float(np.max(all_sizes)),
        "avg": float(np.mean(all_sizes)),
        "median": float(np.median(all_sizes)),
        "total_files": len(all_sizes),
        "sizes_kb": [round(s, 2) for s in all_sizes],
    }


# ===== INSTANCE / BBOX ANALYSIS =====

def analyze_bounding_boxes(labels_dir):
    """Analyze bounding-box statistics from YOLO-format labels for one split.

    Also captures object centers and per-image structure (objects per image and
    class co-occurrence) so downstream figures can show spatial and density
    distributions.
    """
    widths, heights, areas, aspect_ratios, class_ids = [], [], [], [], []
    x_centers, y_centers = [], []
    class_counts = defaultdict(int)
    objects_per_image = []
    cooccurrence = defaultdict(int)  # (class_a, class_b) with a <= b -> count

    for filename in os.listdir(labels_dir):
        if not filename.endswith(".txt"):
            continue
        image_classes = []
        with open(labels_dir / filename, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(float(parts[0]))
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    widths.append(width)
                    heights.append(height)
                    x_centers.append(x_center)
                    y_centers.append(y_center)
                    areas.append(width * height)
                    aspect_ratios.append(width / height if height > 0 else 1.0)
                    class_ids.append(class_id)
                    class_counts[class_id] += 1
                    image_classes.append(class_id)
        if image_classes:
            objects_per_image.append(len(image_classes))
            present = sorted(set(image_classes))
            for i, a in enumerate(present):
                for b in present[i:]:
                    cooccurrence[(a, b)] += 1

    return {
        "widths": widths,
        "heights": heights,
        "x_centers": x_centers,
        "y_centers": y_centers,
        "areas": areas,
        "aspect_ratios": aspect_ratios,
        "class_ids": class_ids,
        "class_counts": dict(class_counts),
        "objects_per_image": objects_per_image,
        "cooccurrence": {f"{a},{b}": c for (a, b), c in cooccurrence.items()},
        "total_boxes": len(widths),
    }


def calc_stats(data):
    """Basic descriptive statistics for a list of values."""
    if not data:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "median": float(np.median(data)),
    }


# ===== REPORT ASSEMBLY =====

def build_report():
    """Run the full analysis and return (report_dict, raw_arrays)."""
    class_ids_sorted = sorted(range(len(CLASS_NAMES)))

    # --- Split-level image/annotation counts ---
    split_stats = {}
    for split in SPLITS:
        d = resolve_split_dir(split)
        split_stats[split] = analyze_set(d / "images", d / "labels")

    total_images = sum(s["images"] for s in split_stats.values() if s)
    total_annotated = sum(s["annotated"] for s in split_stats.values() if s)

    # --- Instances per class, per split ---
    instances_by_split = {}
    for split in SPLITS:
        counts = defaultdict(int)
        labels_path = resolve_split_dir(split) / "labels"
        if labels_path.exists():
            for filename in os.listdir(labels_path):
                if filename.endswith(".txt"):
                    with open(labels_path / filename, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                counts[int(float(parts[0]))] += 1
        instances_by_split[split] = {cid: counts[cid] for cid in class_ids_sorted}

    overall_instances = {
        cid: sum(instances_by_split[s][cid] for s in SPLITS) for cid in class_ids_sorted
    }
    total_instances = sum(overall_instances.values())

    # --- Bounding boxes ---
    bbox_by_split = {}
    for split in SPLITS:
        labels_path = resolve_split_dir(split) / "labels"
        if labels_path.exists():
            bbox_by_split[split] = analyze_bounding_boxes(labels_path)

    raw = {"widths": [], "heights": [], "x_centers": [], "y_centers": [],
           "areas": [], "aspect_ratios": [], "class_ids": [], "objects_per_image": []}
    for split_stats_bbox in bbox_by_split.values():
        for key in raw:
            raw[key].extend(split_stats_bbox[key])

    # Combined class co-occurrence across splits.
    cooccurrence = defaultdict(int)
    for s in bbox_by_split.values():
        for key, count in s["cooccurrence"].items():
            cooccurrence[key] += count
    raw["cooccurrence"] = dict(cooccurrence)

    bbox_summary = {
        "total_boxes": len(raw["widths"]),
        "width": calc_stats(raw["widths"]),
        "height": calc_stats(raw["heights"]),
        "area": calc_stats(raw["areas"]),
        "aspect_ratio": calc_stats(raw["aspect_ratios"]),
        "objects_per_image": calc_stats(raw["objects_per_image"]),
        "per_split": {
            split: {
                "total_boxes": s["total_boxes"],
                "class_counts": {cid: s["class_counts"].get(cid, 0) for cid in class_ids_sorted},
                "aspect_ratio": calc_stats(s["aspect_ratios"]),
                "objects_per_image": calc_stats(s["objects_per_image"]),
            }
            for split, s in bbox_by_split.items()
        },
    }

    file_size_stats = analyze_image_file_sizes()

    def pct(n, d):
        return round(n / d * 100, 2) if d else 0.0

    report = {
        "dataset_overview": {
            "total_images": total_images,
            "total_annotations": total_instances,
            "images_with_annotations": total_annotated,
            "background_images": total_images - total_annotated,
            "num_classes": len(CLASS_NAMES),
            "class_names": CLASS_NAMES,
            "image_format": "JPEG",
            "annotation_format": "YOLO (normalized coordinates)",
            "original_resolution": "960 x 640 pixels",
            "training_resolution": "640 x 640 pixels (resized)",
            "aspect_ratio": "3:2 (1.5:1)",
            "color_space": "RGB",
            "dataset_type": "Object Detection",
            "file_size_kb": {
                "min": round(file_size_stats["min"], 1),
                "max": round(file_size_stats["max"], 1),
                "avg": round(file_size_stats["avg"], 1),
                "median": round(file_size_stats.get("median", 0.0), 1),
                "total_files": file_size_stats["total_files"],
            },
        },
        "splits": {
            split: {
                **(split_stats[split] or {"images": 0, "labels": 0, "annotated": 0, "unannotated": 0}),
                "instances": instances_by_split[split],
                "total_instances": sum(instances_by_split[split].values()),
                "image_pct": pct(split_stats[split]["images"] if split_stats[split] else 0, total_images),
                "instance_pct": pct(sum(instances_by_split[split].values()), total_instances),
            }
            for split in SPLITS
        },
        "class_distribution": {
            class_name(cid): {
                "class_id": cid,
                "count": overall_instances[cid],
                "pct": pct(overall_instances[cid], total_instances),
                "color": class_color(cid),
            }
            for cid in class_ids_sorted
        },
        "bounding_boxes": bbox_summary,
    }
    return report, raw


# ===== JSON OUTPUT =====

def write_json_report(report):
    output_file = DATA_DIR / "dataset_analysis_report.json"
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)
    return output_file


# ===== FIGURES =====

def _apply_style():
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 11,
        "text.color": "#0b0b0b",
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": "#52514e",
        "axes.titlecolor": "#0b0b0b",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.8,
        "xtick.color": "#898781",
        "ytick.color": "#898781",
        "xtick.labelcolor": "#52514e",
        "ytick.labelcolor": "#52514e",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save(fig, name, data):
    """Save a figure as PNG plus a companion JSON of the plotted data."""
    png_path = FIGURE_DIR / f"{name}.png"
    json_path = FIGURE_DIR / f"{name}.json"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    return png_path


def generate_figures(report, raw):
    """Render all figures + companion JSON into data/.figure/."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _apply_style()

    cd = report["class_distribution"]
    names = list(cd.keys())
    counts = [cd[n]["count"] for n in names]
    colors = [cd[n]["color"] for n in names]
    figures = []

    # 1. Overall class distribution (magnitude -> single-hue, sorted desc, direct labels)
    order = np.argsort(counts)[::-1]
    o_names = [names[i] for i in order]
    o_counts = [counts[i] for i in order]
    o_colors = [colors[i] for i in order]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(o_names[::-1], o_counts[::-1], color=o_colors[::-1], height=0.62)
    ax.set_xlabel("Instances")
    ax.set_title("Class distribution (all splits)")
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    total = sum(counts)
    for b, c in zip(bars, o_counts[::-1]):
        ax.text(b.get_width() + total * 0.01, b.get_y() + b.get_height() / 2,
                f"{c}  ({c/total*100:.1f}%)", va="center", ha="left",
                color="#52514e", fontsize=10)
    ax.set_xlim(0, max(counts) * 1.18)
    figures.append(_save(fig, "class_distribution_overall",
                         {"chart": "horizontal_bar", "measure": "instances",
                          "classes": o_names, "counts": o_counts, "total": total}))
    plt.close(fig)

    # 2. Class distribution by split (grouped bars; series = split)
    split_palette = {"train": "#2a78d6", "valid": "#1baf7a", "test": "#eda100"}
    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.8))
    by_split_data = {}
    for i, split in enumerate(SPLITS):
        vals = [report["splits"][split]["instances"][cd[n]["class_id"]] for n in names]
        by_split_data[split] = vals
        ax.bar(x + (i - 1) * w, vals, w, label=split, color=split_palette[split])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Instances")
    ax.set_title("Class distribution by split")
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    ax.legend(frameon=False, title="Split")
    figures.append(_save(fig, "class_distribution_by_split",
                         {"chart": "grouped_bar", "measure": "instances",
                          "classes": names, "series": by_split_data}))
    plt.close(fig)

    # 3. Split distribution — images vs instances share per split
    img_counts = [report["splits"][s]["images"] for s in SPLITS]
    inst_counts = [report["splits"][s]["total_instances"] for s in SPLITS]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, vals, title in ((axes[0], img_counts, "Images"), (axes[1], inst_counts, "Instances")):
        b = ax.bar(SPLITS, vals, color=[split_palette[s] for s in SPLITS], width=0.6)
        ax.set_title(title)
        ax.yaxis.grid(True)
        ax.xaxis.grid(False)
        tot = sum(vals)
        for bar, v in zip(b, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + tot * 0.01,
                    f"{v}\n{v/tot*100:.1f}%", ha="center", va="bottom",
                    color="#52514e", fontsize=9)
        ax.set_ylim(0, max(vals) * 1.18)
    fig.suptitle("Train / validation / test split", fontsize=13, fontweight="bold", color="#0b0b0b")
    figures.append(_save(fig, "split_distribution",
                         {"chart": "bar", "splits": SPLITS,
                          "images": img_counts, "instances": inst_counts}))
    plt.close(fig)

    # 4. Annotation coverage (stacked: annotated vs background per split)
    annotated = [report["splits"][s]["annotated"] for s in SPLITS]
    background = [report["splits"][s]["unannotated"] for s in SPLITS]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.bar(SPLITS, annotated, color="#2a78d6", label="Annotated", width=0.55)
    ax.bar(SPLITS, background, bottom=annotated, color="#e34948", label="Background", width=0.55)
    ax.set_ylabel("Images")
    ax.set_title("Annotation coverage per split")
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    ax.legend(frameon=False)
    for i, s in enumerate(SPLITS):
        ax.text(i, annotated[i] / 2, str(annotated[i]), ha="center", va="center",
                color="#ffffff", fontsize=10, fontweight="bold")
        if background[i] > 0:
            ax.text(i, annotated[i] + background[i] + max(annotated) * 0.01,
                    str(background[i]), ha="center", va="bottom", color="#52514e", fontsize=9)
    figures.append(_save(fig, "annotation_coverage",
                         {"chart": "stacked_bar", "splits": SPLITS,
                          "annotated": annotated, "background": background}))
    plt.close(fig)

    # 5. Bounding-box dimension distributions (small multiples of histograms)
    dims = [("widths", "Width (norm.)"), ("heights", "Height (norm.)"),
            ("areas", "Area (norm.)"), ("aspect_ratios", "Aspect ratio (w/h)")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    hist_data = {}
    for ax, (key, label) in zip(axes.ravel(), dims):
        vals = np.array(raw[key], dtype=float)
        if key == "aspect_ratios":  # clip long tail for readability
            vals = vals[vals <= np.percentile(vals, 99)]
        n, edges = np.histogram(vals, bins=30)
        ax.hist(vals, bins=30, color="#2a78d6", edgecolor="#fcfcfb", linewidth=0.5)
        med = float(np.median(raw[key]))
        ax.axvline(med, color="#e34948", linewidth=1.5, linestyle="--")
        ax.text(med, ax.get_ylim()[1] * 0.92, f"median {med:.2f}", color="#e34948",
                fontsize=9, ha="left" if med < np.mean(ax.get_xlim()) else "right")
        ax.set_title(label)
        ax.set_ylabel("Count")
        ax.yaxis.grid(True)
        ax.xaxis.grid(False)
        hist_data[key] = {"bin_edges": [round(e, 4) for e in edges.tolist()],
                          "counts": n.tolist(), "median": round(med, 4)}
    fig.suptitle("Bounding-box dimension distributions (normalized)",
                 fontsize=13, fontweight="bold", color="#0b0b0b")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    figures.append(_save(fig, "bbox_dimension_distributions",
                         {"chart": "histogram_small_multiples", "histograms": hist_data}))
    plt.close(fig)

    # 6. Bounding-box width vs height scatter, colored by class (identity -> categorical)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    w_arr = np.array(raw["widths"])
    h_arr = np.array(raw["heights"])
    c_arr = np.array(raw["class_ids"])
    for cid in sorted(set(raw["class_ids"])):
        m = c_arr == cid
        ax.scatter(w_arr[m], h_arr[m], s=18, alpha=0.6, color=class_color(cid),
                   edgecolors="none", label=class_name(cid))
    ax.set_xlabel("Width (normalized)")
    ax.set_ylabel("Height (normalized)")
    ax.set_title("Bounding-box width vs height by class")
    ax.legend(frameon=False, markerscale=1.5, fontsize=9)
    ax.grid(True)
    figures.append(_save(fig, "bbox_width_height_scatter",
                         {"chart": "scatter", "x": "width", "y": "height",
                          "points_by_class": {
                              class_name(cid): {
                                  "width": [round(float(v), 4) for v in w_arr[c_arr == cid]],
                                  "height": [round(float(v), 4) for v in h_arr[c_arr == cid]],
                              } for cid in sorted(set(raw["class_ids"]))}}))
    plt.close(fig)

    # 7. Image file-size distribution
    sizes = report["dataset_overview"]["file_size_kb"]
    all_sizes = analyze_image_file_sizes()["sizes_kb"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    n, edges, _ = ax.hist(all_sizes, bins=30, color="#1baf7a", edgecolor="#fcfcfb", linewidth=0.5)
    ax.axvline(sizes["avg"], color="#e34948", linewidth=1.5, linestyle="--")
    ax.text(sizes["avg"], ax.get_ylim()[1] * 0.92, f"avg {sizes['avg']:.0f} KB",
            color="#e34948", fontsize=9, ha="left")
    ax.set_xlabel("File size (KB)")
    ax.set_ylabel("Images")
    ax.set_title("Image file-size distribution")
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    figures.append(_save(fig, "image_file_size_distribution",
                         {"chart": "histogram", "unit": "KB",
                          "bin_edges": [round(e, 2) for e in edges.tolist()],
                          "counts": n.astype(int).tolist(),
                          "min": sizes["min"], "max": sizes["max"], "avg": sizes["avg"]}))
    plt.close(fig)

    # 8. Object center spatial heatmap (where objects sit in the frame)
    from matplotlib.colors import LinearSegmentedColormap
    blue_ramp = LinearSegmentedColormap.from_list(
        "viz_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#184f95", "#0d366b"])
    xc = np.array(raw["x_centers"])
    yc = np.array(raw["y_centers"])
    heat, xedges, yedges = np.histogram2d(xc, yc, bins=24, range=[[0, 1], [0, 1]])
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    im = ax.imshow(heat.T, origin="upper", extent=[0, 1, 1, 0], aspect="auto",
                   cmap=blue_ramp, interpolation="nearest")
    ax.set_xlabel("x center (normalized)")
    ax.set_ylabel("y center (normalized)")
    ax.set_title("Object center spatial distribution")
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Objects", color="#52514e")
    cbar.outline.set_edgecolor("#c3c2b7")
    figures.append(_save(fig, "object_center_heatmap",
                         {"chart": "heatmap_2d", "x": "x_center", "y": "y_center",
                          "x_edges": [round(e, 4) for e in xedges.tolist()],
                          "y_edges": [round(e, 4) for e in yedges.tolist()],
                          "counts": heat.astype(int).tolist()}))
    plt.close(fig)

    # 9. Objects-per-image distribution (annotation density)
    opi = np.array(raw["objects_per_image"])
    max_opi = int(opi.max())
    bins = np.arange(0.5, max_opi + 1.5, 1)
    n, _ = np.histogram(opi, bins=bins)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(range(1, max_opi + 1), n, color="#2a78d6", width=0.85)
    med = float(np.median(opi))
    ax.axvline(med, color="#e34948", linewidth=1.5, linestyle="--")
    ax.text(med, ax.get_ylim()[1] * 0.92, f"median {med:.0f}", color="#e34948",
            fontsize=9, ha="left")
    ax.set_xlabel("Objects per image")
    ax.set_ylabel("Images")
    ax.set_title("Annotation density (objects per image)")
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    figures.append(_save(fig, "objects_per_image",
                         {"chart": "histogram", "objects_per_image": list(range(1, max_opi + 1)),
                          "image_counts": n.tolist(), "median": round(med, 2),
                          "mean": round(float(opi.mean()), 2)}))
    plt.close(fig)

    # 10. Bounding-box area by class (distribution / size profile per class)
    c_arr = np.array(raw["class_ids"])
    area_arr = np.array(raw["areas"])
    present_ids = sorted(set(raw["class_ids"]))
    area_by_class = [area_arr[c_arr == cid] for cid in present_ids]
    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(area_by_class, vert=True, patch_artist=True, widths=0.6,
                    showfliers=True, flierprops=dict(marker="o", markersize=3,
                    markerfacecolor="#898781", markeredgecolor="none", alpha=0.4))
    for patch, cid in zip(bp["boxes"], present_ids):
        patch.set_facecolor(class_color(cid))
        patch.set_alpha(0.85)
        patch.set_edgecolor("#fcfcfb")
    for element in ("whiskers", "caps"):
        for line in bp[element]:
            line.set_color("#898781")
    for line in bp["medians"]:
        line.set_color("#0b0b0b")
        line.set_linewidth(1.5)
    ax.set_xticklabels([class_name(cid) for cid in present_ids], rotation=20, ha="right")
    ax.set_ylabel("Area (normalized)")
    ax.set_title("Bounding-box area by class")
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    figures.append(_save(fig, "bbox_area_by_class",
                         {"chart": "boxplot", "measure": "area",
                          "classes": [class_name(cid) for cid in present_ids],
                          "stats_by_class": {
                              class_name(cid): calc_stats(list(area_arr[c_arr == cid]))
                              for cid in present_ids}}))
    plt.close(fig)

    # 11. Class composition across splits (100% stacked — stratification check)
    fig, ax = plt.subplots(figsize=(9, 5))
    left = np.zeros(len(names))
    comp_data = {}
    for split in SPLITS:
        vals = np.array([report["splits"][split]["instances"][cd[n_]["class_id"]] for n_ in names], dtype=float)
        comp_data[split] = vals
    totals = sum(comp_data[s] for s in SPLITS)
    totals[totals == 0] = 1  # avoid div-by-zero
    for split in SPLITS:
        frac = comp_data[split] / totals * 100
        ax.barh(names, frac, left=left, color=split_palette[split], label=split, height=0.62)
        for i, (f_val, l_val) in enumerate(zip(frac, left)):
            if f_val >= 6:
                ax.text(l_val + f_val / 2, i, f"{f_val:.0f}%", ha="center", va="center",
                        color="#ffffff", fontsize=8, fontweight="bold")
        left += frac
    ax.set_xlabel("Share of instances (%)", labelpad=8)
    ax.set_title("Class composition across splits")
    ax.set_xlim(0, 100)
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    fig.subplots_adjust(bottom=0.24)
    figures.append(_save(fig, "class_split_composition",
                         {"chart": "stacked_bar_100", "classes": names,
                          "split_percent": {s: (comp_data[s] / totals * 100).round(2).tolist() for s in SPLITS},
                          "split_counts": {s: comp_data[s].astype(int).tolist() for s in SPLITS}}))
    plt.close(fig)

    # 12. Class co-occurrence matrix (which pole types share an image)
    k = len(names)
    matrix = np.zeros((k, k), dtype=int)
    for key, count in raw["cooccurrence"].items():
        a, b = (int(v) for v in key.split(","))
        matrix[a, b] = count
        matrix[b, a] = count
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(matrix, cmap=blue_ramp)
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_yticklabels(names)
    ax.set_title("Class co-occurrence (images sharing both classes)")
    ax.grid(False)
    thresh = matrix.max() * 0.55 if matrix.max() else 1
    for i in range(k):
        for j in range(k):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=9,
                    color="#ffffff" if matrix[i, j] > thresh else "#0b0b0b")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Images", color="#52514e")
    cbar.outline.set_edgecolor("#c3c2b7")
    figures.append(_save(fig, "class_cooccurrence",
                         {"chart": "matrix", "classes": names,
                          "note": "diagonal = images containing the class; off-diagonal = images containing both",
                          "matrix": matrix.tolist()}))
    plt.close(fig)

    add_sample_montage(figures, plt)

    return figures


def _load_intensity_display(path):
    """Load an image (incl. 16-bit ``I;16`` intensity strips) as a 0-1 float
    array, contrast-stretched to the 2-98 percentile for legible display."""
    from PIL import Image

    arr = np.asarray(Image.open(path)).astype(np.float32)
    if arr.ndim == 3:  # already RGB/multi-channel -> luminance for a uniform look
        arr = arr[..., :3].mean(axis=2)
    lo, hi = np.percentile(arr, (2, 98))
    return np.clip((arr - lo) / (hi - lo + 1e-6), 0.0, 1.0)


def _select_sample_images(n=6):
    """Pick sample images that together cover as many classes as possible,
    preferring frames with several/varied objects. Returns [(img, label), ...]."""
    from matplotlib import image as _mpimg  # noqa: F401  (ensures mpl is initialised)

    candidates = []
    for split in ("train", "valid", "test"):
        split_dir = resolve_split_dir(split)
        images_dir, labels_dir = split_dir / "images", split_dir / "labels"
        if not images_dir.exists():
            continue
        for img in sorted(images_dir.iterdir()):
            if img.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            label = labels_dir / f"{img.stem}.txt"
            classes = set()
            if label.exists():
                for line in label.read_text().splitlines():
                    parts = line.split()
                    if parts:
                        classes.add(int(float(parts[0])))
            if classes:  # skip background-only frames for the montage
                candidates.append((img, label, classes))

    chosen, covered = [], set()
    # Greedy: repeatedly take the frame adding the most uncovered classes.
    remaining = candidates[:]
    while remaining and len(chosen) < n:
        remaining.sort(key=lambda c: (len(c[2] - covered), len(c[2])), reverse=True)
        img, label, classes = remaining.pop(0)
        chosen.append((img, label))
        covered |= classes
    return chosen


def add_sample_montage(figures, plt):
    """Qualitative Figure 1: real annotated sample frames with drawn boxes."""
    from matplotlib.patches import Rectangle

    samples = _select_sample_images(n=6)
    if not samples:
        return

    n = len(samples)
    fig, axes = plt.subplots(n, 1, figsize=(10, 1.55 * n + 0.4))
    if n == 1:
        axes = [axes]

    used_classes, manifest = set(), []
    for ax, (img_path, label_path) in zip(axes, samples):
        disp = _load_intensity_display(img_path)
        h, w = disp.shape[:2]
        ax.imshow(disp, cmap="gray", aspect="auto", vmin=0, vmax=1)
        boxes = []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                cid = int(float(p[0]))
                cx, cy, bw, bh = (float(v) for v in p[1:5])
                x0, y0 = (cx - bw / 2) * w, (cy - bh / 2) * h
                col = class_color(cid)
                ax.add_patch(Rectangle((x0, y0), bw * w, bh * h, fill=False,
                                       edgecolor=col, linewidth=1.6))
                ax.text(x0 + 1, max(y0 - 2, 6), class_name(cid), fontsize=7,
                        color="#ffffff", va="bottom",
                        bbox=dict(boxstyle="square,pad=0.12", fc=col, ec="none"))
                used_classes.add(cid)
                boxes.append({"class": class_name(cid), "cx": cx, "cy": cy,
                              "w": bw, "h": bh})
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_edgecolor("#c3c2b7")
        ax.set_ylabel(img_path.stem, rotation=0, ha="right", va="center",
                      fontsize=8, color="#898781", labelpad=8)
        manifest.append({"image": img_path.name, "boxes": boxes})

    axes[0].set_title("Sample annotated frames (intensity, boxes coloured by class)")
    # Shared class legend
    handles = [Rectangle((0, 0), 1, 1, fc=class_color(c), ec="none")
               for c in sorted(used_classes)]
    labels = [class_name(c) for c in sorted(used_classes)]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    figures.append(_save(fig, "sample_annotated_frames",
                         {"chart": "image_montage",
                          "note": "qualitative examples; boxes in YOLO-normalised coords",
                          "samples": manifest}))
    plt.close(fig)


def main():
    report, raw = build_report()
    json_path = write_json_report(report)
    figures = generate_figures(report, raw)

    print(f"JSON report saved to: {json_path}")
    print(f"Generated {len(figures)} figures in: {FIGURE_DIR}")
    for p in figures:
        print(f"  - {p.name}  (+ {p.stem}.json)")


if __name__ == "__main__":
    main()
