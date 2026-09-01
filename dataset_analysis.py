#!/usr/bin/env python3
"""
Comprehensive Dataset Analysis Script (instance segmentation)

Builds a fully structured statistical analysis of the YOLO instance-segmentation
dataset — schema, instances, split distribution, image properties, polygon/mask
geometry and annotation validity — and writes it as a machine-readable JSON
report, a self-contained HTML dashboard and a Markdown summary. It also renders
publication-quality figures into ``figure/`` (a symlink to the paper
repository's figures folder), each accompanied by a JSON file holding the exact data that figure plots.

Labels are polygons (``class x1 y1 x2 y2 ... xn yn``); a handful of rows in this
dataset are still plain boxes (``class cx cy w h``). Both are parsed, and each
instance's bounding box is derived from its polygon extent so that every geometry
statistic is correct for masks. Geometry is reported in pixels as well as
normalized units, because the frames are extremely wide (1024x128) and normalized
width/height are not comparable across axes.

Coverage relative to NVIDIA TAO's Data Analytics module (analyze/validate):
  * object count, bounding-box area, image size, invalid-coordinate report and
    images-with-annotations are all produced here;
  * occlusion and truncation are KITTI-only fields that do not exist in YOLO
    labels, so they cannot be computed;
  * the kpi_analyze precision-recall curve needs model inference and belongs with
    the training pipeline (``train.py`` / Ultralytics ``val``), not a dataset
    analyser.
"""

import os
import json
import yaml
import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
# ``figure`` is a symlink to ../PoleSight----WACV-2027/figures, so generated
# figures land straight in the paper repository.
FIGURE_DIR = ROOT_DIR / "figure" / "sample"
# ``paper`` is a symlink to ../PoleSight-IVCNZ2026. Paper figures are written
# separately from the exploratory dump above: one standalone PDF per figure, at
# print size, so they can be placed individually in the manuscript.
PAPER_FIGURE_DIR = ROOT_DIR / "paper" / "figures" / "dataset"

# IEEE two-column geometry, in inches.
COL_W = 3.48
FULL_W = 7.16
# One cell of a full-width 1x3 subfigure row (0.32\textwidth each). The three
# statistics panels are authored at exactly this size so LaTeX places them at
# scale 1.0 — fonts, rules and marks then match across all three.
PANEL_W = 0.32 * FULL_W
PANEL_H = 1.88

SPLIT_ALIASES = {
    "train": ("train",),
    "val": ("valid", "val"),
    "valid": ("valid", "val"),
    "test": ("test",),
}

SPLITS = ["train", "valid", "test"]
IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# Cap on how many offending files are named per validation issue type.
MAX_ISSUE_EXAMPLES = 10

# COCO's small/medium/large thresholds expressed as a fraction of the frame area
# (32^2 and 96^2 px on a 640x480 image), so they stay meaningful at any resolution.
SIZE_CATEGORY_THRESHOLDS = (1024 / 307200, 9216 / 307200)
SIZE_CATEGORIES = ["small", "medium", "large"]


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
# Slot order is fixed and never cycled. These five hues clear the all-pairs
# checks as a set (worst normal-vision dE 15.6, worst protan/deutan dE 9.1);
# the assignment below additionally matches contrast-against-white inversely to
# median instance size, so the thinnest classes get the most legible slots:
#   fence   violet  8.33:1  (median mask  30 px, thinnest)
#   gantry  yellow  2.11:1  (median mask 269 px, largest)
#   light   blue    4.30:1  (56% of all instances)
#   power   aqua    2.74:1  (7 instances)
#   traffic green   4.82:1
CLASS_PALETTE = ["#4a3aa7", "#eda100", "#2a78d6", "#1baf7a", "#008300"]
SPLIT_PALETTE = {"train": "#2a78d6", "valid": "#1baf7a", "test": "#eda100"}


def class_name(class_id):
    if 0 <= class_id < len(CLASS_NAMES):
        return CLASS_NAMES[class_id]
    return f"class_{class_id}"


def class_color(class_id):
    return CLASS_PALETTE[class_id % len(CLASS_PALETTE)]


# ===== FILE HELPERS =====

def list_images(split):
    d = resolve_split_dir(split) / "images"
    if not d.exists():
        return []
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def list_labels(split):
    d = resolve_split_dir(split) / "labels"
    if not d.exists():
        return []
    return sorted(d.rglob("*.txt"))


# ===== ANNOTATION PARSING =====

@dataclass
class Instance:
    """One annotated object, with geometry derived from its polygon extent."""

    split: str
    stem: str
    class_id: int
    kind: str  # "polygon" | "box"
    points: np.ndarray  # (N, 2) normalized vertices; empty for box rows
    cx: float
    cy: float
    w: float
    h: float
    mask_area: float  # normalized polygon area (0.0 for box rows)
    centroid: tuple  # normalized (x, y) — area centroid for polygons

    @property
    def bbox_area(self):
        return self.w * self.h

    @property
    def n_vertices(self):
        return len(self.points)

    @property
    def fill_ratio(self):
        """Polygon area / bounding-box area — how tightly the mask fits its box."""
        area = self.bbox_area
        return self.mask_area / area if (area > 0 and self.kind == "polygon") else None


def _shoelace(xs, ys):
    """Signed polygon area (normalized units) via the shoelace formula."""
    return 0.5 * float(np.dot(xs, np.roll(ys, -1)) - np.dot(np.roll(xs, -1), ys))


def _polygon_centroid(xs, ys, signed_area):
    """Area centroid of a polygon; falls back to the vertex mean when degenerate."""
    if abs(signed_area) < 1e-12:
        return float(np.mean(xs)), float(np.mean(ys))
    cross = xs * np.roll(ys, -1) - np.roll(xs, -1) * ys
    cx = float(np.dot(xs + np.roll(xs, -1), cross) / (6.0 * signed_area))
    cy = float(np.dot(ys + np.roll(ys, -1), cross) / (6.0 * signed_area))
    return cx, cy


def parse_label_file(path, split):
    """Parse one YOLO label file into instances plus any per-row issues found.

    An even coordinate count >= 6 is a polygon, exactly 4 coordinates is a
    bounding box, and anything else is malformed.
    """
    instances, issues = [], []
    seen_rows = set()
    stem = path.stem

    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            class_id = int(float(parts[0]))
        except ValueError:
            issues.append(("malformed_row", f"line {lineno}: unparsable class id"))
            continue

        try:
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            issues.append(("malformed_row", f"line {lineno}: non-numeric coordinate"))
            continue

        if line in seen_rows:
            issues.append(("duplicate_row", f"line {lineno}"))
        seen_rows.add(line)

        if not 0 <= class_id < len(CLASS_NAMES):
            issues.append(("class_id_out_of_range", f"line {lineno}: class {class_id}"))

        n = len(coords)
        if n >= 6 and n % 2 == 0:
            xs = np.array(coords[0::2], dtype=float)
            ys = np.array(coords[1::2], dtype=float)
            signed = _shoelace(xs, ys)
            x0, x1 = float(xs.min()), float(xs.max())
            y0, y1 = float(ys.min()), float(ys.max())
            inst = Instance(
                split=split, stem=stem, class_id=class_id, kind="polygon",
                points=np.stack([xs, ys], axis=1),
                cx=(x0 + x1) / 2, cy=(y0 + y1) / 2, w=x1 - x0, h=y1 - y0,
                mask_area=abs(signed),
                centroid=_polygon_centroid(xs, ys, signed),
            )
            if xs.min() < 0 or ys.min() < 0 or xs.max() > 1 or ys.max() > 1:
                issues.append(("coord_out_of_range", f"line {lineno}"))
            if inst.mask_area <= 0:
                issues.append(("zero_area_polygon", f"line {lineno}"))
        elif n == 4:
            cx, cy, w, h = coords
            issues.append(("box_only_row", f"line {lineno}: class {class_id}"))
            inst = Instance(
                split=split, stem=stem, class_id=class_id, kind="box",
                points=np.empty((0, 2)),
                cx=cx, cy=cy, w=w, h=h, mask_area=0.0, centroid=(cx, cy),
            )
            if min(cx - w / 2, cy - h / 2) < 0 or max(cx + w / 2, cy + h / 2) > 1:
                issues.append(("coord_out_of_range", f"line {lineno}"))
        else:
            issues.append(("malformed_row", f"line {lineno}: {n} coordinate(s)"))
            continue

        if inst.w <= 0 or inst.h <= 0:
            issues.append(("degenerate_extent", f"line {lineno}"))
        instances.append(inst)

    return instances, issues


# ===== IMAGE PROPERTIES =====

def analyze_image_properties():
    """Probe every image for real pixel size, mode, format and file size.

    Only the header is decoded (PIL is lazy), so this stays cheap.
    """
    from PIL import Image

    by_image = {}
    widths, heights, sizes_kb = [], [], []
    resolutions, modes, formats = Counter(), Counter(), Counter()

    for split in SPLITS:
        for path in list_images(split):
            size_kb = path.stat().st_size / 1024
            try:
                with Image.open(path) as im:
                    w, h = im.size
                    mode, fmt = im.mode, im.format
            except OSError:
                by_image[f"{split}/{path.stem}"] = None
                continue
            by_image[f"{split}/{path.stem}"] = {
                "width": w, "height": h, "mode": mode,
                "format": fmt, "size_kb": size_kb,
            }
            widths.append(w)
            heights.append(h)
            sizes_kb.append(size_kb)
            resolutions[f"{w}x{h}"] += 1
            modes[mode] += 1
            formats[fmt or "unknown"] += 1

    return {
        "by_image": by_image,
        "widths": widths,
        "heights": heights,
        "sizes_kb": sizes_kb,
        "resolutions": dict(resolutions.most_common()),
        "modes": dict(modes.most_common()),
        "formats": dict(formats.most_common()),
        "total_files": len(widths),
    }


def _aspect_label(w, h):
    """Human-readable aspect ratio, e.g. '8:1 (8.00:1)'."""
    if not h:
        return "unknown"
    g = np.gcd(int(w), int(h))
    return f"{int(w) // g}:{int(h) // g} ({w / h:.2f}:1)"


def _mode_color_space(mode):
    return {
        "I;16": "16-bit grayscale (single channel)",
        "I": "32-bit integer grayscale",
        "L": "8-bit grayscale",
        "RGB": "RGB",
        "RGBA": "RGBA",
    }.get(mode, mode)


# ===== STATS =====

def calc_stats(data):
    """Basic descriptive statistics for a list of values."""
    data = [v for v in data if v is not None]
    if not data:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "median": float(np.median(data)),
    }


def size_category(area_fraction):
    small, medium = SIZE_CATEGORY_THRESHOLDS
    if area_fraction < small:
        return "small"
    if area_fraction < medium:
        return "medium"
    return "large"


# ===== VALIDATION =====

@dataclass
class Validation:
    counts: Counter = field(default_factory=Counter)
    by_split: dict = field(default_factory=lambda: defaultdict(Counter))
    examples: dict = field(default_factory=lambda: defaultdict(list))

    def add(self, issue_type, split, where):
        self.counts[issue_type] += 1
        self.by_split[issue_type][split] += 1
        if len(self.examples[issue_type]) < MAX_ISSUE_EXAMPLES:
            self.examples[issue_type].append(where)

    def to_dict(self):
        return {
            "total_issues": int(sum(self.counts.values())),
            "clean": not self.counts,
            "issues": {
                issue: {
                    "count": int(count),
                    "by_split": {s: int(n) for s, n in self.by_split[issue].items()},
                    "examples": self.examples[issue],
                    "truncated": count > len(self.examples[issue]),
                }
                for issue, count in self.counts.most_common()
            },
        }


ISSUE_DESCRIPTIONS = {
    "box_only_row": "Bounding-box row in a segmentation dataset (no polygon)",
    "malformed_row": "Malformed row (bad coordinate count or non-numeric value)",
    "coord_out_of_range": "Coordinate outside the normalized [0, 1] range",
    "degenerate_extent": "Zero or negative box extent",
    "subpixel_instance": "Instance smaller than one pixel in width or height",
    "zero_area_polygon": "Polygon with zero area",
    "class_id_out_of_range": "Class id not present in data.yaml",
    "duplicate_row": "Exact duplicate annotation row in the same file",
    "image_without_label": "Image with no label file",
    "label_without_image": "Label file with no matching image",
}


# ===== REPORT ASSEMBLY =====

def build_report():
    """Run the full analysis and return (report_dict, raw_arrays)."""
    class_ids_sorted = sorted(range(len(CLASS_NAMES)))
    image_props = analyze_image_properties()
    validation = Validation()

    # --- Parse every label file, collecting instances and validity issues ---
    instances_by_split = {}
    label_stems = {}
    for split in SPLITS:
        found = []
        stems = set()
        for path in list_labels(split):
            stems.add(path.stem)
            parsed, issues = parse_label_file(path, split)
            found.extend(parsed)
            rel = path.relative_to(DATA_DIR)
            for issue_type, detail in issues:
                validation.add(issue_type, split, f"{rel} ({detail})")
        instances_by_split[split] = found
        label_stems[split] = stems

    # --- Pair images with labels; flag orphans on both sides ---
    split_stats = {}
    for split in SPLITS:
        images = list_images(split)
        image_stems = {p.stem for p in images}
        annotated = 0
        for path in images:
            label_path = resolve_split_dir(split) / "labels" / f"{path.stem}.txt"
            if label_path.exists() and label_path.stat().st_size > 0:
                annotated += 1
            elif not label_path.exists():
                validation.add("image_without_label", split,
                               str(path.relative_to(DATA_DIR)))
        for stem in sorted(label_stems[split] - image_stems):
            validation.add("label_without_image", split, f"{split}/labels/{stem}.txt")
        split_stats[split] = {
            "images": len(images),
            "labels": len(label_stems[split]),
            "annotated": annotated,
            "unannotated": len(images) - annotated,
        }

    # --- Attach pixel geometry to every instance ---
    raw = {k: [] for k in (
        "widths", "heights", "areas", "aspect_ratios", "widths_px", "heights_px",
        "bbox_areas_px", "mask_areas_px", "aspect_px", "fill_ratios", "n_vertices",
        "x_centers", "y_centers", "class_ids", "kinds", "size_categories",
        "objects_per_image")}
    default_w = int(np.median(image_props["widths"])) if image_props["widths"] else 1
    default_h = int(np.median(image_props["heights"])) if image_props["heights"] else 1

    per_split_instances = {}
    for split in SPLITS:
        for inst in instances_by_split[split]:
            meta = image_props["by_image"].get(f"{split}/{inst.stem}")
            img_w = meta["width"] if meta else default_w
            img_h = meta["height"] if meta else default_h
            w_px, h_px = inst.w * img_w, inst.h * img_h
            if 0 < w_px < 1 or 0 < h_px < 1:
                validation.add("subpixel_instance", split, f"{split}/labels/{inst.stem}.txt")
            raw["widths"].append(inst.w)
            raw["heights"].append(inst.h)
            raw["areas"].append(inst.bbox_area)
            raw["aspect_ratios"].append(inst.w / inst.h if inst.h > 0 else 0.0)
            raw["widths_px"].append(w_px)
            raw["heights_px"].append(h_px)
            raw["bbox_areas_px"].append(w_px * h_px)
            raw["mask_areas_px"].append(inst.mask_area * img_w * img_h)
            raw["aspect_px"].append(w_px / h_px if h_px > 0 else 0.0)
            raw["fill_ratios"].append(inst.fill_ratio)
            raw["n_vertices"].append(inst.n_vertices if inst.kind == "polygon" else None)
            raw["x_centers"].append(inst.centroid[0])
            raw["y_centers"].append(inst.centroid[1])
            raw["class_ids"].append(inst.class_id)
            raw["kinds"].append(inst.kind)
            area_ref = inst.mask_area if inst.kind == "polygon" else inst.bbox_area
            raw["size_categories"].append(size_category(area_ref))
        per_split_instances[split] = instances_by_split[split]

    # --- Per-image structure: density and class co-occurrence ---
    cooccurrence = Counter()
    for split in SPLITS:
        per_image = defaultdict(list)
        for inst in instances_by_split[split]:
            per_image[inst.stem].append(inst.class_id)
        for classes in per_image.values():
            raw["objects_per_image"].append(len(classes))
            present = sorted(set(classes))
            for i, a in enumerate(present):
                for b in present[i:]:
                    cooccurrence[f"{a},{b}"] += 1
    raw["cooccurrence"] = dict(cooccurrence)

    # --- Instance counts per class ---
    counts_by_split = {
        split: Counter(inst.class_id for inst in instances_by_split[split])
        for split in SPLITS
    }
    instances_per_split = {
        split: {cid: int(counts_by_split[split][cid]) for cid in class_ids_sorted}
        for split in SPLITS
    }
    overall_instances = {
        cid: sum(instances_per_split[s][cid] for s in SPLITS) for cid in class_ids_sorted
    }
    total_instances = sum(overall_instances.values())
    all_instances = [i for s in SPLITS for i in instances_by_split[s]]
    polygon_instances = sum(1 for i in all_instances if i.kind == "polygon")
    box_instances = len(all_instances) - polygon_instances

    total_images = sum(s["images"] for s in split_stats.values())
    total_annotated = sum(s["annotated"] for s in split_stats.values())

    def pct(n, d):
        return round(n / d * 100, 2) if d else 0.0

    # --- Image properties block ---
    res_top = next(iter(image_props["resolutions"]), "unknown")
    mode_top = next(iter(image_props["modes"]), "unknown")
    fmt_top = next(iter(image_props["formats"]), "unknown")
    if image_props["widths"]:
        common_w, common_h = (int(v) for v in res_top.split("x"))
    else:
        common_w = common_h = 0

    bbox_summary = {
        "total_instances": len(all_instances),
        "width": calc_stats(raw["widths"]),
        "height": calc_stats(raw["heights"]),
        "area": calc_stats(raw["areas"]),
        "aspect_ratio": calc_stats(raw["aspect_ratios"]),
        "width_px": calc_stats(raw["widths_px"]),
        "height_px": calc_stats(raw["heights_px"]),
        "area_px": calc_stats(raw["bbox_areas_px"]),
        "aspect_px": calc_stats(raw["aspect_px"]),
        "objects_per_image": calc_stats(raw["objects_per_image"]),
        "per_split": {
            split: {
                "total_instances": len(instances_by_split[split]),
                "class_counts": instances_per_split[split],
                "width_px": calc_stats([i.w * common_w for i in instances_by_split[split]]),
                "height_px": calc_stats([i.h * common_h for i in instances_by_split[split]]),
            }
            for split in SPLITS
        },
    }

    # --- Segmentation-specific geometry ---
    cls_arr = np.array(raw["class_ids"]) if raw["class_ids"] else np.array([], dtype=int)
    kind_arr = np.array(raw["kinds"]) if raw["kinds"] else np.array([], dtype=object)
    size_cat_arr = np.array(raw["size_categories"]) if raw["size_categories"] else np.array([], dtype=object)

    def class_mask(cid, polygons_only=False):
        if cls_arr.size == 0:
            return np.zeros(0, dtype=bool)
        m = cls_arr == cid
        return m & (kind_arr == "polygon") if polygons_only else m

    def sel(key, mask):
        vals = [v for v, keep in zip(raw[key], mask) if keep]
        return [v for v in vals if v is not None]

    segmentation = {
        "polygon_instances": polygon_instances,
        "box_instances": box_instances,
        "vertices": calc_stats(raw["n_vertices"]),
        "mask_area_px": calc_stats([a for a, k in zip(raw["mask_areas_px"], raw["kinds"]) if k == "polygon"]),
        "fill_ratio": calc_stats(raw["fill_ratios"]),
        "size_category_thresholds": {
            "small": f"< {SIZE_CATEGORY_THRESHOLDS[0] * 100:.2f}% of frame area",
            "medium": f"{SIZE_CATEGORY_THRESHOLDS[0] * 100:.2f}–{SIZE_CATEGORY_THRESHOLDS[1] * 100:.2f}% of frame area",
            "large": f">= {SIZE_CATEGORY_THRESHOLDS[1] * 100:.2f}% of frame area",
        },
        "per_class": {
            class_name(cid): {
                "instances": int(overall_instances[cid]),
                "polygons": int(np.count_nonzero(class_mask(cid, True))),
                "vertices": calc_stats(sel("n_vertices", class_mask(cid, True))),
                "mask_area_px": calc_stats(sel("mask_areas_px", class_mask(cid, True))),
                "fill_ratio": calc_stats(sel("fill_ratios", class_mask(cid, True))),
                "size_categories": {
                    cat: int(np.count_nonzero(class_mask(cid) & (size_cat_arr == cat)))
                    for cat in SIZE_CATEGORIES
                },
            }
            for cid in class_ids_sorted
        },
    }

    report = {
        "dataset_overview": {
            "total_images": total_images,
            "total_annotations": total_instances,
            "polygon_annotations": polygon_instances,
            "box_annotations": box_instances,
            "images_with_annotations": total_annotated,
            "background_images": total_images - total_annotated,
            "num_classes": len(CLASS_NAMES),
            "class_names": CLASS_NAMES,
            "image_format": fmt_top,
            "annotation_format": "YOLO instance segmentation (normalized polygons)",
            "original_resolution": f"{common_w} x {common_h} pixels" if common_w else "unknown",
            "training_resolution": "640 x 640 pixels (resized)",
            "aspect_ratio": _aspect_label(common_w, common_h) if common_w else "unknown",
            "color_space": _mode_color_space(mode_top),
            "dataset_type": "Instance Segmentation",
            "file_size_kb": {
                "min": round(float(np.min(image_props["sizes_kb"])), 1) if image_props["sizes_kb"] else 0.0,
                "max": round(float(np.max(image_props["sizes_kb"])), 1) if image_props["sizes_kb"] else 0.0,
                "avg": round(float(np.mean(image_props["sizes_kb"])), 1) if image_props["sizes_kb"] else 0.0,
                "median": round(float(np.median(image_props["sizes_kb"])), 1) if image_props["sizes_kb"] else 0.0,
                "total_files": image_props["total_files"],
            },
        },
        "image_properties": {
            "resolutions": image_props["resolutions"],
            "modes": image_props["modes"],
            "formats": image_props["formats"],
            "width": calc_stats(image_props["widths"]),
            "height": calc_stats(image_props["heights"]),
            "total_files": image_props["total_files"],
        },
        "splits": {
            split: {
                **split_stats[split],
                "instances": instances_per_split[split],
                "total_instances": sum(instances_per_split[split].values()),
                "image_pct": pct(split_stats[split]["images"], total_images),
                "instance_pct": pct(sum(instances_per_split[split].values()), total_instances),
            }
            for split in SPLITS
        },
        "class_distribution": {
            class_name(cid): {
                "class_id": cid,
                "count": overall_instances[cid],
                "polygons": int(np.count_nonzero(class_mask(cid, True))),
                "boxes": int(overall_instances[cid] - np.count_nonzero(class_mask(cid, True))),
                "pct": pct(overall_instances[cid], total_instances),
                "color": class_color(cid),
            }
            for cid in class_ids_sorted
        },
        "bounding_boxes": bbox_summary,
        "segmentation_geometry": segmentation,
        "validation": validation.to_dict(),
    }
    raw["image_props"] = image_props
    return report, raw


# ===== JSON OUTPUT =====

def write_json_report(report):
    output_file = DATA_DIR / "dataset_analysis_report.json"
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)
    return output_file


# ===== MARKDOWN OUTPUT =====

def _md_table(headers, rows, align=None):
    """Render a GitHub-flavoured markdown table (first column left-aligned)."""
    align = align or ["left"] + ["right"] * (len(headers) - 1)
    rule = {"left": ":---", "right": "---:", "center": ":---:"}
    lines = ["| " + " | ".join(str(h) for h in headers) + " |",
             "| " + " | ".join(rule[a] for a in align) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def generate_markdown_report(report, figures):
    """Render the same content as the HTML dashboard as a Markdown document."""
    from datetime import datetime

    ov = report["dataset_overview"]
    fs = ov["file_size_kb"]
    cd = report["class_distribution"]
    bb = report["bounding_boxes"]
    ip = report["image_properties"]
    seg = report["segmentation_geometry"]
    val = report["validation"]

    figure_paths = {p.stem: p for p in figures}
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    def stat_row(label, d, fmt="{:.2f}"):
        return [label, fmt.format(d["mean"]), fmt.format(d["std"]),
                fmt.format(d["min"]), fmt.format(d["median"]), fmt.format(d["max"])]

    parts = [
        "# Dataset Analysis Report",
        "",
        f"Instance segmentation · {ov['num_classes']} classes · generated {generated}",
        "",
        "## Key figures",
        "",
        _md_table(["Metric", "Value", "Detail"], [
            ["Images", f"{ov['total_images']:,}",
             f"{ov['images_with_annotations']:,} annotated · {ov['background_images']:,} background"],
            ["Annotations", f"{ov['total_annotations']:,}",
             f"{ov['polygon_annotations']:,} polygons · {ov['box_annotations']:,} box rows"],
            ["Classes", ov["num_classes"], ", ".join(ov["class_names"])],
            ["Instances / image", f"{bb['objects_per_image']['mean']:.1f}",
             f"median {bb['objects_per_image']['median']:.0f} · max {bb['objects_per_image']['max']:.0f}"],
            ["Mask fill ratio", f"{seg['fill_ratio']['median']:.2f}",
             f"median · {seg['vertices']['median']:.0f} vertices typical"],
            ["Resolution", ov["original_resolution"], f"{ov['image_format']} · {ov['color_space']}"],
            ["Avg file size", f"{fs['avg']:.0f} KB", f"{fs['min']:.0f}–{fs['max']:.0f} KB"],
            ["Validation issues", f"{val['total_issues']:,}",
             "clean" if val["clean"] else f"{len(val['issues'])} issue type(s)"],
        ], ["left", "right", "left"]),
        "",
        "## Dataset properties",
        "",
        _md_table(["Property", "Value"], [
            ["Dataset type", ov["dataset_type"]],
            ["Annotation format", ov["annotation_format"]],
            ["Image format", ov["image_format"]],
            ["Color space", ov["color_space"]],
            ["Resolution", ov["original_resolution"]],
            ["Aspect ratio", ov["aspect_ratio"]],
            ["Training resolution", ov["training_resolution"]],
        ], ["left", "left"]),
        "",
        "## Image properties",
        "",
        f"Pixel modes present: {', '.join(f'{m} ({c:,})' for m, c in ip['modes'].items())}",
        "",
        _md_table(["Resolution", "Images", "Share"],
                  [[res, f"{count:,}", f"{count / ip['total_files'] * 100:.1f}%"]
                   for res, count in ip["resolutions"].items()]
                  + [["**total**", f"**{ip['total_files']:,}**", "**100%**"]]),
        "",
        "## Split distribution",
        "",
        _md_table(["Split", "Images", "Img %", "Annotated", "Background", "Instances", "Inst %"],
                  [[split,
                    f"{report['splits'][split]['images']:,}",
                    f"{report['splits'][split]['image_pct']:.1f}%",
                    f"{report['splits'][split]['annotated']:,}",
                    f"{report['splits'][split]['unannotated']:,}",
                    f"{report['splits'][split]['total_instances']:,}",
                    f"{report['splits'][split]['instance_pct']:.1f}%"]
                   for split in SPLITS]
                  + [["**total**", f"**{ov['total_images']:,}**", "**100%**",
                      f"**{ov['images_with_annotations']:,}**", f"**{ov['background_images']:,}**",
                      f"**{ov['total_annotations']:,}**", "**100%**"]]),
        "",
        "## Class distribution",
        "",
        _md_table(["Class", "Instances", "Polygons", "Box rows", "Share"],
                  [[name, f"{info['count']:,}", f"{info['polygons']:,}",
                    f"{info['boxes']:,}", f"{info['pct']:.1f}%"]
                   for name, info in cd.items()]),
        "",
        "## Instance geometry",
        "",
        "Bounding boxes are derived from polygon extents and reported in pixels.",
        "",
        _md_table(["Measure", "Mean", "Std", "Min", "Median", "Max"], [
            stat_row("Width (px)", bb["width_px"], "{:.1f}"),
            stat_row("Height (px)", bb["height_px"], "{:.1f}"),
            stat_row("Bounding-box area (px²)", bb["area_px"], "{:.0f}"),
            stat_row("Aspect ratio (w/h, px)", bb["aspect_px"]),
            stat_row("Instances / image", bb["objects_per_image"], "{:.1f}"),
        ]),
        "",
        "## Segmentation geometry",
        "",
        _md_table(["Measure", "Mean", "Std", "Min", "Median", "Max"], [
            stat_row("Mask area (px²)", seg["mask_area_px"], "{:.0f}"),
            stat_row("Mask fill ratio", seg["fill_ratio"], "{:.3f}"),
            stat_row("Vertices per polygon", seg["vertices"], "{:.1f}"),
        ]),
        "",
        f"Per class (medians; size split uses {seg['size_category_thresholds']['small']} / "
        f"{seg['size_category_thresholds']['large']}):",
        "",
        _md_table(["Class", "Polygons", "Mask area (px²)", "Fill ratio", "Vertices", "S / M / L"],
                  [[name, f"{info['polygons']:,}",
                    f"{info['mask_area_px']['median']:.0f}",
                    f"{info['fill_ratio']['median']:.2f}",
                    f"{info['vertices']['median']:.0f}",
                    f"{info['size_categories']['small']:,} / "
                    f"{info['size_categories']['medium']:,} / "
                    f"{info['size_categories']['large']:,}"]
                   for name, info in seg["per_class"].items()]),
        "",
        "## Data validation",
        "",
    ]

    if val["issues"]:
        rows = []
        for issue, info in val["issues"].items():
            splits_txt = ", ".join(f"{s}: {n}" for s, n in info["by_split"].items())
            examples = "<br>".join(f"`{e}`" for e in info["examples"][:3])
            if info["truncated"]:
                examples += f"<br>… {info['count'] - len(info['examples'][:3])} more"
            rows.append([f"{ISSUE_DESCRIPTIONS.get(issue, issue)}<br>`{issue}`",
                         f"{info['count']:,}", splits_txt, examples])
        parts.append(_md_table(["Issue", "Count", "By split", "Examples"],
                               rows, ["left", "right", "left", "left"]))
    else:
        parts.append("No annotation issues found.")

    parts += ["", "## Figures", ""]
    for stem, title, caption in FIGURE_META:
        path = figure_paths.get(stem)
        if not path or not path.exists():
            continue
        rel = os.path.relpath(path, DATA_DIR)
        parts += [f"### {title}", "", caption, "", f"![{title}]({rel})", ""]

    parts += ["---", "", f"Generated by `dataset_analysis.py` · {generated}", ""]

    output_file = DATA_DIR / "dataset_analysis_report.md"
    output_file.write_text("\n".join(parts), encoding="utf-8")
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


def _style_boxplot(bp, colors):
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor("#fcfcfb")
    for element in ("whiskers", "caps"):
        for line in bp[element]:
            line.set_color("#898781")
    for line in bp["medians"]:
        line.set_color("#0b0b0b")
        line.set_linewidth(1.5)


def generate_figures(report, raw):
    """Render all figures + companion JSON into .figure/."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _apply_style()

    blue_ramp = LinearSegmentedColormap.from_list(
        "viz_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#184f95", "#0d366b"])

    cd = report["class_distribution"]
    names = list(cd.keys())
    counts = [cd[n]["count"] for n in names]
    colors = [cd[n]["color"] for n in names]
    figures = []

    cls_arr = np.array(raw["class_ids"])
    kind_arr = np.array(raw["kinds"])
    present_ids = sorted(set(raw["class_ids"]))

    def values_for(key, cid, polygons_only=False):
        m = cls_arr == cid
        if polygons_only:
            m = m & (kind_arr == "polygon")
        return np.array([v for v, keep in zip(raw[key], m) if keep and v is not None],
                        dtype=float)

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
    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.8))
    by_split_data = {}
    for i, split in enumerate(SPLITS):
        vals = [report["splits"][split]["instances"][cd[n]["class_id"]] for n in names]
        by_split_data[split] = vals
        ax.bar(x + (i - 1) * w, vals, w, label=split, color=SPLIT_PALETTE[split])
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
        b = ax.bar(SPLITS, vals, color=[SPLIT_PALETTE[s] for s in SPLITS], width=0.6)
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

    # 5. Instance geometry distributions, in pixels (small multiples of histograms)
    dims = [("widths_px", "Width (px)"), ("heights_px", "Height (px)"),
            ("bbox_areas_px", "Bounding-box area (px²)"),
            ("aspect_px", "Aspect ratio (w/h, pixel space)")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    hist_data = {}
    for ax, (key, label) in zip(axes.ravel(), dims):
        vals = np.array([v for v in raw[key] if v is not None], dtype=float)
        plot_vals = vals[vals <= np.percentile(vals, 99)] if vals.size else vals
        n, edges = np.histogram(plot_vals, bins=30)
        ax.hist(plot_vals, bins=30, color="#2a78d6", edgecolor="#fcfcfb", linewidth=0.5)
        med = float(np.median(vals)) if vals.size else 0.0
        ax.axvline(med, color="#e34948", linewidth=1.5, linestyle="--")
        ax.text(med, ax.get_ylim()[1] * 0.92, f"median {med:.2f}", color="#e34948",
                fontsize=9, ha="left" if med < np.mean(ax.get_xlim()) else "right")
        ax.set_title(label)
        ax.set_ylabel("Instances")
        ax.yaxis.grid(True)
        ax.xaxis.grid(False)
        hist_data[key] = {"bin_edges": [round(e, 4) for e in edges.tolist()],
                          "counts": n.tolist(), "median": round(med, 4),
                          "note": "histogram clipped at the 99th percentile"}
    fig.suptitle("Instance geometry distributions (pixels, from polygon extents)",
                 fontsize=13, fontweight="bold", color="#0b0b0b")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    figures.append(_save(fig, "bbox_dimension_distributions",
                         {"chart": "histogram_small_multiples", "unit": "pixels",
                          "histograms": hist_data}))
    plt.close(fig)

    # 6. Width vs height scatter, coloured by class (identity -> categorical)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    w_arr = np.array(raw["widths_px"])
    h_arr = np.array(raw["heights_px"])
    for cid in present_ids:
        m = cls_arr == cid
        ax.scatter(w_arr[m], h_arr[m], s=18, alpha=0.6, color=class_color(cid),
                   edgecolors="none", label=class_name(cid))
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("Instance width vs height by class")
    ax.legend(frameon=False, markerscale=1.5, fontsize=9)
    ax.grid(True)
    figures.append(_save(fig, "bbox_width_height_scatter",
                         {"chart": "scatter", "x": "width_px", "y": "height_px",
                          "points_by_class": {
                              class_name(cid): {
                                  "width_px": [round(float(v), 2) for v in w_arr[cls_arr == cid]],
                                  "height_px": [round(float(v), 2) for v in h_arr[cls_arr == cid]],
                              } for cid in present_ids}}))
    plt.close(fig)

    # 7. Image file-size distribution
    sizes = report["dataset_overview"]["file_size_kb"]
    all_sizes = raw["image_props"]["sizes_kb"]
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

    # 8. Mask centroid spatial heatmap (where objects sit in the frame)
    ip = report["image_properties"]
    frame_w = ip["width"]["median"] or 1
    frame_h = ip["height"]["median"] or 1
    frame_aspect = frame_h / frame_w
    ny = 12
    nx = max(int(round(ny / frame_aspect)), 12)
    xc = np.array(raw["x_centers"])
    yc = np.array(raw["y_centers"])
    heat, xedges, yedges = np.histogram2d(xc, yc, bins=[nx, ny], range=[[0, 1], [0, 1]])
    fig, ax = plt.subplots(figsize=(9, 2.6))
    im = ax.imshow(heat.T, origin="upper", extent=[0, 1, 1, 0],
                   cmap=blue_ramp, interpolation="nearest", aspect=frame_aspect)
    ax.set_xlabel("x centroid (normalized)")
    ax.set_ylabel("y centroid (normalized)")
    ax.set_title("Mask centroid spatial distribution")
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Instances", color="#52514e")
    cbar.outline.set_edgecolor("#c3c2b7")
    figures.append(_save(fig, "object_center_heatmap",
                         {"chart": "heatmap_2d", "x": "centroid_x", "y": "centroid_y",
                          "note": "grid matches the frame aspect ratio",
                          "x_edges": [round(e, 4) for e in xedges.tolist()],
                          "y_edges": [round(e, 4) for e in yedges.tolist()],
                          "counts": heat.astype(int).tolist()}))
    plt.close(fig)

    # 9. Objects-per-image distribution (annotation density)
    opi = np.array(raw["objects_per_image"])
    max_opi = int(opi.max()) if opi.size else 1
    n, _ = np.histogram(opi, bins=np.arange(0.5, max_opi + 1.5, 1))
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(range(1, max_opi + 1), n, color="#2a78d6", width=0.85)
    med = float(np.median(opi)) if opi.size else 0.0
    ax.axvline(med, color="#e34948", linewidth=1.5, linestyle="--")
    ax.text(med, ax.get_ylim()[1] * 0.92, f"median {med:.0f}", color="#e34948",
            fontsize=9, ha="left")
    ax.set_xlabel("Instances per image")
    ax.set_ylabel("Images")
    ax.set_title("Annotation density (instances per image)")
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    figures.append(_save(fig, "objects_per_image",
                         {"chart": "histogram", "objects_per_image": list(range(1, max_opi + 1)),
                          "image_counts": n.tolist(), "median": round(med, 2),
                          "mean": round(float(opi.mean()), 2) if opi.size else 0.0}))
    plt.close(fig)

    # 10. Bounding-box area by class (size profile per class)
    fig, ax = plt.subplots(figsize=(9, 5))
    # Log scale cannot show zero-area annotations; those surface in the validation figure.
    data = [values_for("bbox_areas_px", cid) for cid in present_ids]
    data = [d[d > 0] for d in data]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=True,
                    flierprops=dict(marker="o", markersize=3, markerfacecolor="#898781",
                                    markeredgecolor="none", alpha=0.4))
    _style_boxplot(bp, [class_color(cid) for cid in present_ids])
    ax.set_xticklabels([class_name(cid) for cid in present_ids], rotation=20, ha="right")
    ax.set_ylabel("Bounding-box area (px²)")
    ax.set_yscale("log")
    ax.set_title("Bounding-box area by class")
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    figures.append(_save(fig, "bbox_area_by_class",
                         {"chart": "boxplot", "measure": "bbox_area_px", "scale": "log",
                          "classes": [class_name(cid) for cid in present_ids],
                          "stats_by_class": {
                              class_name(cid): calc_stats(list(values_for("bbox_areas_px", cid)))
                              for cid in present_ids}}))
    plt.close(fig)

    # 11. Mask (polygon) area by class — the true segmentation size profile
    fig, ax = plt.subplots(figsize=(9, 5))
    data = [values_for("mask_areas_px", cid, polygons_only=True) for cid in present_ids]
    data = [d[d > 0] for d in data]  # zero-area polygons are reported by the validation figure
    keep = [i for i, d in enumerate(data) if d.size]
    if keep:
        bp = ax.boxplot([data[i] for i in keep], patch_artist=True, widths=0.6,
                        showfliers=True,
                        flierprops=dict(marker="o", markersize=3, markerfacecolor="#898781",
                                        markeredgecolor="none", alpha=0.4))
        _style_boxplot(bp, [class_color(present_ids[i]) for i in keep])
        ax.set_xticklabels([class_name(present_ids[i]) for i in keep], rotation=20, ha="right")
        ax.set_yscale("log")
    ax.set_ylabel("Mask area (px²)")
    ax.set_title("Mask (polygon) area by class")
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    figures.append(_save(fig, "mask_area_by_class",
                         {"chart": "boxplot", "measure": "mask_area_px", "scale": "log",
                          "classes": [class_name(present_ids[i]) for i in keep],
                          "stats_by_class": {
                              class_name(present_ids[i]): calc_stats(list(data[i]))
                              for i in keep}}))
    plt.close(fig)

    # 12. Mask fill ratio by class (polygon area / bbox area — annotation tightness)
    fig, ax = plt.subplots(figsize=(9, 5))
    data = [values_for("fill_ratios", cid, polygons_only=True) for cid in present_ids]
    keep = [i for i, d in enumerate(data) if d.size]
    if keep:
        bp = ax.boxplot([data[i] for i in keep], patch_artist=True, widths=0.6,
                        showfliers=True,
                        flierprops=dict(marker="o", markersize=3, markerfacecolor="#898781",
                                        markeredgecolor="none", alpha=0.4))
        _style_boxplot(bp, [class_color(present_ids[i]) for i in keep])
        ax.set_xticklabels([class_name(present_ids[i]) for i in keep], rotation=20, ha="right")
    ax.axhline(0.785, color="#898781", linewidth=1.2, linestyle=":")
    ax.text(ax.get_xlim()[1], 0.785, " ellipse-like (0.79)", color="#898781",
            fontsize=8, va="center", ha="left")
    ax.set_ylabel("Mask area / bounding-box area")
    ax.set_ylim(0, 1.05)
    ax.set_title("Mask fill ratio by class")
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    figures.append(_save(fig, "mask_fill_ratio_by_class",
                         {"chart": "boxplot", "measure": "fill_ratio",
                          "note": "1.0 = polygon fills its box; low values = thin/diagonal shapes",
                          "classes": [class_name(present_ids[i]) for i in keep],
                          "stats_by_class": {
                              class_name(present_ids[i]): calc_stats(list(data[i]))
                              for i in keep}}))
    plt.close(fig)

    # 13. Polygon vertex-count distribution (annotation detail / complexity)
    verts = np.array([v for v in raw["n_vertices"] if v], dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    if verts.size:
        upper = int(np.percentile(verts, 99))
        bins = np.arange(2.5, max(upper, 4) + 1.5, 1)
        n, edges = np.histogram(verts, bins=bins)
        ax.bar(edges[:-1] + 0.5, n, width=0.85, color="#4a3aa7")
        med = float(np.median(verts))
        ax.axvline(med, color="#e34948", linewidth=1.5, linestyle="--")
        ax.text(med, ax.get_ylim()[1] * 0.92, f"median {med:.0f}", color="#e34948",
                fontsize=9, ha="left")
    else:
        n, edges, med = np.array([]), np.array([]), 0.0
    ax.set_xlabel("Vertices per polygon (clipped at the 99th percentile)")
    ax.set_ylabel("Instances")
    ax.set_title("Polygon vertex-count distribution")
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    figures.append(_save(fig, "polygon_vertex_distribution",
                         {"chart": "histogram", "measure": "vertices",
                          "bin_edges": [float(e) for e in edges.tolist()],
                          "counts": n.tolist(),
                          "stats": calc_stats(list(verts))}))
    plt.close(fig)

    # 14. Instance size categories per class (COCO small/medium/large, relative)
    cat_colors = {"small": "#cde2fb", "medium": "#3987e5", "large": "#0d366b"}
    seg = report["segmentation_geometry"]["per_class"]
    fig, ax = plt.subplots(figsize=(9, 5))
    left = np.zeros(len(names))
    cat_data = {}
    for cat in SIZE_CATEGORIES:
        vals = np.array([seg[n_]["size_categories"][cat] for n_ in names], dtype=float)
        cat_data[cat] = vals.astype(int).tolist()
        ax.barh(names, vals, left=left, color=cat_colors[cat], label=cat, height=0.62)
        for i, (v, l_val) in enumerate(zip(vals, left)):
            if v > 0 and v / max(sum(counts), 1) > 0.01:
                ax.text(l_val + v / 2, i, f"{int(v)}", ha="center", va="center",
                        color="#ffffff" if cat != "small" else "#0b0b0b",
                        fontsize=8, fontweight="bold")
        left += vals
    ax.set_xlabel("Instances", labelpad=8)
    ax.set_title("Instance size categories by class")
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.19),
              title="Size (share of frame area)")
    fig.subplots_adjust(bottom=0.32)
    figures.append(_save(fig, "instance_size_categories",
                         {"chart": "stacked_bar", "classes": names,
                          "thresholds": report["segmentation_geometry"]["size_category_thresholds"],
                          "counts": cat_data}))
    plt.close(fig)

    # 15. Class composition across splits (100% stacked — stratification check)
    fig, ax = plt.subplots(figsize=(9, 5))
    left = np.zeros(len(names))
    comp_data = {
        split: np.array([report["splits"][split]["instances"][cd[n_]["class_id"]] for n_ in names],
                        dtype=float)
        for split in SPLITS
    }
    totals = sum(comp_data[s] for s in SPLITS)
    totals[totals == 0] = 1  # avoid div-by-zero
    for split in SPLITS:
        frac = comp_data[split] / totals * 100
        ax.barh(names, frac, left=left, color=SPLIT_PALETTE[split], label=split, height=0.62)
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

    # 16. Class co-occurrence matrix (which pole types share an image)
    k = len(names)
    matrix = np.zeros((k, k), dtype=int)
    for key, count in raw["cooccurrence"].items():
        a, b = (int(v) for v in key.split(","))
        if a < k and b < k:
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

    add_image_size_figure(figures, report, raw, plt)
    add_validation_figure(figures, report, plt)
    add_sample_montage(figures, plt)

    return figures


def add_image_size_figure(figures, report, raw, plt):
    """TAO's 'image size' graph: resolution mix plus width/height spread."""
    ip = report["image_properties"]
    resolutions = ip["resolutions"]
    if not resolutions:
        return
    top = list(resolutions.items())[:15]
    labels = [r for r, _ in top]
    values = [c for _, c in top]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    bars = ax.barh(labels[::-1], values[::-1], color="#2a78d6", height=0.6)
    ax.set_xlabel("Images")
    ax.set_title("Resolutions present")
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    for b, v in zip(bars, values[::-1]):
        ax.text(b.get_width() + max(values) * 0.01, b.get_y() + b.get_height() / 2,
                f"{v}  ({v/sum(values)*100:.1f}%)", va="center", ha="left",
                color="#52514e", fontsize=9)
    ax.set_xlim(0, max(values) * 1.25)
    # Keep bars readable when only one or two resolutions exist.
    ax.set_ylim(-0.5, max(len(labels), 3) - 0.5)

    ax = axes[1]
    pairs = Counter((w, h) for w, h in zip(raw["image_props"]["widths"],
                                           raw["image_props"]["heights"]))
    xs = [w for (w, _) in pairs]
    ys = [h for (_, h) in pairs]
    ss = [40 + 260 * (c / max(pairs.values())) for c in pairs.values()]
    ax.scatter(xs, ys, s=ss, color="#1baf7a", alpha=0.75, edgecolors="none")
    for (w, h), c in pairs.items():
        ax.annotate(f"{w}x{h} ({c})", (w, h), textcoords="offset points",
                    xytext=(10, 8), fontsize=9, color="#52514e")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("Width vs height (marker size = image count)")
    ax.set_xlim(0, max(xs) * 1.35)
    ax.set_ylim(0, max(ys) * 1.45)
    ax.grid(True)

    fig.suptitle("Image size", fontsize=13, fontweight="bold", color="#0b0b0b")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    figures.append(_save(fig, "image_size_distribution",
                         {"chart": "bar_and_scatter", "resolutions": resolutions,
                          "modes": ip["modes"], "formats": ip["formats"],
                          "width": ip["width"], "height": ip["height"]}))
    plt.close(fig)


def add_validation_figure(figures, report, plt):
    """TAO's 'invalid bounding box coordinates' graph, generalised to all issues."""
    validation = report["validation"]
    issues = validation["issues"]

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    if issues:
        labels = list(issues.keys())[::-1]
        display = [lbl.replace("_", " ") for lbl in labels]
        left = np.zeros(len(labels))
        for split in SPLITS:
            vals = np.array([issues[lbl]["by_split"].get(split, 0) for lbl in labels], dtype=float)
            ax.barh(display, vals, left=left, color=SPLIT_PALETTE[split], label=split, height=0.6)
            left += vals
        totals = left
        for i, total in enumerate(totals):
            ax.text(total + max(totals) * 0.01, i, f"{int(total)}", va="center",
                    ha="left", color="#52514e", fontsize=10)
        ax.set_xlim(0, max(totals) * 1.15)
        ax.set_xlabel("Annotations / files affected")
        ax.legend(frameon=False, title="Split")
    else:
        ax.text(0.5, 0.5, "No annotation issues found", ha="center", va="center",
                fontsize=14, color="#1baf7a", fontweight="bold", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("Annotation validation issues")
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    figures.append(_save(fig, "annotation_issues",
                         {"chart": "stacked_bar", "total_issues": validation["total_issues"],
                          "descriptions": {k: ISSUE_DESCRIPTIONS.get(k, k) for k in issues},
                          "issues": {k: {"count": v["count"], "by_split": v["by_split"]}
                                     for k, v in issues.items()}}))
    plt.close(fig)


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
    preferring frames with several/varied objects. Returns [(img, split), ...]."""
    candidates = []
    for split in SPLITS:
        split_dir = resolve_split_dir(split)
        labels_dir = split_dir / "labels"
        for img in list_images(split):
            label = labels_dir / f"{img.stem}.txt"
            if not label.exists():
                continue
            instances, _ = parse_label_file(label, split)
            classes = {i.class_id for i in instances}
            if classes:  # skip background-only frames for the montage
                candidates.append((img, split, classes))

    chosen, covered = [], set()
    # Greedy: repeatedly take the frame adding the most uncovered classes.
    remaining = candidates[:]
    while remaining and len(chosen) < n:
        remaining.sort(key=lambda c: (len(c[2] - covered), len(c[2])), reverse=True)
        img, split, classes = remaining.pop(0)
        chosen.append((img, split))
        covered |= classes
    return chosen


def add_sample_montage(figures, plt):
    """Qualitative Figure 1: real annotated frames with polygon masks drawn."""
    from matplotlib.patches import Polygon, Rectangle

    samples = _select_sample_images(n=6)
    if not samples:
        return

    n = len(samples)
    fig, axes = plt.subplots(n, 1, figsize=(10, 1.55 * n + 0.4))
    if n == 1:
        axes = [axes]

    used_classes, manifest, has_box_rows = set(), [], False
    for ax, (img_path, split) in zip(axes, samples):
        disp = _load_intensity_display(img_path)
        h, w = disp.shape[:2]
        ax.imshow(disp, cmap="gray", aspect="auto", vmin=0, vmax=1)
        label_path = resolve_split_dir(split) / "labels" / f"{img_path.stem}.txt"
        instances, _ = parse_label_file(label_path, split)
        shapes = []
        for inst in instances:
            col = class_color(inst.class_id)
            if inst.kind == "polygon":
                pts = inst.points * np.array([w, h])
                ax.add_patch(Polygon(pts, closed=True, facecolor=col, alpha=0.25,
                                     edgecolor="none"))
                ax.add_patch(Polygon(pts, closed=True, fill=False, edgecolor=col,
                                     linewidth=1.4))
                label_text = class_name(inst.class_id)
            else:  # box-only row — drawn as a dashed rectangle so it stands out
                has_box_rows = True
                x0, y0 = (inst.cx - inst.w / 2) * w, (inst.cy - inst.h / 2) * h
                ax.add_patch(Rectangle((x0, y0), inst.w * w, inst.h * h, fill=False,
                                       edgecolor=col, linewidth=1.4, linestyle="--"))
                label_text = f"{class_name(inst.class_id)} (box)"
            tx, ty = (inst.cx - inst.w / 2) * w, (inst.cy - inst.h / 2) * h
            ax.text(tx + 1, max(ty - 2, 6), label_text, fontsize=7, color="#ffffff",
                    va="bottom", bbox=dict(boxstyle="square,pad=0.12", fc=col, ec="none"))
            used_classes.add(inst.class_id)
            shapes.append({"class": class_name(inst.class_id), "kind": inst.kind,
                           "vertices": inst.n_vertices,
                           "bbox": [round(inst.cx, 4), round(inst.cy, 4),
                                    round(inst.w, 4), round(inst.h, 4)]})
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_edgecolor("#c3c2b7")
        ax.set_ylabel(img_path.stem, rotation=0, ha="right", va="center",
                      fontsize=8, color="#898781", labelpad=8)
        manifest.append({"image": img_path.name, "split": split, "instances": shapes})

    title = "Sample annotated frames (intensity, polygons coloured by class"
    axes[0].set_title(title + "; dashed = box-only row)" if has_box_rows else title + ")")
    # Shared class legend
    handles = [Rectangle((0, 0), 1, 1, fc=class_color(c), ec="none")
               for c in sorted(used_classes)]
    labels = [class_name(c) for c in sorted(used_classes)]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    figures.append(_save(fig, "sample_annotated_frames",
                         {"chart": "image_montage",
                          "note": "qualitative examples; polygons in YOLO-normalised coords",
                          "samples": manifest}))
    plt.close(fig)


# ===== HTML REPORT =====

# Figure display order + human-readable titles/captions for the HTML report.
FIGURE_META = [
    ("sample_annotated_frames", "Sample annotated frames",
     "Qualitative examples with ground-truth polygons coloured by class."),
    ("class_distribution_overall", "Class distribution (all splits)",
     "Instance count per class across the whole dataset."),
    ("class_distribution_by_split", "Class distribution by split",
     "Per-class instance counts broken down by train / validation / test."),
    ("split_distribution", "Train / validation / test split",
     "Share of images and instances in each split."),
    ("annotation_coverage", "Annotation coverage per split",
     "Annotated vs. background (empty-label) images in each split."),
    ("class_split_composition", "Class composition across splits",
     "For each class, how its instances are stratified across splits."),
    ("objects_per_image", "Annotation density",
     "Distribution of the number of instances per annotated image."),
    ("bbox_dimension_distributions", "Instance geometry",
     "Width, height, area and aspect ratio in pixels, derived from polygon extents."),
    ("bbox_width_height_scatter", "Width vs. height by class",
     "Instance width against height in pixels, coloured by class."),
    ("bbox_area_by_class", "Bounding-box area by class",
     "Size profile of each class's bounding boxes (log scale)."),
    ("mask_area_by_class", "Mask area by class",
     "True polygon area per class — the segmentation size profile (log scale)."),
    ("instance_size_categories", "Instance size categories",
     "Small / medium / large instances per class, using COCO thresholds scaled to the frame."),
    ("mask_fill_ratio_by_class", "Mask fill ratio",
     "Polygon area divided by bounding-box area — how tightly masks fit their boxes."),
    ("polygon_vertex_distribution", "Polygon vertex counts",
     "How many vertices annotators used per mask; a proxy for annotation detail."),
    ("object_center_heatmap", "Mask centroid spatial distribution",
     "Where object centroids fall within the frame."),
    ("class_cooccurrence", "Class co-occurrence",
     "How often pairs of classes appear together in the same image."),
    ("image_size_distribution", "Image size",
     "Resolutions present in the dataset and their image counts."),
    ("annotation_issues", "Annotation validation issues",
     "Invalid, malformed or suspect annotations found, by type and split."),
    ("image_file_size_distribution", "Image file-size distribution",
     "Distribution of image file sizes on disk (KB)."),
]


# ===== PAPER FIGURES =====
# Standalone, print-sized PDFs for the manuscript. Each figure is written on its
# own so it can be placed independently; nothing here is collaged. The screen
# figures above stay as they are — this is a separate, smaller set.

PAPER_CLASS_LABEL = {
    "fence_pole": "Fence", "gantry_sign_pole": "Gantry", "light_pole": "Light",
    "power_pole": "Power", "traffic_pole": "Traffic",
}
# Marker shape carries identity alongside hue, so the classes stay separable in
# greyscale print and for readers who cannot use the colour channel.
PAPER_CLASS_MARKER = {
    "fence_pole": "s", "gantry_sign_pole": "D", "light_pole": "o",
    "power_pole": "^", "traffic_pole": "v",
}
INK, INK2, MUTED, HAIRLINE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def _apply_paper_style():
    """Print styling: serif to match IEEEtran, no grid, embeddable fonts."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7.5,
        "axes.titleweight": "normal",
        "axes.titlecolor": INK,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.5,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save_paper(fig, name):
    """Write one standalone paper figure as PDF (vector, for LaTeX)."""
    path = PAPER_FIGURE_DIR / f"{name}.pdf"
    fig.savefig(path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def _polygon_mask(raw):
    """Boolean mask selecting polygon instances; box rows have no mask geometry."""
    return [k == "polygon" for k in raw["kinds"]]


def _frame_area_px(report):
    """Pixel area of the dominant frame resolution, e.g. '1024x128' -> 131072."""
    res = next(iter(report["image_properties"]["resolutions"]), None)
    if not res or "x" not in res:
        return 1
    w, h = res.split("x")
    return int(w) * int(h)


def paper_fig_instance_size(report, raw, plt):
    """Mask-area distribution with the COCO small/medium/large boundaries."""
    import numpy as np

    keep = _polygon_mask(raw)
    areas = np.array([a for a, k in zip(raw["mask_areas_px"], keep) if k])
    frame = _frame_area_px(report)
    small_t, med_t = (t * frame for t in SIZE_CATEGORY_THRESHOLDS)

    fig, ax = plt.subplots(figsize=(COL_W, 1.85))
    fig.subplots_adjust(left=0.145, right=0.98, bottom=0.235, top=0.94)
    bins = np.logspace(np.log10(max(areas.min(), 0.5)), np.log10(areas.max()), 34)
    ax.hist(areas, bins=bins, color="#2a78d6", edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    for x in (small_t, med_t):
        ax.axvline(x, color=INK, lw=0.7, ls=(0, (3, 2)))

    n = len(areas)
    shares = {c: sum(1 for a in areas if size_category(a / frame) == c) / n
              for c in SIZE_CATEGORIES}
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.28)
    for xpos, cat, label in ((small_t / 25, "small", "small"),
                             (small_t * 2.8, "medium", "medium")):
        ax.text(xpos, ymax * 1.24, f"{label}\n{shares[cat] * 100:.0f}%",
                ha="center", va="top", fontsize=5.8, color=INK2, linespacing=1.15)
    ax.set_xlim(bins[0] * 0.8, bins[-1] * 2.2)
    ax.set_xlabel("Mask area (px, log)")
    ax.set_ylabel("Instances")
    return _save_paper(fig, "instance_size")


def paper_fig_class_cooccurrence(raw, plt):
    """Lower-triangular class co-occurrence: how often two classes share a frame.

    The diagonal is omitted on purpose — it is the per-class image count, which
    Table I already carries, and at 466 it would swamp the colour ramp and leave
    every off-diagonal cell white. Classes are ordered by instance frequency so
    the strongest pairs sit top-left.
    """
    import numpy as np
    import matplotlib as mpl

    freq = Counter(raw["class_ids"])
    order = sorted(range(len(CLASS_NAMES)), key=lambda c: -freq.get(c, 0))
    pairs = raw["cooccurrence"]

    def pair_count(a, b):
        return pairs.get(f"{min(a, b)},{max(a, b)}", 0)

    n = len(order)
    rows, cols = order[1:], order[:-1]           # drop the empty first row / last col
    grid = np.full((n - 1, n - 1), np.nan)
    for i, ra in enumerate(rows):
        for j, cb in enumerate(cols):
            if j <= i:
                grid[i, j] = pair_count(ra, cb)

    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
    fig.subplots_adjust(left=0.215, right=0.99, bottom=0.02, top=0.86)
    finite = grid[np.isfinite(grid)]
    # pcolormesh (not imshow) so empty cells are genuinely absent and the white
    # cell separation is a drawn edge rather than an interpolated seam.
    edges = np.arange(n) - 0.5
    im = ax.pcolormesh(edges, edges, np.ma.masked_invalid(grid), cmap="Blues",
                       norm=mpl.colors.LogNorm(vmin=max(finite.min(), 1),
                                               vmax=finite.max()),
                       edgecolors="white", linewidth=1.2)
    ax.invert_yaxis()
    ax.set_aspect("equal")

    # Counts are labelled on every cell; the ramp is reinforcement, not the only
    # channel. Ink flips to white once the cell is dark enough to need it.
    hi = np.log10(max(finite.max(), 2))
    for i in range(n - 1):
        for j in range(n - 1):
            if not np.isfinite(grid[i, j]):
                continue
            v = int(grid[i, j])
            dark = np.log10(max(v, 1)) / hi > 0.70  # every label clears 4.5:1
            ax.text(j, i, f"{v}", ha="center", va="center", fontsize=6.5,
                    color="white" if dark else INK)

    ax.set_xticks(range(n - 1))
    ax.set_yticks(range(n - 1))
    ax.set_xticklabels([PAPER_CLASS_LABEL.get(CLASS_NAMES[c], CLASS_NAMES[c]) for c in cols])
    ax.set_yticklabels([PAPER_CLASS_LABEL.get(CLASS_NAMES[c], CLASS_NAMES[c]) for c in rows])
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return _save_paper(fig, "class_cooccurrence")


def paper_fig_instance_shape(raw, plt):
    """Width vs height density, with each class median marked and labelled."""
    import numpy as np
    import matplotlib as mpl

    keep = _polygon_mask(raw)
    w = np.array([v for v, k in zip(raw["widths_px"], keep) if k])
    h = np.array([v for v, k in zip(raw["heights_px"], keep) if k])
    cls = [c for c, k in zip(raw["class_ids"], keep) if k]

    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
    fig.subplots_adjust(left=0.185, right=0.975, bottom=0.165, top=0.97)
    ax.hexbin(w, h, xscale="log", yscale="log", gridsize=26, mincnt=1,
              cmap="Greys", norm=mpl.colors.LogNorm(vmax=90), linewidths=0, alpha=0.85)
    for ratio, label, at in ((1, "h = w", 105), (10, "h = 10w", 9.0)):
        xs = np.array([0.7, 400])
        ax.plot(xs, ratio * xs, color=INK, lw=0.6, ls=(0, (3, 2)), zorder=3)
        ax.text(at, at * ratio * 1.30, label, fontsize=5.8, color=INK2,
                rotation=32, ha="center")

    # (x-factor, y-factor, ha) offsets keep the direct labels off their markers
    # and off each other; Power and Light have near-identical medians.
    offsets = {"fence_pole": (0.55, 1.0, "right"),
               "gantry_sign_pole": (1.0, 0.60, "center"),
               "light_pole": (1.85, 1.18, "left"),
               "power_pole": (0.74, 1.06, "right"),
               "traffic_pole": (1.0, 0.58, "center")}
    for cid, name in enumerate(CLASS_NAMES):
        xs = [x for x, c in zip(w, cls) if c == cid]
        ys = [y for y, c in zip(h, cls) if c == cid]
        if not xs:
            continue
        mx, my = float(np.median(xs)), float(np.median(ys))
        ax.plot(mx, my, PAPER_CLASS_MARKER.get(name, "o"), color=class_color(cid),
                ms=5, mec="white", mew=0.7, zorder=5)
        dx, dy, ha = offsets.get(name, (1.0, 1.4, "center"))
        ax.annotate(PAPER_CLASS_LABEL.get(name, name), (mx * dx, my * dy),
                    color=INK, fontsize=6, ha=ha, va="center", zorder=6)

    ax.set_xlim(0.85, 450)
    ax.set_ylim(0.85, 200)
    ax.set_xlabel("Width (px, log)")
    ax.set_ylabel("Height (px, log)")
    return _save_paper(fig, "instance_shape")


def paper_fig_centroid_density(raw, plt):
    """Where instances sit in the range image: azimuth vs elevation."""
    import numpy as np
    import matplotlib as mpl

    keep = _polygon_mask(raw)
    x = np.array([v for v, k in zip(raw["x_centers"], keep) if k])
    y = np.array([v for v, k in zip(raw["y_centers"], keep) if k])

    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
    # Fill the panel: the map is the smallest element in the 1x3 row otherwise.
    # The cost is more vertical exaggeration, which the caption states.
    fig.subplots_adjust(left=0.155, right=0.85, bottom=0.205, top=0.965)
    counts, _, _ = np.histogram2d(x, y, bins=[48, 16], range=[[0, 1], [0, 1]])
    im = ax.imshow(counts.T, origin="upper", extent=[0, 1, 1, 0], aspect="auto",
                   cmap="Blues", interpolation="nearest")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("Azimuth (normalised $u$)")
    ax.set_ylabel("Elevation ($v$)")
    cb = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.03)
    cb.set_label("Instances", fontsize=6.5, labelpad=1.5)
    cb.ax.tick_params(labelsize=6, length=1.8)
    cb.locator = mpl.ticker.MaxNLocator(nbins=4, integer=True)
    cb.update_ticks()
    cb.outline.set_linewidth(0.4)
    return _save_paper(fig, "centroid_density")


def paper_fig_sample_frames(plt, n=3):
    """One standalone figure per annotated frame, full page width.

    Polygons only, no in-image text: identity comes from the legend on the last
    frame, so the frames themselves stay readable at print size.
    """
    import numpy as np
    from matplotlib.patches import Polygon
    from matplotlib.lines import Line2D
    import matplotlib.patheffects as pe

    samples = _select_sample_images(n=n)
    paths = []
    for idx, (img_path, split) in enumerate(samples, start=1):
        display = _load_intensity_display(img_path)
        if display is None:
            continue
        img_h, img_w = display.shape[:2]
        fig, ax = plt.subplots(figsize=(FULL_W, FULL_W * img_h / img_w * 1.06))
        fig.subplots_adjust(left=0.002, right=0.998, bottom=0.002, top=0.998)
        ax.imshow(display, cmap="gray", vmin=0, vmax=1, aspect="auto")

        label = resolve_split_dir(split) / "labels" / f"{img_path.stem}.txt"
        instances, _ = parse_label_file(label, split)
        for inst in instances:
            if inst.kind != "polygon" or inst.n_vertices < 3:
                continue
            xy = inst.points * np.array([img_w, img_h])
            ax.add_patch(Polygon(
                xy, closed=True, fill=False, edgecolor=class_color(inst.class_id),
                lw=1.0, joinstyle="round",
                path_effects=[pe.Stroke(linewidth=2.0, foreground="white", alpha=0.85),
                              pe.Normal()]))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(HAIRLINE)
            spine.set_linewidth(0.5)
        paths.append(_save_paper(fig, f"sample_frame_{idx}"))

    # The legend ships as its own file so it can be placed once beneath whatever
    # sequence of frames the manuscript ends up using.
    fig, ax = plt.subplots(figsize=(FULL_W, 0.22))
    ax.axis("off")
    handles = [Line2D([], [], color=class_color(cid), lw=1.6,
                      marker=PAPER_CLASS_MARKER.get(name, "o"), ms=3.6,
                      mec="white", mew=0.5, label=PAPER_CLASS_LABEL.get(name, name))
               for cid, name in enumerate(CLASS_NAMES)]
    ax.legend(handles=handles, loc="center", ncol=len(handles), frameon=False,
              handlelength=1.5, columnspacing=1.6)
    paths.append(_save_paper(fig, "sample_frames_legend"))
    return paths


def generate_paper_figures(report, raw):
    """Render the standalone manuscript figures into PAPER_FIGURE_DIR."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _apply_paper_style()
    try:
        paths = [
            paper_fig_class_cooccurrence(raw, plt),
            paper_fig_instance_shape(raw, plt),
            paper_fig_centroid_density(raw, plt),
        ]
        paths.extend(paper_fig_sample_frames(plt))
    finally:
        _apply_style()  # restore the screen style for the HTML dashboard
    return paths


def _img_data_uri(png_path):
    """Base64-encode a PNG so it can be inlined into a self-contained HTML file."""
    import base64

    with open(png_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _esc(value):
    from html import escape

    return escape(str(value))


def generate_html_report(report, figures):
    """Render a single self-contained HTML dashboard from the report + figures."""
    ov = report["dataset_overview"]
    fs = ov["file_size_kb"]
    cd = report["class_distribution"]
    bb = report["bounding_boxes"]
    ip = report["image_properties"]
    seg = report["segmentation_geometry"]
    val = report["validation"]
    from datetime import datetime

    figure_paths = {p.stem: p for p in figures}

    def stat_card(label, value, sub=""):
        sub_html = f'<div class="card-sub">{_esc(sub)}</div>' if sub else ""
        return (f'<div class="card"><div class="card-value">{_esc(value)}</div>'
                f'<div class="card-label">{_esc(label)}</div>{sub_html}</div>')

    # --- Overview cards ---
    cards = "".join([
        stat_card("Images", f"{ov['total_images']:,}",
                  f"{ov['images_with_annotations']:,} annotated · {ov['background_images']} background"),
        stat_card("Annotations", f"{ov['total_annotations']:,}",
                  f"{ov['polygon_annotations']:,} polygons · {ov['box_annotations']} box rows"),
        stat_card("Classes", ov["num_classes"], ", ".join(ov["class_names"][:2]) + " …"),
        stat_card("Instances / image", f"{bb['objects_per_image']['mean']:.1f}",
                  f"median {bb['objects_per_image']['median']:.0f} · max {bb['objects_per_image']['max']:.0f}"),
        stat_card("Mask fill ratio", f"{seg['fill_ratio']['median']:.2f}",
                  f"median · {seg['vertices']['median']:.0f} vertices typical"),
        stat_card("Resolution", ov["original_resolution"], f"{ov['image_format']} · {ov['color_space']}"),
        stat_card("Avg file size", f"{fs['avg']:.0f} KB", f"{fs['min']:.0f}–{fs['max']:.0f} KB"),
        stat_card("Validation issues", f"{val['total_issues']:,}",
                  "clean" if val["clean"] else f"{len(val['issues'])} issue type(s)"),
    ])

    # --- Metadata list ---
    meta_rows = "".join(
        f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>" for k, v in [
            ("Dataset type", ov["dataset_type"]),
            ("Annotation format", ov["annotation_format"]),
            ("Image format", ov["image_format"]),
            ("Color space", ov["color_space"]),
            ("Resolution", ov["original_resolution"]),
            ("Aspect ratio", ov["aspect_ratio"]),
            ("Training resolution", ov["training_resolution"]),
        ])

    # --- Image properties ---
    img_rows = "".join(
        f"<tr><td class='name'>{_esc(res)}</td><td>{count:,}</td>"
        f"<td>{count / ip['total_files'] * 100:.1f}%</td></tr>"
        for res, count in ip["resolutions"].items())
    img_rows += (
        f"<tr class='total'><td class='name'>total</td>"
        f"<td>{ip['total_files']:,}</td><td>100%</td></tr>")
    mode_summary = ", ".join(f"{m} ({c:,})" for m, c in ip["modes"].items())

    # --- Split table ---
    split_rows = ""
    for split in SPLITS:
        s = report["splits"][split]
        split_rows += (
            f"<tr><td class='name'>{_esc(split)}</td>"
            f"<td>{s['images']:,}</td><td>{s['image_pct']:.1f}%</td>"
            f"<td>{s['annotated']:,}</td><td>{s['unannotated']:,}</td>"
            f"<td>{s['total_instances']:,}</td><td>{s['instance_pct']:.1f}%</td></tr>")
    split_rows += (
        f"<tr class='total'><td class='name'>total</td>"
        f"<td>{ov['total_images']:,}</td><td>100%</td>"
        f"<td>{ov['images_with_annotations']:,}</td><td>{ov['background_images']:,}</td>"
        f"<td>{ov['total_annotations']:,}</td><td>100%</td></tr>")

    # --- Class distribution table (with inline share bars) ---
    class_rows = ""
    for name, info in cd.items():
        pct = info["pct"]
        class_rows += (
            f"<tr><td class='name'>"
            f"<span class='swatch' style='background:{_esc(info['color'])}'></span>{_esc(name)}</td>"
            f"<td>{info['count']:,}</td><td>{info['polygons']:,}</td><td>{info['boxes']:,}</td>"
            f"<td class='bar-cell'><div class='bar-track'>"
            f"<div class='bar-fill' style='width:{pct:.1f}%;background:{_esc(info['color'])}'></div></div>"
            f"<span class='bar-pct'>{pct:.1f}%</span></td></tr>")

    # --- Geometry summary table ---
    def stat_row(label, d, fmt="{:.2f}"):
        return (f"<tr><td class='name'>{_esc(label)}</td>"
                f"<td>{fmt.format(d['mean'])}</td><td>{fmt.format(d['std'])}</td>"
                f"<td>{fmt.format(d['min'])}</td><td>{fmt.format(d['median'])}</td>"
                f"<td>{fmt.format(d['max'])}</td></tr>")
    bb_rows = "".join([
        stat_row("Width (px)", bb["width_px"], "{:.1f}"),
        stat_row("Height (px)", bb["height_px"], "{:.1f}"),
        stat_row("Bounding-box area (px²)", bb["area_px"], "{:.0f}"),
        stat_row("Aspect ratio (w/h, px)", bb["aspect_px"]),
        stat_row("Instances / image", bb["objects_per_image"], "{:.1f}"),
    ])
    seg_rows = "".join([
        stat_row("Mask area (px²)", seg["mask_area_px"], "{:.0f}"),
        stat_row("Mask fill ratio", seg["fill_ratio"], "{:.3f}"),
        stat_row("Vertices per polygon", seg["vertices"], "{:.1f}"),
    ])

    # --- Per-class segmentation geometry ---
    seg_class_rows = ""
    for name, info in seg["per_class"].items():
        cats = info["size_categories"]
        seg_class_rows += (
            f"<tr><td class='name'>"
            f"<span class='swatch' style='background:{_esc(cd[name]['color'])}'></span>{_esc(name)}</td>"
            f"<td>{info['polygons']:,}</td>"
            f"<td>{info['mask_area_px']['median']:.0f}</td>"
            f"<td>{info['fill_ratio']['median']:.2f}</td>"
            f"<td>{info['vertices']['median']:.0f}</td>"
            f"<td>{cats['small']:,} / {cats['medium']:,} / {cats['large']:,}</td></tr>")

    # --- Validation table ---
    if val["issues"]:
        val_rows = ""
        for issue, info in val["issues"].items():
            splits_txt = ", ".join(f"{s}: {n}" for s, n in info["by_split"].items())
            examples = "<br>".join(_esc(e) for e in info["examples"][:3])
            if info["truncated"]:
                examples += f"<br><span class='faint'>… {info['count'] - len(info['examples'][:3])} more</span>"
            val_rows += (
                f"<tr><td class='name'>{_esc(ISSUE_DESCRIPTIONS.get(issue, issue))}"
                f"<div class='faint'>{_esc(issue)}</div></td>"
                f"<td class='bad'>{info['count']:,}</td>"
                f"<td class='name'>{_esc(splits_txt)}</td>"
                f"<td class='name examples'>{examples}</td></tr>")
        val_block = (
            "<table><thead><tr><th class='name'>Issue</th><th>Count</th>"
            "<th class='name'>By split</th><th class='name'>Examples</th></tr></thead>"
            f"<tbody>{val_rows}</tbody></table>")
    else:
        val_block = "<div class='ok-box'>No annotation issues found.</div>"

    # --- Figures ---
    fig_html = ""
    for stem, title, caption in FIGURE_META:
        path = figure_paths.get(stem)
        if not path or not path.exists():
            continue
        fig_html += (
            f"<figure class='fig'><figcaption><h3>{_esc(title)}</h3>"
            f"<p>{_esc(caption)}</p></figcaption>"
            f"<img src='{_img_data_uri(path)}' alt='{_esc(title)}' loading='lazy'></figure>")

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dataset Analysis Report</title>
<style>
  :root {{
    --bg:#fcfcfb; --panel:#ffffff; --ink:#0b0b0b; --muted:#52514e; --faint:#898781;
    --line:#e1e0d9; --edge:#c3c2b7; --accent:#2a78d6; --bad:#e34948; --good:#1baf7a;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"DejaVu Sans",sans-serif;
    line-height:1.5; }}
  .wrap {{ max-width:1100px; margin:0 auto; padding:40px 24px 80px; }}
  header h1 {{ font-size:28px; margin:0 0 4px; letter-spacing:-.01em; }}
  header .sub {{ color:var(--faint); font-size:14px; margin-bottom:32px; }}
  h2 {{ font-size:18px; margin:44px 0 14px; padding-bottom:8px;
    border-bottom:1px solid var(--line); }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
    gap:14px; margin-bottom:8px; }}
  .card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px;
    padding:16px 18px; }}
  .card-value {{ font-size:24px; font-weight:700; letter-spacing:-.01em; }}
  .card-label {{ color:var(--muted); font-size:13px; margin-top:2px; }}
  .card-sub {{ color:var(--faint); font-size:11.5px; margin-top:6px; }}
  table {{ width:100%; border-collapse:collapse; background:var(--panel);
    border:1px solid var(--line); border-radius:10px; overflow:hidden; font-size:13.5px; }}
  th,td {{ padding:9px 14px; text-align:right; border-bottom:1px solid var(--line); }}
  thead th {{ background:#f4f3ee; color:var(--muted); font-weight:600; font-size:12px;
    text-transform:uppercase; letter-spacing:.03em; }}
  td.name, th.name {{ text-align:left; }}
  td.examples {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:11.5px;
    color:var(--muted); }}
  td.bad {{ color:var(--bad); font-weight:700; }}
  .faint {{ color:var(--faint); font-size:11.5px; }}
  tr:last-child td {{ border-bottom:none; }}
  tr.total td {{ font-weight:700; background:#f7f6f1; }}
  .meta-table th {{ text-align:left; color:var(--muted); font-weight:600; width:40%; }}
  .ok-box {{ background:var(--panel); border:1px solid var(--line); border-left:4px solid var(--good);
    border-radius:10px; padding:14px 18px; color:var(--muted); }}
  .note {{ color:var(--faint); font-size:12.5px; margin:-6px 0 14px; }}
  .swatch {{ display:inline-block; width:11px; height:11px; border-radius:3px;
    margin-right:8px; vertical-align:middle; }}
  .bar-cell {{ display:flex; align-items:center; gap:10px; text-align:left; }}
  .bar-track {{ flex:1; height:8px; background:var(--line); border-radius:5px; overflow:hidden;
    min-width:80px; }}
  .bar-fill {{ height:100%; border-radius:5px; }}
  .bar-pct {{ color:var(--muted); font-size:12px; min-width:44px; }}
  .figs {{ display:flex; flex-direction:column; gap:34px; }}
  figure.fig {{ margin:0; background:var(--panel); border:1px solid var(--line);
    border-radius:12px; padding:18px 20px 22px; }}
  figure.fig figcaption h3 {{ margin:0 0 2px; font-size:16px; }}
  figure.fig figcaption p {{ margin:0 0 14px; color:var(--faint); font-size:13px; }}
  figure.fig img {{ display:block; max-width:100%; height:auto; margin:0 auto; }}
  footer {{ margin-top:56px; color:var(--faint); font-size:12px; text-align:center; }}
  @media (max-width:640px) {{ th,td {{ padding:7px 9px; }} .wrap {{ padding:24px 14px 60px; }} }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Dataset Analysis Report</h1>
    <div class="sub">Instance segmentation · {ov['num_classes']} classes · generated {generated}</div>
  </header>

  <section class="cards">{cards}</section>

  <h2>Dataset properties</h2>
  <table class="meta-table"><tbody>{meta_rows}</tbody></table>

  <h2>Image properties</h2>
  <p class="note">Pixel modes present: {_esc(mode_summary)}</p>
  <table>
    <thead><tr><th class="name">Resolution</th><th>Images</th><th>Share</th></tr></thead>
    <tbody>{img_rows}</tbody>
  </table>

  <h2>Split distribution</h2>
  <table>
    <thead><tr><th class="name">Split</th><th>Images</th><th>Img %</th>
      <th>Annotated</th><th>Background</th><th>Instances</th><th>Inst %</th></tr></thead>
    <tbody>{split_rows}</tbody>
  </table>

  <h2>Class distribution</h2>
  <table>
    <thead><tr><th class="name">Class</th><th>Instances</th><th>Polygons</th><th>Box rows</th>
      <th class="name">Share</th></tr></thead>
    <tbody>{class_rows}</tbody>
  </table>

  <h2>Instance geometry</h2>
  <p class="note">Bounding boxes are derived from polygon extents and reported in pixels.</p>
  <table>
    <thead><tr><th class="name">Measure</th><th>Mean</th><th>Std</th>
      <th>Min</th><th>Median</th><th>Max</th></tr></thead>
    <tbody>{bb_rows}</tbody>
  </table>

  <h2>Segmentation geometry</h2>
  <table>
    <thead><tr><th class="name">Measure</th><th>Mean</th><th>Std</th>
      <th>Min</th><th>Median</th><th>Max</th></tr></thead>
    <tbody>{seg_rows}</tbody>
  </table>
  <p class="note">Per class (medians; size split uses {_esc(seg['size_category_thresholds']['small'])} /
    {_esc(seg['size_category_thresholds']['large'])}):</p>
  <table>
    <thead><tr><th class="name">Class</th><th>Polygons</th><th>Mask area (px²)</th>
      <th>Fill ratio</th><th>Vertices</th><th>S / M / L</th></tr></thead>
    <tbody>{seg_class_rows}</tbody>
  </table>

  <h2>Data validation</h2>
  {val_block}

  <h2>Figures</h2>
  <section class="figs">{fig_html}</section>

  <footer>Generated by dataset_analysis.py · {generated}</footer>
</div>
</body>
</html>"""

    output_file = DATA_DIR / "dataset_analysis_report.html"
    output_file.write_text(html, encoding="utf-8")
    return output_file


def main():
    report, raw = build_report()
    json_path = write_json_report(report)
    figures = generate_figures(report, raw)
    paper_figures = generate_paper_figures(report, raw)
    html_path = generate_html_report(report, figures)
    md_path = generate_markdown_report(report, figures)

    val = report["validation"]
    print(f"JSON report saved to: {json_path}")
    print(f"HTML report saved to: {html_path}")
    print(f"Markdown report saved to: {md_path}")
    print(f"Generated {len(figures)} figures in: {FIGURE_DIR}")
    for p in figures:
        print(f"  - {p.name}  (+ {p.stem}.json)")
    print(f"Generated {len(paper_figures)} paper figures in: {PAPER_FIGURE_DIR}")
    for p in paper_figures:
        print(f"  - {p.name}")
    if val["clean"]:
        print("\nValidation: no annotation issues found.")
    else:
        print(f"\nValidation: {val['total_issues']} issue(s) across "
              f"{len(val['issues'])} type(s):")
        for issue, info in val["issues"].items():
            print(f"  - {issue}: {info['count']}  ({ISSUE_DESCRIPTIONS.get(issue, '')})")


if __name__ == "__main__":
    main()
