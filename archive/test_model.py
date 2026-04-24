import json
from pathlib import Path
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

# BGR palette, one colour per class
_PALETTE_BGR = [
    (255, 56,  56),   # deer_fence_pole  — red
    (255, 157, 31),   # gantry_sign_pole — orange
    (56,  220, 56),   # light_pole       — green
    (56,  56,  255),  # power_pole       — blue
    (200, 56,  255),  # traffic_pole     — purple
]

_HERE = Path(__file__).parent

MODEL_PATH = _HERE / "best.pt"
IMAGE_DIR = _HERE / "Knuutilanranta"
OUTPUT_JSON = _HERE / "results/detections.json"
OUTPUT_PLOTS = _HERE / "results/plots"
OUTPUT_ANNOTATED = _HERE / "results/annotated"
CONF_THRESHOLD = 0.2

CLASS_NAMES = ["deer_fence_pole", "gantry_sign_pole", "light_pole", "power_pole", "traffic_pole"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def _annotate_and_save(img_path: Path, r) -> str:
    """Composite segmentation masks + boxes + labels onto the image and save it."""
    orig = cv2.imread(str(img_path))
    overlay = orig.copy()

    cls_ids = r.boxes.cls.tolist()
    confs   = r.boxes.conf.tolist()
    boxes   = r.boxes.xyxy.tolist()

    # fill masks
    if r.masks is not None:
        for mask_t, cls_id in zip(r.masks.data, cls_ids):
            mask = mask_t.cpu().numpy() > 0.5
            overlay[mask] = _PALETTE_BGR[int(cls_id)]

    result = cv2.addWeighted(orig, 0.4, overlay, 0.6, 0)

    # draw boxes and labels
    for (x1, y1, x2, y2), cls_id, conf in zip(boxes, cls_ids, confs):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        color = _PALETTE_BGR[int(cls_id)]
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES[int(cls_id)]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(result, (x1, max(y1 - th - 6, 0)), (x1 + tw + 4, max(y1, th + 6)), color, -1)
        cv2.putText(result, label, (x1 + 2, max(y1 - 3, th + 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    out_path = Path(OUTPUT_ANNOTATED) / img_path.name
    cv2.imwrite(str(out_path), result)
    return str(out_path)


def run_inference():
    model = YOLO(MODEL_PATH)
    image_paths = sorted(Path(IMAGE_DIR).glob("*.png")) + sorted(Path(IMAGE_DIR).glob("*.jpg"))
    Path(OUTPUT_ANNOTATED).mkdir(parents=True, exist_ok=True)

    results_list = []
    for img_path in image_paths:
        preds = model.predict(str(img_path), conf=CONF_THRESHOLD, verbose=True, retina_masks=True)
        detections = []
        annotated_path = None

        for r in preds:
            if r.boxes is not None:
                for cls_id, conf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                    detections.append({
                        "class_id": int(cls_id),
                        "class_name": CLASS_NAMES[int(cls_id)],
                        "confidence": round(float(conf), 4),
                    })
                if detections:
                    annotated_path = _annotate_and_save(img_path, r)

        results_list.append({
            "image": img_path.name,
            "no_detection": len(detections) == 0,
            "annotated_image": annotated_path,
            "detections": detections,
        })

    return results_list


def save_json(results, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved detections -> {path}")


def _build_stats(results):
    """Pre-compute all stats used across multiple plots."""
    class_confs = {name: [] for name in CLASS_NAMES}
    # per-image counts: {image: {class: count}}
    per_image_class_counts = []
    dets_per_image = []

    for entry in results:
        counts_this = Counter(d["class_name"] for d in entry["detections"])
        per_image_class_counts.append(counts_this)
        dets_per_image.append(len(entry["detections"]))
        for det in entry["detections"]:
            class_confs[det["class_name"]].append(det["confidence"])

    no_det_count = sum(1 for e in results if e["no_detection"])
    return class_confs, per_image_class_counts, dets_per_image, no_det_count


def plot_instance_counts(class_confs, no_det_count, total_images, output_dir):
    """Bar chart: detection count per class + no-detection images."""
    labels = CLASS_NAMES + ["no_detection"]
    counts = [len(class_confs[c]) for c in CLASS_NAMES] + [no_det_count]
    bar_colors = COLORS + ["#b0b0b0"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, counts, color=bar_colors, edgecolor="white")
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_title(f"Detection count per class  (total images: {total_images})")
    ax.set_ylabel("Count")
    ax.set_xlabel("Class")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path = f"{output_dir}/instance_counts.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {path}")


def plot_confidence_histograms(class_confs, output_dir):
    """Per-class confidence distribution."""
    present = [c for c in CLASS_NAMES if class_confs[c]]
    if not present:
        print("No detections — skipping confidence histograms.")
        return

    ncols = min(3, len(present))
    nrows = (len(present) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, cls_name in enumerate(present):
        confs = class_confs[cls_name]
        color = COLORS[CLASS_NAMES.index(cls_name)]
        ax = axes[i]
        ax.hist(confs, bins=20, range=(0, 1), color=color, edgecolor="white", alpha=0.85)
        mean_c = np.mean(confs)
        ax.axvline(mean_c, color="black", linestyle="--", linewidth=1.2, label=f"mean={mean_c:.2f}")
        ax.set_title(f"{cls_name}  (n={len(confs)})", fontsize=9)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)

    for j in range(len(present), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Confidence distribution per class", fontsize=12)
    plt.tight_layout()
    path = f"{output_dir}/confidence_histograms.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


def plot_detections_per_image(dets_per_image, output_dir):
    """Histogram of total detections per image."""
    arr = np.array(dets_per_image)
    max_val = int(arr.max()) if arr.max() > 0 else 1
    bins = min(max_val + 1, 40)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(arr, bins=bins, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(arr.mean(), color="tomato", linestyle="--", linewidth=1.2,
               label=f"mean={arr.mean():.1f}")
    ax.set_title("Detections per image")
    ax.set_xlabel("Number of detections")
    ax.set_ylabel("Number of images")
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = f"{output_dir}/detections_per_image.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {path}")


def plot_instances_per_image_per_class(per_image_class_counts, output_dir):
    """For each class, histogram of how many instances appear per image (images that have ≥1)."""
    present = []
    class_instance_lists = {}
    for cls_name in CLASS_NAMES:
        vals = [c[cls_name] for c in per_image_class_counts if c[cls_name] > 0]
        if vals:
            present.append(cls_name)
            class_instance_lists[cls_name] = vals

    if not present:
        return

    ncols = min(3, len(present))
    nrows = (len(present) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, cls_name in enumerate(present):
        vals = class_instance_lists[cls_name]
        color = COLORS[CLASS_NAMES.index(cls_name)]
        ax = axes[i]
        max_val = max(vals)
        bins = range(1, max_val + 2)
        ax.hist(vals, bins=bins, align="left", color=color, edgecolor="white", alpha=0.85)
        ax.set_title(f"{cls_name}\n(images with ≥1: {len(vals)})", fontsize=9)
        ax.set_xlabel("Instances per image")
        ax.set_ylabel("Number of images")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    for j in range(len(present), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Instances per image per class  (images with ≥1 detection)", fontsize=12)
    plt.tight_layout()
    path = f"{output_dir}/instances_per_image_per_class.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


def plot_class_share_pie(class_confs, output_dir):
    """Pie chart of each class's share of total detections."""
    labels = [c for c in CLASS_NAMES if class_confs[c]]
    sizes = [len(class_confs[c]) for c in labels]
    if not sizes:
        return
    colors = [COLORS[CLASS_NAMES.index(c)] for c in labels]

    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct=lambda p: f"{p:.1f}%\n({int(round(p * sum(sizes) / 100))})",
        startangle=140, pctdistance=0.75,
    )
    for t in autotexts:
        t.set_fontsize(8)
    ax.set_title("Class share of all detections")
    plt.tight_layout()
    path = f"{output_dir}/class_share_pie.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {path}")


def print_summary(class_confs, no_det_count, total_images):
    print("\n── Detection summary ──────────────────────────────────────")
    print(f"{'Class':<22} {'Count':>7} {'Mean conf':>10} {'Min':>7} {'Max':>7}")
    print("-" * 57)
    for cls_name in CLASS_NAMES:
        confs = class_confs[cls_name]
        if confs:
            print(f"{cls_name:<22} {len(confs):>7} {np.mean(confs):>10.3f} "
                  f"{min(confs):>7.3f} {max(confs):>7.3f}")
        else:
            print(f"{cls_name:<22} {'0':>7} {'—':>10}")
    print(f"\n{'Images with no detection':<22} {no_det_count:>7}  ({no_det_count/total_images*100:.1f}%)")
    print()


def plot_all(results, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    class_confs, per_image_class_counts, dets_per_image, no_det_count = _build_stats(results)
    total = len(results)

    plot_instance_counts(class_confs, no_det_count, total, output_dir)
    plot_confidence_histograms(class_confs, output_dir)
    plot_detections_per_image(dets_per_image, output_dir)
    plot_instances_per_image_per_class(per_image_class_counts, output_dir)
    plot_class_share_pie(class_confs, output_dir)
    print_summary(class_confs, no_det_count, total)


if __name__ == "__main__":
    print(f"Running inference on {IMAGE_DIR} ...")
    results = run_inference()
    save_json(results, OUTPUT_JSON)
    plot_all(results, OUTPUT_PLOTS)
