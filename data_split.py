from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from check_labels import (
    download_from_roboflow,
    load_class_names,
    load_env,
    load_roboflow_config,
    roboflow_stem,
)

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42

# Default source when --source is not passed. "local" uses labels already in
# data/.draft/labels; "roboflow" downloads the export and rebuilds them first.
SOURCE = "roboflow"  # "local" or "roboflow"

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
SOURCE_LABELS_DIR = DATA_DIR / ".draft" / "labels"
# Temp raw Roboflow export (yolov8 layout) for --source roboflow. It is deleted
# after the split — the split pairs against archive/ originals, so the export's
# own images/labels are never kept.
ROBOFLOW_DOWNLOAD_DIR = DATA_DIR / ".draft" / "_roboflow"
# Unmatched (leftover) Roboflow labels — those with no archive/ image — are kept
# here after the split.
LEFTOVER_LABELS_DIR = ROOT_DIR / ".draft"
ENV_PATH = ROOT_DIR / ".env"
# Images live under archive/<dataset>/... (scanned recursively). Drop new
# datasets (e.g. vt7) anywhere under archive/ and rerun to include them.
SOURCE_IMAGES_DIR = ROOT_DIR / "archive"
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def download_roboflow_export() -> Path:
    """Download the Roboflow export (yolov8 layout) and return its path."""
    api_key = load_env(ENV_PATH).get("ROBOFLOW_API")
    if not api_key:
        raise SystemExit(f"ROBOFLOW_API not found in {ENV_PATH} (needed for --source roboflow)")
    # Coordinates come from <repo>/roboflow.yaml.
    return download_from_roboflow(ROBOFLOW_DOWNLOAD_DIR, api_key)


def write_data_yaml(export_dir: Path) -> None:
    """Regenerate data/data.yaml from the export's class names + roboflow.yaml."""
    names = load_class_names(export_dir)  # {id: name} from the export's data.yaml
    ordered = [names[i] for i in sorted(names)]
    cfg = load_roboflow_config()

    lines = [
        "train: ../train/images",
        "val: ../valid/images",
        "test: ../test/images",
        "",
        f"nc: {len(ordered)}",
        f"names: {ordered!r}",
        "",
        "roboflow:",
    ]
    for key in ("workspace", "project", "version", "license", "url"):
        if cfg.get(key):
            lines.append(f"  {key}: {cfg[key]}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {DATA_DIR / 'data.yaml'} (nc={len(ordered)})")


def build_roboflow_label_index(export_dir: Path) -> dict[str, Path]:
    """Index every export label by its renamed numeric stem -> source path.

    Labels are not copied/renamed on disk here; the numeric stem is resolved in
    memory and copy_sample writes each label out as `<stem>.txt` when it lands
    in a split, so the rename happens implicitly.
    """
    label_index: dict[str, Path] = {}
    for split in SPLITS:
        labels_dir = export_dir / split / "labels"
        if not labels_dir.is_dir():
            continue
        for label_path in sorted(labels_dir.glob("*.txt")):
            stem = roboflow_stem(label_path.name)
            if stem in label_index:
                raise SystemExit(f"duplicate label stem after rename: {stem}")
            label_index[stem] = label_path
    return label_index


def save_leftover_labels(label_index: dict[str, Path], missing: list[str]) -> None:
    """Keep only the unmatched labels in LEFTOVER_LABELS_DIR (renamed)."""
    LEFTOVER_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    for stale in LEFTOVER_LABELS_DIR.glob("*.txt"):
        stale.unlink()  # drop leftovers from a previous run
    for stem in missing:
        shutil.copy2(label_index[stem], LEFTOVER_LABELS_DIR / f"{stem}.txt")
    print(f"leftover (unmatched) labels: {len(missing)} -> {LEFTOVER_LABELS_DIR}")


def build_image_index(images_dir: Path) -> dict[str, Path]:
    image_index: dict[str, Path] = {}
    for image_path in sorted(images_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        stem = image_path.stem
        existing = image_index.get(stem)
        if existing is not None:
            raise SystemExit(
                f"duplicate image stem '{stem}': {existing} and {image_path}"
            )
        image_index[stem] = image_path

    return image_index


def build_label_index(labels_dir: Path) -> dict[str, Path]:
    label_index: dict[str, Path] = {}
    for label_path in sorted(labels_dir.glob("*.txt")):
        stem = label_path.stem
        if stem in label_index:
            raise SystemExit(f"duplicate label stem found: {stem}")
        label_index[stem] = label_path

    return label_index


def split_items(items: list[str], train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> tuple[list[str], list[str], list[str]]:
    if not items:
        return [], [], []

    if not abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    shuffled = items[:]
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_count = int(round(total * train_ratio))
    val_count = int(round(total * val_ratio))

    if train_count + val_count > total:
        overflow = train_count + val_count - total
        if val_count >= overflow:
            val_count -= overflow
        else:
            train_count -= overflow - val_count
            val_count = 0

    test_count = total - train_count - val_count

    train_items = shuffled[:train_count]
    val_items = shuffled[train_count:train_count + val_count]
    test_items = shuffled[train_count + val_count:train_count + val_count + test_count]
    return train_items, val_items, test_items


def reset_output_dirs(dry_run: bool) -> None:
    for split in SPLITS:
        for kind in ("images", "labels"):
            target = DATA_DIR / split / kind
            if not dry_run and target.exists():
                shutil.rmtree(target)
            if not dry_run:
                target.mkdir(parents=True, exist_ok=True)


def copy_sample(
    stem: str,
    split: str,
    image_index: dict[str, Path],
    label_index: dict[str, Path],
    dry_run: bool,
) -> bool:
    image_path = image_index.get(stem)
    label_path = label_index[stem]

    if image_path is None:
        print(f"skip: no image found for label {stem}.txt")
        return False

    is_background = label_path.stat().st_size == 0
    tag = "background" if is_background else "labeled"
    print(f"copy [{tag}]: {image_path.name} + {label_path.name} -> {split}")

    if dry_run:
        return True

    image_dst = DATA_DIR / split / "images" / image_path.name
    shutil.copy2(image_path, image_dst)

    label_dst = DATA_DIR / split / "labels" / f"{stem}.txt"
    shutil.copy2(label_path, label_dst)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match .draft labels to archive images and copy into train/valid/test."
    )
    parser.add_argument(
        "--source",
        choices=["local", "roboflow"],
        default=SOURCE,
        help=(
            "'local' uses labels already in data/.draft/labels; 'roboflow' downloads "
            "the Roboflow export first and flattens+renames its labels there "
            f"(default: {SOURCE})."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without changing files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    export_dir: Path | None = None
    if args.source == "roboflow":
        export_dir = download_roboflow_export()

    if not SOURCE_IMAGES_DIR.exists():
        raise SystemExit(f"images directory not found: {SOURCE_IMAGES_DIR}")

    image_index = build_image_index(SOURCE_IMAGES_DIR)
    if export_dir is not None:
        label_index = build_roboflow_label_index(export_dir)
    else:
        if not SOURCE_LABELS_DIR.exists():
            raise SystemExit(f"labels directory not found: {SOURCE_LABELS_DIR}")
        label_index = build_label_index(SOURCE_LABELS_DIR)

    matched = [stem for stem in label_index if stem in image_index]
    missing = [stem for stem in label_index if stem not in image_index]
    if missing:
        print(f"warning: {len(missing)} labels have no matching image (skipped)")

    # Empty label files are background/negative samples; split them separately
    # so they are distributed proportionally across train/valid/test.
    foreground = [stem for stem in matched if label_index[stem].stat().st_size > 0]
    background = [stem for stem in matched if label_index[stem].stat().st_size == 0]

    fg_train, fg_valid, fg_test = split_items(foreground, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED)
    bg_train, bg_valid, bg_test = split_items(background, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED + 1)

    per_split = {
        "train": (fg_train, bg_train),
        "valid": (fg_valid, bg_valid),
        "test": (fg_test, bg_test),
    }

    reset_output_dirs(args.dry_run)

    copied = {split: 0 for split in SPLITS}
    for split in SPLITS:
        fg_stems, bg_stems = per_split[split]
        for stem in fg_stems + bg_stems:
            if copy_sample(stem, split, image_index, label_index, args.dry_run):
                copied[split] += 1

    print(
        "Split complete: "
        f"labels={len(label_index)}, matched={len(matched)}, missing={len(missing)}, "
        f"foreground={len(foreground)}, background={len(background)}"
    )
    for split in SPLITS:
        fg_stems, bg_stems = per_split[split]
        print(
            f"{split}: labeled={len(fg_stems)}, background={len(bg_stems)}, "
            f"total={len(fg_stems) + len(bg_stems)}, copied={copied[split]}"
        )

    if export_dir is not None:
        if not args.dry_run:
            # Regenerate data/data.yaml, keep only the unmatched labels, and
            # drop the raw export (images/labels).
            write_data_yaml(export_dir)
            save_leftover_labels(label_index, missing)
        shutil.rmtree(export_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
