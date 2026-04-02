from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
BG_RATIO = 0.1  # Background images are sampled as a percentage of labeled samples.
SEED = 42

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
SOURCE_IMAGES_DIR = DATA_DIR / "draft" / "images"
SOURCE_LABELS_DIR = DATA_DIR / "draft" / "labels"
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def build_image_index(images_dir: Path) -> dict[str, Path]:
    image_index: dict[str, Path] = {}
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        stem = image_path.stem
        if stem in image_index:
            raise SystemExit(f"duplicate image stem found: {stem}")
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


def ensure_output_dirs() -> None:
    for split in SPLITS:
        (DATA_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DATA_DIR / split / "labels").mkdir(parents=True, exist_ok=True)


def move_sample(
    stem: str,
    split: str,
    image_index: dict[str, Path],
    label_index: dict[str, Path],
    dry_run: bool,
) -> None:
    image_path = image_index.get(stem)
    label_path = label_index.get(stem)

    if image_path is None:
        print(f"skip: missing image for label {stem}.txt")
        return

    if label_path is None:
        print(f"background: {image_path.name}")
    elif label_path.stat().st_size == 0:
        print(f"background: {image_path.name} (empty label)")
    else:
        print(f"move: {image_path.name} + {label_path.name} -> {split}")

    if dry_run:
        return

    image_dst = DATA_DIR / split / "images" / image_path.name
    if image_dst.exists():
        raise SystemExit(f"destination image already exists: {image_dst}")
    shutil.move(str(image_path), str(image_dst))

    if label_path is None:
        label_dst = DATA_DIR / split / "labels" / f"{stem}.txt"
        if label_dst.exists():
            raise SystemExit(f"destination label already exists: {label_dst}")
        label_dst.touch()
        return

    label_dst = DATA_DIR / split / "labels" / label_path.name
    if label_dst.exists():
        raise SystemExit(f"destination label already exists: {label_dst}")
    shutil.move(str(label_path), str(label_dst))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split draft images and labels into train/valid/test folders.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned moves without changing files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not SOURCE_IMAGES_DIR.exists():
        raise SystemExit(f"images directory not found: {SOURCE_IMAGES_DIR}")
    if not SOURCE_LABELS_DIR.exists():
        raise SystemExit(f"labels directory not found: {SOURCE_LABELS_DIR}")

    ensure_output_dirs()

    image_index = build_image_index(SOURCE_IMAGES_DIR)
    label_index = build_label_index(SOURCE_LABELS_DIR)

    labeled_stems = [stem for stem in label_index if stem in image_index and label_index[stem].stat().st_size > 0]
    background_candidates = [stem for stem, image_path in image_index.items() if stem not in label_index or label_index.get(stem, image_path).stat().st_size == 0]
    random.Random(SEED).shuffle(background_candidates)

    missing_images = [stem for stem in label_index if stem not in image_index]
    if missing_images:
        print(f"warning: {len(missing_images)} labels have no matching image")

    background_target = min(len(background_candidates), int(round(len(labeled_stems) * BG_RATIO)))
    selected_backgrounds = background_candidates[:background_target]

    labeled_train, labeled_valid, labeled_test = split_items(labeled_stems, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED)
    bg_train, bg_valid, bg_test = split_items(selected_backgrounds, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED + 1)

    split_counts = {
        "train": {"labeled": len(labeled_train), "background": len(bg_train)},
        "valid": {"labeled": len(labeled_valid), "background": len(bg_valid)},
        "test": {"labeled": len(labeled_test), "background": len(bg_test)},
    }

    for stem in labeled_train + bg_train:
        move_sample(stem, "train", image_index, label_index, args.dry_run)
    for stem in labeled_valid + bg_valid:
        move_sample(stem, "valid", image_index, label_index, args.dry_run)
    for stem in labeled_test + bg_test:
        move_sample(stem, "test", image_index, label_index, args.dry_run)

    print(
        "Dataset split complete: "
        f"labeled={len(labeled_stems)}, "
        f"background_candidates={len(background_candidates)}, "
        f"background_selected={len(selected_backgrounds)}"
    )
    for split_name in SPLITS:
        counts = split_counts[split_name]
        print(
            f"{split_name}: labeled={counts['labeled']}, background={counts['background']}, "
            f"total={counts['labeled'] + counts['background']}"
        )


if __name__ == "__main__":
    main()
