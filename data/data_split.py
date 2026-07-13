from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
SOURCE_LABELS_DIR = DATA_DIR / ".draft" / "labels"
# Images live under archive/<dataset>/... (scanned recursively). Drop new
# datasets (e.g. vt7) anywhere under archive/ and rerun to include them.
SOURCE_IMAGES_DIR = ROOT_DIR / "archive"
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


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
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without changing files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not SOURCE_IMAGES_DIR.exists():
        raise SystemExit(f"images directory not found: {SOURCE_IMAGES_DIR}")
    if not SOURCE_LABELS_DIR.exists():
        raise SystemExit(f"labels directory not found: {SOURCE_LABELS_DIR}")

    image_index = build_image_index(SOURCE_IMAGES_DIR)
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


if __name__ == "__main__":
    main()
