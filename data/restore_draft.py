from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
SOURCE_SPLITS = ("train", "valid", "test")
DEST_DIR = DATA_DIR / "draft"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_destination_dirs() -> None:
    (DEST_DIR / "images").mkdir(parents=True, exist_ok=True)
    (DEST_DIR / "labels").mkdir(parents=True, exist_ok=True)


def collect_images(split_dir: Path) -> list[Path]:
    images_dir = split_dir / "images"
    if not images_dir.exists():
        return []
    return [path for path in sorted(images_dir.iterdir()) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def restore_split(split: str, dry_run: bool) -> tuple[int, int]:
    split_dir = DATA_DIR / split
    images = collect_images(split_dir)
    restored_images = 0
    restored_labels = 0

    for image_path in images:
        label_path = split_dir / "labels" / f"{image_path.stem}.txt"

        image_dst = DEST_DIR / "images" / image_path.name
        if image_dst.exists():
            raise SystemExit(f"destination image already exists: {image_dst}")

        print(f"move image: {image_path.relative_to(DATA_DIR)} -> draft/images/{image_path.name}")
        if not dry_run:
            shutil.move(str(image_path), str(image_dst))
        restored_images += 1

        if label_path.exists():
            label_dst = DEST_DIR / "labels" / label_path.name
            if label_dst.exists():
                raise SystemExit(f"destination label already exists: {label_dst}")

            print(f"move label: {label_path.relative_to(DATA_DIR)} -> draft/labels/{label_path.name}")
            if not dry_run:
                shutil.move(str(label_path), str(label_dst))
            restored_labels += 1

    return restored_images, restored_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move images and labels from train/valid/test back into data/draft."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned moves without changing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for split in SOURCE_SPLITS:
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            raise SystemExit(f"split directory not found: {split_dir}")

    ensure_destination_dirs()

    total_images = 0
    total_labels = 0
    for split in SOURCE_SPLITS:
        images, labels = restore_split(split, args.dry_run)
        total_images += images
        total_labels += labels

    print(
        "restore complete: "
        f"images={total_images}, labels={total_labels}, dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()