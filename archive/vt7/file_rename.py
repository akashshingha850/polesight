#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rename images and update annotations. Supports two modes: "
            "standard (paired images/labels) or COCO (JSON with file_name updates)."
        )
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        help="Directory containing images.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        help="Directory containing label files (standard mode only).",
    )
    parser.add_argument(
        "--coco-json",
        type=Path,
        help="COCO JSON file with image metadata (COCO mode).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for renamed images (COCO mode) or in-place rename (standard mode).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=5301,
        help="Starting number for new filenames (default: 5301, i.e. 05301).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=5,
        help="Zero-padding width for generated names (default: 5).",
    )
    parser.add_argument(
        "--label-ext",
        default=".txt",
        help="Annotation file extension (default: .txt).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually perform renaming. Without this flag, script runs in dry-run mode.",
    )
    return parser.parse_args()


def collect_pairs(images_dir: Path, labels_dir: Path, label_ext: str) -> list[tuple[Path, Path]]:
    """Collect matched image/label pairs for standard mode."""
    images_by_stem = {
        p.stem: p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }
    labels_by_stem = {
        p.stem: p
        for p in labels_dir.iterdir()
        if p.is_file() and p.suffix.lower() == label_ext.lower()
    }

    common_stems = sorted(set(images_by_stem) & set(labels_by_stem), key=lambda x: int(x) if x.isdigit() else x)
    return [(images_by_stem[stem], labels_by_stem[stem]) for stem in common_stems]


def build_plan(
    pairs: list[tuple[Path, Path]],
    start: int,
    width: int,
) -> list[tuple[Path, Path]]:
    """Build a rename plan for paired files."""
    plan: list[tuple[Path, Path]] = []
    for index, (image_file, label_file) in enumerate(pairs):
        new_stem = str(start + index).zfill(width)
        plan.append((image_file, image_file.with_name(new_stem + image_file.suffix.lower())))
        plan.append((label_file, label_file.with_name(new_stem + label_file.suffix.lower())))
    return plan


def has_target_conflicts(plan: list[tuple[Path, Path]]) -> list[Path]:
    """Check for conflicting target filenames in a plan."""
    targets = [dst for _, dst in plan]
    duplicate_targets = {p for p in targets if targets.count(p) > 1}

    existing_not_from_sources = []
    source_set = {src for src, _ in plan}
    for _, dst in plan:
        if dst.exists() and dst not in source_set:
            existing_not_from_sources.append(dst)

    return sorted(duplicate_targets | set(existing_not_from_sources))


def execute_plan(plan: list[tuple[Path, Path]]) -> None:
    """Execute a rename plan using a safe two-phase strategy."""
    temp_moves: list[tuple[Path, Path]] = []
    for src, _ in plan:
        temp = src.with_name(src.name + ".renametmp")
        counter = 1
        while temp.exists():
            temp = src.with_name(src.name + f".renametmp{counter}")
            counter += 1
        src.rename(temp)
        temp_moves.append((temp, src))

    source_to_temp = {original: temp for temp, original in temp_moves}
    for src, dst in plan:
        source_to_temp[src].rename(dst)


def load_coco_json(json_path: Path) -> dict:
    """Load COCO JSON file."""
    with open(json_path) as f:
        return json.load(f)


def save_coco_json(data: dict, json_path: Path) -> None:
    """Save COCO JSON file."""
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


def process_coco_mode(
    images_dir: Path,
    coco_json: Path,
    output_dir: Path,
    start: int,
    width: int,
    apply: bool,
) -> None:
    """Process images and COCO JSON, copying renamed files to output_dir."""
    images_dir = images_dir.resolve()
    output_dir = output_dir.resolve()

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not coco_json.is_file():
        raise FileNotFoundError(f"COCO JSON file not found: {coco_json}")

    coco_data = load_coco_json(coco_json)
    if "images" not in coco_data:
        raise ValueError("COCO JSON missing 'images' key")

    image_files = sorted(
        [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.stem,
    )

    if not image_files:
        print("No images found.")
        return

    filename_map = {}
    plan = []
    for index, src in enumerate(image_files):
        new_stem = str(start + index).zfill(width)
        new_name = new_stem + src.suffix.lower()
        filename_map[src.name] = new_name
        plan.append((src.name, new_name))

    updated_count = 0
    for img_entry in coco_data.get("images", []):
        old_name = img_entry.get("file_name", "")
        if old_name in filename_map:
            img_entry["file_name"] = filename_map[old_name]
            updated_count += 1

    print(f"Image files found: {len(image_files)}")
    print(f"COCO entries updated: {updated_count}")
    print(f"Start index: {start}")
    print(f"Name width: {width}")
    print("")
    print("Planned renames (first 20):")
    for src_name, dst_name in plan[:20]:
        print(f"  {src_name} -> {dst_name}")
    if len(plan) > 20:
        print(f"  ... and {len(plan) - 20} more")

    if not apply:
        print("")
        print("Dry-run only. Re-run with --apply to perform the copy/rename.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for src in image_files:
        new_name = filename_map[src.name]
        dst = output_dir / new_name
        shutil.copy2(src, dst)

    output_json = output_dir / coco_json.name
    save_coco_json(coco_data, output_json)

    print("")
    print(f"Copied {len(image_files)} renamed images to {output_dir}")
    print(f"Updated COCO JSON saved to {output_json}")


def main() -> None:
    args = parse_args()

    if args.width <= 0:
        raise ValueError("--width must be positive.")

    if args.coco_json:
        if not args.images_dir:
            raise ValueError("--images-dir required for COCO mode")
        if not args.output_dir:
            raise ValueError("--output-dir required for COCO mode")
        process_coco_mode(
            args.images_dir,
            args.coco_json,
            args.output_dir,
            args.start,
            args.width,
            args.apply,
        )
        return

    if not args.images_dir or not args.labels_dir:
        raise ValueError("--images-dir and --labels-dir required for standard mode")

    if not args.images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {args.images_dir}")
    if not args.labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {args.labels_dir}")

    label_ext = args.label_ext if args.label_ext.startswith(".") else f".{args.label_ext}"
    pairs = collect_pairs(args.images_dir, args.labels_dir, label_ext)

    if not pairs:
        print("No matching image/annotation pairs found.")
        return

    plan = build_plan(pairs, args.start, args.width)
    conflicts = has_target_conflicts(plan)
    if conflicts:
        print("Rename aborted due to conflicts with existing files:")
        for path in conflicts:
            print(f"  - {path}")
        return

    print(f"Matched pairs: {len(pairs)}")
    print(f"Start index : {args.start}")
    print(f"Name width  : {args.width}")
    print("")
    print("Planned renames:")
    for src, dst in plan:
        print(f"  {src.name} -> {dst.name}")

    if not args.apply:
        print("")
        print("Dry-run only. Re-run with --apply to perform the rename.")
        return

    execute_plan(plan)
    print("")
    print("Renaming completed.")


if __name__ == "__main__":
    main()
