from __future__ import annotations

import argparse
from pathlib import Path

SPLITS = ("train", "valid", "val", "test", "draft")

FIX = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate YOLO segmentation label files and report malformed rows."
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1] / "data"),
        help="Dataset root directory containing split folders (default: <repo>/data)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLITS),
        help=(
            "Split folders to scan under --root. "
            "Each split is expected to contain a labels/ directory."
        ),
    )
    return parser.parse_args()


def has_too_few_values(label_file: Path) -> bool:
    lines = label_file.read_text(encoding="utf-8").splitlines()

    for raw_line in lines:
        parts = raw_line.strip().split()
        if not parts:
            continue

        # Segmentation row requires class + at least 3 xy pairs => 7 values.
        if len(parts) < 7:
            return True

    return False


def remove_problematic_lines(label_file: Path) -> int:
    lines = label_file.read_text(encoding="utf-8").splitlines()
    kept_lines: list[str] = []
    removed_count = 0

    for raw_line in lines:
        parts = raw_line.strip().split()
        if not parts:
            continue

        # Segmentation row requires class + at least 3 xy pairs => 7 values.
        if len(parts) < 7:
            removed_count += 1
            continue

        kept_lines.append(raw_line.strip())

    if removed_count:
        content = "\n".join(kept_lines)
        if content:
            content += "\n"
        label_file.write_text(content, encoding="utf-8")

    return removed_count


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()

    if not root.exists() or not root.is_dir():
        raise SystemExit(f"dataset root not found: {root}")

    bad_files: set[Path] = set()
    fixed_files: list[tuple[Path, int]] = []
    missing_label_dirs: list[Path] = []

    for split in args.splits:
        labels_dir = root / split / "labels"
        if not labels_dir.exists() or not labels_dir.is_dir():
            missing_label_dirs.append(labels_dir)
            continue

        files = sorted(labels_dir.glob("*.txt"))
        for label_file in files:
            if has_too_few_values(label_file):
                bad_files.add(label_file)
                if FIX:
                    removed_count = remove_problematic_lines(label_file)
                    if removed_count:
                        fixed_files.append((label_file, removed_count))

    if FIX and fixed_files:
        for fixed_file, removed_count in fixed_files:
            rel_path = fixed_file.relative_to(root)
            print(f"fixed: {rel_path} (removed {removed_count} line(s))")

        # Re-check fixed files before reporting failures.
        bad_files = {path for path in bad_files if has_too_few_values(path)}

    for missing in missing_label_dirs:
        print(f"warning: labels directory not found: {missing}")

    for bad_file in sorted(bad_files):
        print(bad_file.relative_to(root))

    if bad_files:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
