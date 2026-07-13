from __future__ import annotations

import argparse
import re
from pathlib import Path


ROBOfLOW_LABEL_PATTERN = re.compile(r"^(\d+)_png\.rf\.[^.]+\.txt$")


def rename_labels(labels_dir: Path, dry_run: bool = False) -> tuple[int, int, int]:
	renamed = 0
	skipped = 0
	conflicts = 0

	for path in sorted(labels_dir.glob("*.txt")):
		match = ROBOfLOW_LABEL_PATTERN.match(path.name)
		if not match:
			skipped += 1
			continue

		new_name = f"{match.group(1)}.txt"
		target = path.with_name(new_name)

		if target.exists() and target != path:
			conflicts += 1
			print(f"conflict: {path.name} -> {new_name} (target exists)")
			continue

		print(f"rename: {path.name} -> {new_name}")
		if not dry_run:
			path.rename(target)
		renamed += 1

	return renamed, skipped, conflicts


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Rename Roboflow label files to simple numeric names."
	)
	parser.add_argument(
		"labels_dir",
		nargs="?",
		default="data/.draft/labels",
		help="Path to labels directory (default: data/.draft/labels)",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Show planned renames without changing files.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	labels_dir = Path(args.labels_dir)

	if not labels_dir.exists() or not labels_dir.is_dir():
		raise SystemExit(f"labels directory not found: {labels_dir}")

	renamed, skipped, conflicts = rename_labels(labels_dir, dry_run=args.dry_run)
	print(f"done: renamed={renamed}, skipped={skipped}, conflicts={conflicts}")


if __name__ == "__main__":
	main()
