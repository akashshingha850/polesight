from pathlib import Path
import shutil


def main() -> None:
	source_dir = Path(__file__).resolve().parent
	keep_dir = source_dir / "keep"
	keep_dir.mkdir(exist_ok=True)

	image_files = sorted(
		path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png"
	)
	selected_files = image_files[::3]

	moved_count = 0
	skipped_count = 0

	for image_path in selected_files:
		destination_path = keep_dir / image_path.name
		if destination_path.exists():
			skipped_count += 1
			continue

		shutil.move(str(image_path), str(destination_path))
		moved_count += 1

	print(f"Scanned .png files: {len(image_files)}")
	print(f"Selected every 3rd file: {len(selected_files)}")
	print(f"Moved to keep/: {moved_count}")
	print(f"Skipped (already existed): {skipped_count}")


if __name__ == "__main__":
	main()
