import random
import shutil
from pathlib import Path

_HERE = Path(__file__).parent

IMAGE_DIR = _HERE / "Knuutilanranta"
KEEP_DIR = IMAGE_DIR / "keep"
SKIP_DIR = IMAGE_DIR / "skip"


def main():
    KEEP_DIR.mkdir(parents=True, exist_ok=True)
    SKIP_DIR.mkdir(parents=True, exist_ok=True)

    # only images directly in Knuutilanranta, not in subfolders
    images = [p for p in IMAGE_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg"}]
    random.shuffle(images)

    half = len(images) // 2
    keep_images = images[:half]
    skip_images = images[half:]

    for p in keep_images:
        shutil.move(str(p), str(KEEP_DIR / p.name))
    for p in skip_images:
        shutil.move(str(p), str(SKIP_DIR / p.name))

    print(f"Total: {len(images)}  |  keep: {len(keep_images)}  |  skip: {len(skip_images)}")


if __name__ == "__main__":
    main()
