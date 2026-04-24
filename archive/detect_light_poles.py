import shutil
from pathlib import Path

from ultralytics import YOLO

_HERE = Path(__file__).parent

MODEL_PATH = _HERE / "best.pt"
IMAGE_DIR = _HERE / "Knuutilanranta"
OUTPUT_DIR = _HERE / "Knuutilanranta" / "light_poles"
CONF_THRESHOLD = 0.1
LIGHT_POLE_CLASS_ID = 2  # index in ["deer_fence_pole", "gantry_sign_pole", "light_pole", ...]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_PATH)
    image_paths = sorted(IMAGE_DIR.glob("*.png")) + sorted(IMAGE_DIR.glob("*.jpg"))
    total = len(image_paths)
    print(f"Found {total} images. Running inference...")

    # stream=True yields one result at a time without holding all in memory;
    # passing the full list means warmup happens only once.
    moved = 0
    print("Warming up model on first image (may take a moment)...")
    for i, img_path in enumerate(image_paths):
        # predictor is cached after the first call — warmup only runs once
        results = model.predict(str(img_path), conf=CONF_THRESHOLD, verbose=False)
        result = results[0]
        if result.boxes is not None and LIGHT_POLE_CLASS_ID in [int(c) for c in result.boxes.cls.tolist()]:
            dest = OUTPUT_DIR / img_path.name
            shutil.move(str(img_path), str(dest))
            print(f"[{i+1}/{total}] Moved: {img_path.name}")
            moved += 1
        elif (i + 1) % 100 == 0:
            print(f"[{i+1}/{total}] processed...")

    print(f"\nDone — {moved}/{total} images moved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
