#!/usr/bin/env python3
"""Per-image metadata sidecars for the prepared PoleSight dataset.

Writes two files beside the splits:

``data/tags.json``
    The Roboflow project's own image tags, keyed by stem. Kept separate because
    it is a straight mirror of what annotators set in Roboflow, with no local
    interpretation layered on top.

``data/metadata.json``
    One row per prepared image: where it landed, what it holds, and — the part
    that cannot be recovered from the files afterwards — which imaging domain it
    came from.

Why the domain field exists: this dataset merges two acquisition modes. Most
drives record 16-bit single-channel intensity panoramas, and a capture that
predates them may not. ``dataset_prepare.py --rgb`` converts everything to 8-bit
RGB so the trainer sees one format, which means the prepared image no longer says which mode it came
from. Its archive/ original still does, so both the route and the pre-conversion
format are read from there and recorded here. Anything that wants to stratify,
weight, or report per domain needs this file.

Usage:
    python image_metadata.py            # rebuild both sidecars
    python image_metadata.py --no-api   # skip Roboflow; omit tags and ids
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# The archive rule lives in dataset_prepare so this file cannot describe a
# different set of images than the one the preparer actually used.
from dataset_prepare import build_image_index, source_route

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
ARCHIVE_DIR = ROOT_DIR / "archive"
METADATA_PATH = DATA_DIR / "metadata.json"
TAGS_PATH = DATA_DIR / "tags.json"

SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Pillow mode -> (domain, channels, bit depth). The domain name is the thing
# downstream code should branch on; mode and depth are recorded so the mapping
# stays auditable rather than being a bare label.
MODE_PROFILES = {
    "I;16": ("ir_intensity", 1, 16),
    "I": ("ir_intensity", 1, 32),
    "L": ("ir_intensity", 1, 8),
    "RGB": ("colour", 3, 8),
    "RGBA": ("colour", 4, 8),
}


def load_class_names() -> dict[int, str]:
    """Class id -> name, from the prepared data.yaml."""
    import yaml

    path = DATA_DIR / "data.yaml"
    if not path.exists():
        raise SystemExit(f"not found: {path} (run dataset_prepare.py first)")
    names = yaml.safe_load(path.read_text(encoding="utf-8")).get("names", [])
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {index: str(name) for index, name in enumerate(names)}


def index_archive() -> dict[str, Path]:
    """Stem -> its original under archive/, or {} if archive is absent.

    Built with dataset_prepare's own indexer rather than a second walk, so this
    file can only ever describe the images that preparer would pick: the same
    intensity_filtered/-only rule, applied once. The result answers both
    provenance questions at once — the first path component is the capture route,
    and the file itself is the pre-conversion image.
    """
    if not ARCHIVE_DIR.is_dir():
        return {}
    return build_image_index(ARCHIVE_DIR)


def source_profile(path: Path) -> tuple[str, str, int, int]:
    """(pillow mode, domain, channels, bit depth) of an archive/ original.

    Read from archive/ rather than the prepared image: after --rgb every prepared
    file reports RGB, so the original is the only remaining record of what the
    capture actually was. Opened per prepared stem, not per archive file, since
    archive/ holds several times more images than any one dataset version uses.
    """
    from PIL import Image

    with Image.open(path) as image:
        mode = image.mode
    domain, channels, depth = MODE_PROFILES.get(mode, ("unknown", 0, 0))
    return mode, domain, channels, depth


def fetch_roboflow(config) -> dict[str, dict]:
    """Stem -> {id, created, split, tags} straight from the Roboflow project.

    ``search_all`` is repeated until the id set stops growing: a single pass is
    not guaranteed to enumerate the whole project.
    """
    from roboflow import Roboflow

    client = Roboflow(api_key=config["api_key"])
    project = client.workspace(config["workspace"]).project(config["project"])
    by_id: dict[str, dict] = {}
    unchanged = 0
    for _ in range(10):
        before = len(by_id)
        for page in project.search_all(
            fields=["id", "name", "split", "tags", "created"], limit=100
        ):
            for row in page:
                if row.get("id") and row.get("name"):
                    by_id[row["id"]] = row
        unchanged = unchanged + 1 if len(by_id) == before else 0
        if unchanged >= 2:
            break

    records: dict[str, dict] = {}
    for row in by_id.values():
        records[Path(row["name"]).stem] = {
            "roboflow_id": row["id"],
            "roboflow_split": row.get("split"),
            "uploaded_at": (
                datetime.fromtimestamp(row["created"] / 1000).isoformat(timespec="seconds")
                if row.get("created") else None
            ),
            "tags": sorted(row.get("tags") or []),
        }
    return records


def read_label(path: Path) -> Counter:
    """Instance count per class id for one YOLO label file."""
    counts: Counter[int] = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            counts[int(line.split()[0])] += 1
    return counts


def build_rows(class_names, sources, roboflow) -> dict[str, dict]:
    """Assemble one metadata row per prepared image."""
    from PIL import Image

    rows: dict[str, dict] = {}
    for split in SPLITS:
        images_dir = DATA_DIR / split / "images"
        if not images_dir.is_dir():
            continue
        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            stem = image_path.stem
            label_path = DATA_DIR / split / "labels" / f"{stem}.txt"
            counts = read_label(label_path) if label_path.exists() else Counter()

            with Image.open(image_path) as image:
                width, height, mode = image.width, image.height, image.mode
            stored_channels = MODE_PROFILES.get(mode, ("unknown", 0, 0))[1]

            # No archive/ original means the prepared file was not produced by
            # the current pipeline (a stale split, or a hand-placed image), so
            # report what is on disk and leave the route unattributed.
            origin = sources.get(stem)
            if origin is None:
                source_mode, domain, source_channels, source_depth = (
                    (mode,) + MODE_PROFILES.get(mode, ("unknown", 0, 0))
                )
                route = None
            else:
                source_mode, domain, source_channels, source_depth = source_profile(origin)
                route = source_route(origin, ARCHIVE_DIR)
            record = roboflow.get(stem, {})

            rows[stem] = {
                "split": split,
                "image": image_path.name,
                "width": width,
                "height": height,
                # What the trainer loads.
                "mode": mode,
                "channels": stored_channels,
                "bit_depth": 8,
                # What the sensor produced. `converted` is true where --rgb
                # re-encoded a 16-bit capture into 8-bit RGB.
                "domain": domain,
                "source_mode": source_mode,
                "source_channels": source_channels,
                "source_bit_depth": source_depth,
                "converted": source_mode != mode,
                "route": route,
                "source_image": str(origin.relative_to(ROOT_DIR)) if origin else None,
                "background": sum(counts.values()) == 0,
                "n_instances": sum(counts.values()),
                "classes": sorted(class_names.get(c, f"class_{c}") for c in counts),
                "instances_per_class": {
                    class_names.get(c, f"class_{c}"): n for c, n in sorted(counts.items())
                },
                "file_size_kb": round(image_path.stat().st_size / 1024, 1),
                **record,
            }
    return rows


def summarise(rows: dict[str, dict]) -> dict:
    """Cross-tabulations worth having without re-reading every row."""
    by_domain_split: dict[str, Counter] = defaultdict(Counter)
    for row in rows.values():
        by_domain_split[row["domain"]][row["split"]] += 1
    return {
        "domain_counts": dict(Counter(r["domain"] for r in rows.values())),
        "domain_by_split": {d: dict(c) for d, c in by_domain_split.items()},
        "route_counts": dict(Counter(r["route"] or "unknown" for r in rows.values())),
        "channel_counts": dict(Counter(r["channels"] for r in rows.values())),
        "source_mode_counts": dict(Counter(r["source_mode"] for r in rows.values())),
        "converted_images": sum(r["converted"] for r in rows.values()),
        "background_images": sum(r["background"] for r in rows.values()),
        "split_counts": dict(Counter(r["split"] for r in rows.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Skip the Roboflow lookup; tags, ids and upload times are omitted.",
    )
    args = parser.parse_args()

    class_names = load_class_names()
    sources = index_archive()

    roboflow: dict[str, dict] = {}
    if not args.no_api:
        import dataset_prepare as dp

        config = dp.load_roboflow_config()
        if config.get("api_key"):
            roboflow = fetch_roboflow(config)
            print(f"roboflow: {len(roboflow)} image records")
        else:
            print("roboflow: no ROBOFLOW_API in .env; continuing without tags")

    rows = build_rows(class_names, sources, roboflow)
    if not rows:
        raise SystemExit("no prepared images found; run dataset_prepare.py first")

    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fields = sorted({key for row in rows.values() for key in row})
    populated = {
        field: sum(1 for row in rows.values() if row.get(field) not in (None, [], {}))
        for field in fields
    }

    tagged = {stem: row["tags"] for stem, row in rows.items() if row.get("tags")}
    TAGS_PATH.write_text(json.dumps({
        "source": "roboflow project image tags",
        "generated": generated,
        "images": len(rows),
        "tagged_images": len(tagged),
        "tag_counts": dict(Counter(t for tags in tagged.values() for t in tags)),
        "tags": tagged,
    }, indent=2) + "\n", encoding="utf-8")

    METADATA_PATH.write_text(json.dumps({
        "source": (
            "prepared splits in data/ + their pre-conversion originals and route "
            "layout under archive/ + roboflow image records"
        ),
        "generated": generated,
        "images": len(rows),
        "classes": [class_names[k] for k in sorted(class_names)],
        "fields": fields,
        "populated_by_field": populated,
        **summarise(rows),
        "metadata": rows,
    }, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {METADATA_PATH.relative_to(ROOT_DIR)} ({len(rows)} images)")
    print(f"wrote {TAGS_PATH.relative_to(ROOT_DIR)} ({len(tagged)} tagged)")
    summary = summarise(rows)
    print("  domains:", summary["domain_counts"])
    print("  routes: ", summary["route_counts"])
    print("  channels:", summary["channel_counts"], "| converted:", summary["converted_images"])


if __name__ == "__main__":
    main()
