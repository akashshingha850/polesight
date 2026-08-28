from __future__ import annotations

import argparse
import ast
import random
import re
import shutil
from collections import Counter
from pathlib import Path

# Keep PoleSight's existing 80:10:10 split. Unlike Roadsight, PoleSight has no
# season field and its captures do not need to be grouped by location. The split
# therefore balances foreground/background images and per-class instances only.
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42
CLASS_BALANCE_WEIGHT = 0.5

# "local" uses labels already in data/.draft/labels. "roboflow" downloads the
# version pinned in roboflow.yaml and rebuilds the local dataset from that export.
SOURCE = "roboflow"

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
SOURCE_LABELS_DIR = DATA_DIR / ".draft" / "labels"
ROBOFLOW_DOWNLOAD_DIR = DATA_DIR / ".draft" / "_roboflow"
LEFTOVER_LABELS_DIR = ROOT_DIR / ".draft" / "leftover"
ENV_PATH = ROOT_DIR / ".env"

# Original images are stored below archive/<dataset>/... and are intentionally
# scanned recursively. Directory names are not used as split groups.
SOURCE_IMAGES_DIR = ROOT_DIR / "archive"
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ROBOFLOW_LABEL_PATTERN = re.compile(
    r"^(?P<stem>.+?)_(?:jpg|jpeg|png)\.rf\.[^.]+\.txt$",
    re.IGNORECASE,
)


def load_env(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE environment file without another dependency."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_roboflow_config() -> dict[str, str]:
    """Read the flat Roboflow coordinates from roboflow.yaml."""
    config_path = ROOT_DIR / "roboflow.yaml"
    config: dict[str, str] = {}
    if not config_path.exists():
        return config
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        config[key.strip()] = value.strip()
    return config


def download_from_roboflow(destination: Path, api_key: str) -> Path:
    """Download or reuse the configured raw YOLO export."""
    config = load_roboflow_config()
    missing = [
        key for key in ("workspace", "project", "version") if not config.get(key)
    ]
    if missing:
        raise SystemExit(
            "cannot download from Roboflow: missing "
            f"{', '.join(missing)} in {ROOT_DIR / 'roboflow.yaml'}"
        )
    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise SystemExit(
            "the 'roboflow' package is required for --source roboflow "
            "(pip install roboflow)"
        ) from exc

    complete = all(
        (destination / split / "labels").is_dir() for split in SPLITS
    )
    if complete:
        print(f"reusing Roboflow download at {destination}")
        return destination

    print(
        f"downloading {config['workspace']}/{config['project']} "
        f"v{config['version']} -> {destination}"
    )
    client = Roboflow(api_key=api_key)
    project = client.workspace(config["workspace"]).project(config["project"])
    version = project.version(int(config["version"]))
    dataset = version.download("yolov8", location=str(destination), overwrite=True)
    return Path(dataset.location)


def load_class_names(dataset_root: Path) -> dict[int, str]:
    """Read class names from a Roboflow data.yaml list or mapping."""
    yaml_path = dataset_root / "data.yaml"
    if not yaml_path.exists():
        return {}

    lines = yaml_path.read_text(encoding="utf-8").splitlines()
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line.startswith("names:"):
            continue
        value = line.split("names:", 1)[1].strip()
        if value:
            try:
                names = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                return {}
            if isinstance(names, dict):
                return {int(class_id): str(name) for class_id, name in names.items()}
            if isinstance(names, (list, tuple)):
                return {class_id: str(name) for class_id, name in enumerate(names)}
            return {}

        names = []
        for following in lines[index + 1 :]:
            item = following.strip()
            if item.startswith("- "):
                names.append(item[2:].strip().strip('"').strip("'"))
            elif item and not following.startswith((" ", "\t")):
                break
        return {class_id: name for class_id, name in enumerate(names)}
    return {}


def roboflow_stem(filename: str) -> str:
    """Recover an original stem from a Roboflow-mangled label filename."""
    match = ROBOFLOW_LABEL_PATTERN.match(filename)
    return match.group("stem") if match else Path(filename).stem


def download_roboflow_export() -> Path:
    """Download the Roboflow version configured for this repository."""
    api_key = load_env(ENV_PATH).get("ROBOFLOW_API")
    if not api_key:
        raise SystemExit(
            f"ROBOFLOW_API not found in {ENV_PATH} (needed for --source roboflow)"
        )
    return download_from_roboflow(ROBOFLOW_DOWNLOAD_DIR, api_key)


def write_data_yaml(export_dir: Path) -> None:
    """Regenerate data/data.yaml from the export and roboflow.yaml."""
    names = load_class_names(export_dir)
    ordered = [names[class_id] for class_id in sorted(names)]
    if not ordered:
        raise SystemExit(f"no class names found in {export_dir / 'data.yaml'}")

    cfg = load_roboflow_config()
    lines = [
        "train: ../train/images",
        "val: ../valid/images",
        "test: ../test/images",
        "",
        f"nc: {len(ordered)}",
        f"names: {ordered!r}",
        "",
        "roboflow:",
    ]
    for key in ("workspace", "project", "version", "license", "url"):
        if cfg.get(key):
            lines.append(f"  {key}: {cfg[key]}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    yaml_path = DATA_DIR / "data.yaml"
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {yaml_path} (nc={len(ordered)})")


def build_roboflow_label_index(export_dir: Path) -> dict[str, Path]:
    """Index exported labels by their original image stem."""
    label_index: dict[str, Path] = {}
    for split in SPLITS:
        labels_dir = export_dir / split / "labels"
        if not labels_dir.is_dir():
            continue
        for label_path in sorted(labels_dir.glob("*.txt")):
            stem = roboflow_stem(label_path.name)
            if stem in label_index:
                raise SystemExit(
                    f"duplicate label stem after Roboflow rename: '{stem}' "
                    f"({label_index[stem].name} and {label_path.name}). "
                    "Use a Roboflow version without generated augmentations."
                )
            label_index[stem] = label_path
    return label_index


def save_leftover_labels(label_index: dict[str, Path], missing: list[str]) -> None:
    """Keep unmatched labels for inspection, replacing leftovers from the last run."""
    LEFTOVER_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    for stale in LEFTOVER_LABELS_DIR.glob("*.txt"):
        stale.unlink()
    for stem in missing:
        shutil.copy2(label_index[stem], LEFTOVER_LABELS_DIR / f"{stem}.txt")
    print(f"leftover labels: {len(missing)} -> {LEFTOVER_LABELS_DIR}")


def build_image_index(images_dir: Path) -> dict[str, Path]:
    """Recursively index source images, failing on ambiguous duplicate stems."""
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
    """Index clean local labels by stem."""
    label_index: dict[str, Path] = {}
    for label_path in sorted(labels_dir.glob("*.txt")):
        stem = label_path.stem
        if stem in label_index:
            raise SystemExit(f"duplicate label stem found: {stem}")
        label_index[stem] = label_path
    return label_index


def class_counts(label_path: Path) -> Counter[int]:
    """Return the number of instances of each class in one YOLO label file."""
    counts: Counter[int] = Counter()
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        parts = raw_line.split()
        if not parts:
            continue
        try:
            class_id = int(float(parts[0]))
        except ValueError as exc:
            raise SystemExit(
                f"invalid class id in {label_path}, line {line_number}: {parts[0]!r}"
            ) from exc
        counts[class_id] += 1
    return counts


def _deviation_delta(
    placed: Counter, wanted: Counter, profile: Counter
) -> float:
    """Increase in normalized squared deviation after adding a sample."""
    delta = 0.0
    for key, count in profile.items():
        target = max(1.0, wanted[key])
        before = placed[key] - wanted[key]
        delta += ((before + count) ** 2 - before**2) / target
    return delta


def stratified_split(
    matched: list[str], label_index: dict[str, Path]
) -> dict[str, tuple[list[str], list[str]]]:
    """Split individual images while balancing fg/bg and class instances.

    This is the PoleSight counterpart of Roadsight's grouped, seasonal splitter.
    There is deliberately no season stratum and no location group: every image is
    assigned independently. Multi-class images are still balanced by their class
    instance counts, which avoids starving validation or test of a rare pole class.
    """
    if not matched:
        return {split: ([], []) for split in SPLITS}

    ratios = dict(zip(SPLITS, (TRAIN_RATIO, VAL_RATIO, TEST_RATIO)))
    is_background = {
        stem: label_index[stem].stat().st_size == 0 for stem in matched
    }
    stratum_of = {
        stem: "background" if is_background[stem] else "foreground"
        for stem in matched
    }
    strata = ("foreground", "background")

    instances_of = {stem: class_counts(label_index[stem]) for stem in matched}
    class_totals: Counter[int] = Counter()
    for counts in instances_of.values():
        class_totals.update(counts)

    stratum_totals = Counter(stratum_of.values())
    wanted_images = {
        split: Counter(
            {stratum: ratios[split] * stratum_totals[stratum] for stratum in strata}
        )
        for split in SPLITS
    }
    wanted_instances = {
        split: Counter(
            {
                class_id: ratios[split] * total
                for class_id, total in class_totals.items()
            }
        )
        for split in SPLITS
    }
    placed_images = {split: Counter() for split in SPLITS}
    placed_instances = {split: Counter() for split in SPLITS}

    # Seeded random order prevents filenames from deciding ties. Samples carrying
    # rare classes go first because they are the hardest to place well.
    rarity = {
        stem: sum(
            count / class_totals[class_id]
            for class_id, count in instances_of[stem].items()
        )
        for stem in matched
    }
    order = sorted(matched)
    random.Random(SEED).shuffle(order)
    order.sort(key=lambda stem: rarity[stem], reverse=True)

    image_scale = float(max(1, len(matched)))
    instance_scale = float(max(1, sum(class_totals.values())))
    weight = CLASS_BALANCE_WEIGHT if class_totals else 0.0
    assigned: dict[str, str] = {}

    for stem in order:
        image_profile = Counter({stratum_of[stem]: 1})
        instance_profile = instances_of[stem]

        def cost_increase(split: str) -> tuple[float, str]:
            image_delta = _deviation_delta(
                placed_images[split], wanted_images[split], image_profile
            )
            cost = (1.0 - weight) * image_delta / image_scale
            if weight:
                instance_delta = _deviation_delta(
                    placed_instances[split],
                    wanted_instances[split],
                    instance_profile,
                )
                cost += weight * instance_delta / instance_scale
            return cost, split

        destination = min(SPLITS, key=cost_increase)
        assigned[stem] = destination
        placed_images[destination].update(image_profile)
        placed_instances[destination].update(instance_profile)

    # A squared-error objective can rationally round a very rare class to zero in
    # a 10% split (for example, a target of 0.7 instances). If that class occurs
    # in at least three different images, repair coverage by swapping same-stratum
    # images. This keeps every split's image and fg/bg counts unchanged.
    class_image_totals = Counter(
        class_id
        for counts in instances_of.values()
        for class_id in counts
    )
    for class_id in sorted(class_totals, key=lambda cid: (class_image_totals[cid], cid)):
        if class_image_totals[class_id] < len(SPLITS):
            continue
        for destination in SPLITS:
            if placed_instances[destination][class_id]:
                continue

            swaps: list[tuple[float, str, str, str]] = []
            for candidate, donor in assigned.items():
                candidate_profile = instances_of[candidate]
                if class_id not in candidate_profile or donor == destination:
                    continue
                # Do not fix one empty class cell by creating another in the donor.
                if any(
                    placed_instances[donor][cid] <= count
                    for cid, count in candidate_profile.items()
                ):
                    continue

                for replacement, replacement_split in assigned.items():
                    if replacement_split != destination:
                        continue
                    if stratum_of[replacement] != stratum_of[candidate]:
                        continue
                    replacement_profile = instances_of[replacement]
                    if class_id in replacement_profile:
                        continue
                    # Likewise, retain every class already represented in the
                    # destination after its replacement moves to the donor.
                    if any(
                        placed_instances[destination][cid] <= count
                        for cid, count in replacement_profile.items()
                    ):
                        continue

                    # Prefer swaps whose class profiles are most alike. The stem
                    # fields make the choice deterministic when penalties tie.
                    profile_distance = sum(
                        abs(candidate_profile[cid] - replacement_profile[cid])
                        for cid in candidate_profile.keys() | replacement_profile.keys()
                    )
                    swaps.append(
                        (profile_distance, candidate, replacement, donor)
                    )

            if not swaps:
                continue
            _, candidate, replacement, donor = min(swaps)
            candidate_profile = instances_of[candidate]
            replacement_profile = instances_of[replacement]
            assigned[candidate] = destination
            assigned[replacement] = donor
            placed_instances[donor].subtract(candidate_profile)
            placed_instances[donor].update(replacement_profile)
            placed_instances[destination].subtract(replacement_profile)
            placed_instances[destination].update(candidate_profile)
            print(
                f"coverage repair: class {class_id} -> {destination} "
                f"(swapped {candidate} with {replacement})"
            )

    per_split: dict[str, tuple[list[str], list[str]]] = {
        split: ([], []) for split in SPLITS
    }
    for stem, split in assigned.items():
        bucket = 1 if is_background[stem] else 0
        per_split[split][bucket].append(stem)
    for split in SPLITS:
        per_split[split][0].sort()
        per_split[split][1].sort()

    print("stratification: foreground/background images + per-class instances")
    for stratum in strata:
        total = stratum_totals[stratum]
        distribution = ", ".join(
            f"{split}={placed_images[split][stratum]}" for split in SPLITS
        )
        print(f"{stratum}: n={total} -> {distribution}")
    for class_id in sorted(class_totals):
        total = class_totals[class_id]
        distribution = ", ".join(
            f"{split}={placed_instances[split][class_id]} "
            f"({placed_instances[split][class_id] / total:.0%})"
            for split in SPLITS
        )
        print(f"class {class_id}: n={total} instances -> {distribution}")

    starved = [
        f"class {class_id} missing from {split}"
        for class_id in sorted(class_totals)
        for split in SPLITS
        if not placed_instances[split][class_id]
    ]
    if starved:
        print("warning: " + "; ".join(starved))

    return per_split


def reset_output_dirs(dry_run: bool) -> None:
    """Recreate the generated split directories unless this is a dry run."""
    for split in SPLITS:
        for kind in ("images", "labels"):
            target = DATA_DIR / split / kind
            if dry_run:
                continue
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)


def copy_sample(
    stem: str,
    split: str,
    image_index: dict[str, Path],
    label_index: dict[str, Path],
    dry_run: bool,
) -> bool:
    """Copy one matched image/label pair to its generated split."""
    image_path = image_index.get(stem)
    if image_path is None:
        print(f"skip: no image found for label {stem}.txt")
        return False

    label_path = label_index[stem]
    sample_type = "background" if label_path.stat().st_size == 0 else "labeled"
    print(f"copy [{sample_type}]: {image_path.name} + {label_path.name} -> {split}")
    if dry_run:
        return True

    shutil.copy2(image_path, DATA_DIR / split / "images" / image_path.name)
    shutil.copy2(label_path, DATA_DIR / split / "labels" / f"{stem}.txt")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match labels to archive images and prepare a class-balanced "
            "train/valid/test dataset."
        )
    )
    parser.add_argument(
        "--source",
        choices=("local", "roboflow"),
        default=SOURCE,
        help=(
            "'local' uses data/.draft/labels; 'roboflow' downloads the version "
            f"pinned in roboflow.yaml (default: {SOURCE})."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Calculate and print the split without changing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    export_dir: Path | None = None
    if args.source == "roboflow":
        export_dir = download_roboflow_export()

    if not SOURCE_IMAGES_DIR.is_dir():
        raise SystemExit(f"images directory not found: {SOURCE_IMAGES_DIR}")
    image_index = build_image_index(SOURCE_IMAGES_DIR)
    if not image_index:
        raise SystemExit(f"no source images found under {SOURCE_IMAGES_DIR}")

    if export_dir is not None:
        label_index = build_roboflow_label_index(export_dir)
    else:
        if not SOURCE_LABELS_DIR.is_dir():
            raise SystemExit(f"labels directory not found: {SOURCE_LABELS_DIR}")
        label_index = build_label_index(SOURCE_LABELS_DIR)
    if not label_index:
        raise SystemExit("no label files found; refusing to replace the current splits")

    matched = sorted(stem for stem in label_index if stem in image_index)
    missing = sorted(stem for stem in label_index if stem not in image_index)
    if missing:
        print(f"warning: {len(missing)} labels have no matching image (skipped)")

    per_split = stratified_split(matched, label_index)
    reset_output_dirs(args.dry_run)

    copied = {split: 0 for split in SPLITS}
    for split in SPLITS:
        foreground, background = per_split[split]
        for stem in foreground + background:
            if copy_sample(stem, split, image_index, label_index, args.dry_run):
                copied[split] += 1

    foreground_count = sum(
        label_index[stem].stat().st_size > 0 for stem in matched
    )
    background_count = len(matched) - foreground_count
    print(
        "Split complete: "
        f"labels={len(label_index)}, matched={len(matched)}, missing={len(missing)}, "
        f"foreground={foreground_count}, background={background_count}"
    )
    for split in SPLITS:
        foreground, background = per_split[split]
        print(
            f"{split}: labeled={len(foreground)}, background={len(background)}, "
            f"total={len(foreground) + len(background)}, copied={copied[split]}"
        )

    if export_dir is not None:
        if not args.dry_run:
            write_data_yaml(export_dir)
            save_leftover_labels(label_index, missing)
        shutil.rmtree(export_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
