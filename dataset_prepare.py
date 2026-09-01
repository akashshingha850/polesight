from __future__ import annotations

import argparse
import ast
import csv
import datetime
import json
import random
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
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
# current Roboflow version (see .env) and rebuilds the local dataset from it.
SOURCE = "roboflow"

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
SOURCE_LABELS_DIR = DATA_DIR / ".draft" / "labels"
ROBOFLOW_DOWNLOAD_DIR = DATA_DIR / ".draft" / "_roboflow"
# Written into the download so a stale export is never reused across a version bump.
ROBOFLOW_VERSION_STAMP = ".polesight_version"
ROBOFLOW_API_URL = "https://api.roboflow.com"
# Export coordinates live in .env, which is already required for the API key and
# is not committed. Nothing about the download is recorded outside it, so data/
# stays a pure build output that can be wiped and regenerated.
ROBOFLOW_PROJECT_DEFAULT = "polesight-7kgwj"
ROBOFLOW_WORKSPACE_DEFAULT = "polesight"
# An empty generation recipe on purpose: the split, the resize and every
# augmentation happen locally, so a Roboflow version must stay a plain snapshot
# of the annotated images. Anything set here would be baked in twice.
ROBOFLOW_VERSION_SETTINGS = {"preprocessing": {}, "augmentation": {}}
# Slack when comparing a version's creation time against the project's last
# modification: both are written by the same generate_version transaction and
# can land a few seconds apart in either order. Without it, every run would read
# its own fresh version as already stale and regenerate forever.
STALE_TOLERANCE_SECONDS = 120
PROJECT_STAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
VERSION_POLL_SECONDS = 10
VERSION_TIMEOUT_SECONDS = 1800
LEFTOVER_LABELS_DIR = ROOT_DIR / ".draft" / "leftover"
# ``leftover/`` keeps the unmatched label files themselves; this CSV is the
# readable index over them, naming the tags each missing image would carry.
MISSING_IMAGES_LOG = LEFTOVER_LABELS_DIR / "missing_images.csv"
MISSING_PREVIEW_LIMIT = 20
ENV_PATH = ROOT_DIR / ".env"

# archive/ is the only source of pixels. Roboflow decides *which* images are in
# the dataset and supplies the labels; the images themselves are always read from
# an intensity_filtered/ directory below archive/, at whatever depth a drive sits
# (archive/vt7/..., archive/alakylantie/University-tuira-isko/...). Directory
# names are not used as split groups.
#
# Only intensity_filtered/ is indexed, and this is the load-bearing rule of the
# whole file: a drive stores the same frame under the same stem in every
# rendering it produced — intensity/, range/, range_filtered/, rgb/, class/ — so
# a recursive scan does not have one image per stem, it has eight. Any rule other
# than "one named directory wins" pairs labels with whichever rendering the walk
# reached first. Everything outside intensity_filtered/ is skipped rather than
# guessed at, including archive/roboflow_backfill/: those images were recovered
# from Roboflow as 8-bit colour renders, and every one of them is also present as
# its 16-bit intensity_filtered/ original, which is the copy this dataset wants.
#
# Nothing here downloads pixels. A Roboflow export transcodes its images to
# 8-bit JPEG, and even the untouched upload can be an 8-bit colour render of a
# 16-bit greyscale capture, so pairing labels against Roboflow pixels would put
# part of the dataset in a different intensity domain from the rest. A label
# whose image is not under archive/ becomes a leftover (see MISSING_IMAGES_LOG)
# and is filed by hand rather than fetched.
SOURCE_IMAGES_DIR = ROOT_DIR / "archive"
ROUTE_IMAGES_SUBDIR = "intensity_filtered"
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
    """Roboflow export coordinates, read from .env with sane fallbacks."""
    env = load_env(ENV_PATH)
    return {
        "api_key": env.get("ROBOFLOW_API", ""),
        "workspace": env.get("ROBOFLOW_WORKSPACE", ROBOFLOW_WORKSPACE_DEFAULT),
        "project": env.get("ROBOFLOW_PROJECT", ROBOFLOW_PROJECT_DEFAULT),
        "version": env.get("ROBOFLOW_VERSION", ""),
    }


def set_env_value(path: Path, key: str, value: str) -> None:
    """Rewrite (or append) KEY=value in .env, leaving every other line alone."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    for index, line in enumerate(lines):
        if line.strip().startswith(f"{key}=") or line.strip().startswith(f"{key} ="):
            lines[index] = f"{key}={value}"
            break
    else:
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def project_stamp(updated: float) -> str:
    """Roboflow's epoch 'updated' field as a comparable, readable UTC stamp."""
    moment = datetime.datetime.fromtimestamp(float(updated), datetime.timezone.utc)
    return moment.strftime(PROJECT_STAMP_FORMAT)


def roboflow_get(path: str, api_key: str) -> dict:
    """GET one Roboflow REST endpoint and return the decoded JSON."""
    url = f"{ROBOFLOW_API_URL}/{path}?" + urllib.parse.urlencode({"api_key": api_key})
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:200].decode("utf-8", "replace")
        raise SystemExit(f"Roboflow API error on {path}: HTTP {exc.code} {detail}")
    except urllib.error.URLError as exc:
        raise SystemExit(f"cannot reach the Roboflow API: {exc.reason}")


def version_number(version_meta: dict) -> int:
    """'polesight/polesight-7kgwj/9' -> 9."""
    return int(str(version_meta.get("id", "")).rsplit("/", 1)[-1])


def fetch_project_state(config: dict[str, str], api_key: str) -> tuple[int, float, dict]:
    """Return (project image count, last-modified epoch, newest version metadata).

    ``updated`` moves whenever the project changes — including annotation edits
    that leave the image count untouched — so it is the signal that catches
    relabelled images.
    """
    data = roboflow_get(f"{config['workspace']}/{config['project']}", api_key)
    project = data.get("project", {})
    images = int(project.get("images", 0))
    updated = float(project.get("updated", 0) or 0.0)
    versions = [v for v in data.get("versions", []) if v.get("id")]
    newest = max(versions, key=version_number) if versions else {}
    return images, updated, newest


def version_staleness(meta: dict, images: int, updated: float) -> str | None:
    """Say why this version cannot represent the project, or None if it can.

    The version's own metadata carries every signal needed, so nothing has to be
    recorded on our side between runs: a stale version either covers a different
    number of images than the project does now, or predates the project's last
    modification (which is what catches an annotation-only edit — a relabel, or
    a class dropped from the ontology — that leaves the image count untouched).
    """
    if meta.get("augmentation"):
        return "was generated with augmentation enabled"
    if int(meta.get("images", -1)) != images:
        return f"covers {meta.get('images')} images, project now has {images}"
    created = float(meta.get("created") or 0.0)
    if updated > created + STALE_TOLERANCE_SECONDS:
        return (f"was generated {project_stamp(created)} but the project was "
                f"modified {project_stamp(updated)} (annotations or classes "
                f"changed since)")
    return None


def wait_for_version(config: dict[str, str], api_key: str, number: int) -> dict:
    """Block until Roboflow has finished generating the version."""
    deadline = time.monotonic() + VERSION_TIMEOUT_SECONDS
    path = f"{config['workspace']}/{config['project']}/{number}"
    while True:
        meta = roboflow_get(path, api_key).get("version", {})
        if not meta.get("generating") and float(meta.get("progress", 1)) >= 1:
            return meta
        if time.monotonic() > deadline:
            raise SystemExit(
                f"version {number} still generating after "
                f"{VERSION_TIMEOUT_SECONDS}s; re-run once Roboflow finishes"
            )
        print(f"  generating version {number}: "
              f"{float(meta.get('progress', 0)) * 100:.0f}%")
        time.sleep(VERSION_POLL_SECONDS)


def generate_roboflow_version(config: dict[str, str], api_key: str) -> int:
    """Ask Roboflow to cut a new version of the project and wait for it."""
    from roboflow import Roboflow

    client = Roboflow(api_key=api_key)
    project = client.workspace(config["workspace"]).project(config["project"])
    number = int(project.generate_version(settings=ROBOFLOW_VERSION_SETTINGS))
    meta = wait_for_version(config, api_key, number)
    set_env_value(ENV_PATH, "ROBOFLOW_VERSION", str(number))
    print(f"generated version {number} ({meta.get('images', '?')} images); "
          f"pinned ROBOFLOW_VERSION={number} in {relative_to_root(ENV_PATH)}")
    return number


def resolve_roboflow_version(
    config: dict[str, str], api_key: str, pin: bool, dry_run: bool
) -> int:
    """Decide which version to download, generating one when the data moved on.

    The decision rests entirely on what Roboflow reports about the project and
    its newest version (see ``version_staleness``), so nothing has to be tracked
    between runs and there is no local baseline that can go missing or stale.
    ``--pin-version`` opts out, reusing ROBOFLOW_VERSION from .env for a
    reproducible rebuild of an exact past version.
    """
    pinned = config.get("version") or ""
    if pin:
        if not pinned:
            raise SystemExit(
                f"--pin-version needs ROBOFLOW_VERSION set in {relative_to_root(ENV_PATH)}"
            )
        print(f"pinned to Roboflow version {pinned} (--pin-version)")
        return int(pinned)

    images, updated, newest = fetch_project_state(config, api_key)
    if not newest:
        raise SystemExit("no versions listed for the Roboflow project")
    latest = version_number(newest)
    print(f"Roboflow project: {images} images, modified {project_stamp(updated)} | "
          f"newest version {latest}: {newest.get('images')} images, "
          f"generated {project_stamp(float(newest.get('created') or 0.0))}")

    reason = version_staleness(newest, images, updated)
    if reason is None:
        if str(latest) != pinned and not dry_run:
            set_env_value(ENV_PATH, "ROBOFLOW_VERSION", str(latest))
            print(f"pinned ROBOFLOW_VERSION={latest} in {relative_to_root(ENV_PATH)}")
        return latest

    print(f"version {latest} {reason}")
    if dry_run:
        print(f"dry run: would generate a new version; using version {latest}")
        return latest

    return generate_roboflow_version(config, api_key)


def download_from_roboflow(destination: Path, api_key: str, version: int) -> Path:
    """Download or reuse the raw YOLO export for one version."""
    config = load_roboflow_config()
    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise SystemExit(
            "the 'roboflow' package is required for --source roboflow "
            "(pip install roboflow)"
        ) from exc

    stamp = destination / ROBOFLOW_VERSION_STAMP
    complete = all((destination / split / "labels").is_dir() for split in SPLITS)
    downloaded = stamp.read_text(encoding="utf-8").strip() if stamp.exists() else ""
    if complete and downloaded == str(version):
        print(f"reusing Roboflow v{version} download at {destination}")
        return destination

    # Purge before downloading a different version. Roboflow's ``overwrite=True``
    # replaces files it writes but leaves behind any it does not, and every label
    # carries a version-specific ``.rf.<hash>`` in its name — so a previous
    # version's labels survive alongside the new ones and surface later as
    # duplicate stems, one carrying the old class indexing.
    if destination.exists():
        if downloaded:
            print(f"clearing stale v{downloaded} download at {relative_to_root(destination)}")
        shutil.rmtree(destination, ignore_errors=True)

    print(
        f"downloading {config['workspace']}/{config['project']} "
        f"v{version} -> {destination}"
    )
    client = Roboflow(api_key=api_key)
    project = client.workspace(config["workspace"]).project(config["project"])
    dataset = project.version(version).download(
        "yolov8", location=str(destination), overwrite=True
    )
    location = Path(dataset.location)
    (location / ROBOFLOW_VERSION_STAMP).write_text(f"{version}\n", encoding="utf-8")
    return location


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


def download_roboflow_export(pin: bool, dry_run: bool) -> Path:
    """Resolve the Roboflow version for this run and download its export."""
    config = load_roboflow_config()
    api_key = config["api_key"]
    missing = [
        f"ROBOFLOW_{key.upper()}" for key in ("api_key", "workspace", "project")
        if not config.get(key)
    ]
    if missing:
        raise SystemExit(
            f"cannot download from Roboflow: set {', '.join(missing)} in "
            f"{relative_to_root(ENV_PATH)}"
        )
    version = resolve_roboflow_version(config, api_key, pin, dry_run)
    return download_from_roboflow(ROBOFLOW_DOWNLOAD_DIR, api_key, version)


def write_data_yaml(export_dir: Path) -> None:
    """Regenerate data/data.yaml from the export's class names + .env config."""
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
    for key in ("workspace", "project", "version"):
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


def save_leftover_labels(
    label_index: dict[str, Path], missing: list[str], dry_run: bool
) -> None:
    """Keep unmatched labels for inspection, replacing leftovers from the last run.

    Called on every run, including one where nothing is missing: a leftover is
    only meaningful beside the run that produced it, so the previous run's copies
    are cleared first — the same reason save_missing_image_log deletes a stale
    log rather than leaving it to misreport a clean run. When the labels came
    from a Roboflow export this is also the only copy that survives, since the
    export directory is deleted once the split is written.
    """
    if dry_run:
        return
    LEFTOVER_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    for stale in LEFTOVER_LABELS_DIR.glob("*.txt"):
        stale.unlink()
    for stem in missing:
        shutil.copy2(label_index[stem], LEFTOVER_LABELS_DIR / f"{stem}.txt")
    if missing:
        print(f"leftover labels: {len(missing)} -> "
              f"{relative_to_root(LEFTOVER_LABELS_DIR)}")


def relative_to_root(path: Path) -> str:
    """Path relative to the repository root when it lives inside it."""
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def describe_tags(counts: Counter[int], class_names: dict[int, str]) -> str:
    """Render one label's class counts as "light_pole x2; fence_pole x1"."""
    if not counts:
        return "background"
    return "; ".join(
        f"{class_names.get(class_id, f'class_{class_id}')} x{count}"
        for class_id, count in sorted(counts.items())
    )


def save_missing_image_log(
    label_index: dict[str, Path],
    missing: list[str],
    class_names: dict[int, str],
    dry_run: bool,
) -> None:
    """Log every label whose image was not found, together with its tags.

    A bare count is not enough to go looking for the images: each row names the
    stem, how many instances the label holds and which classes they are, so a
    capture that never made it into ``archive/`` can be traced back to what it
    would have contributed to the split.
    """
    if not missing:
        if not dry_run and MISSING_IMAGES_LOG.exists():
            MISSING_IMAGES_LOG.unlink()  # a stale log would misreport a clean run
        return

    rows = []
    lost: Counter[int] = Counter()
    for stem in missing:
        label_path = label_index[stem]
        counts = class_counts(label_path)
        lost.update(counts)
        rows.append((
            stem,
            sum(counts.values()),
            describe_tags(counts, class_names),
            relative_to_root(label_path),
        ))

    print(
        f"warning: {len(rows)} labels have no matching image under "
        f"{relative_to_root(SOURCE_IMAGES_DIR)} (skipped)"
    )
    for stem, total, tags, _ in rows[:MISSING_PREVIEW_LIMIT]:
        unit = "instance" if total == 1 else "instances"
        print(f"  not found: {stem} -> {total} {unit} [{tags}]")
    if len(rows) > MISSING_PREVIEW_LIMIT:
        print(f"  ... and {len(rows) - MISSING_PREVIEW_LIMIT} more")
    if lost:
        summary = ", ".join(
            f"{class_names.get(class_id, f'class_{class_id}')}={count}"
            for class_id, count in sorted(lost.items())
        )
        print(f"  instances lost with them: {summary}")
    background = sum(1 for _, total, _, _ in rows if total == 0)
    if background:
        print(f"  ({background} of them are background labels)")

    if dry_run:
        print(f"dry run: not writing {relative_to_root(MISSING_IMAGES_LOG)}")
        return

    MISSING_IMAGES_LOG.parent.mkdir(parents=True, exist_ok=True)
    with MISSING_IMAGES_LOG.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("stem", "instances", "tags", "label_source"))
        writer.writerows(rows)
    print(f"missing-image log: {len(rows)} -> {relative_to_root(MISSING_IMAGES_LOG)}")


def is_source_image(path: Path, images_dir: Path) -> bool:
    """True for an image this dataset is allowed to take pixels from.

    See the ROUTE_IMAGES_SUBDIR comment above: only a drive's intensity_filtered/
    rendering counts, wherever that directory sits below images_dir.
    """
    if path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file():
        return False
    return ROUTE_IMAGES_SUBDIR in path.relative_to(images_dir).parts[:-1]


def source_route(path: Path, images_dir: Path) -> str:
    """The drive an indexed image was captured on, e.g. 'alakylantie/isko-university'.

    Everything above intensity_filtered/ is the drive's identity, so the route is
    read from the path rather than assumed to be a single directory level: drives
    are nested one deep under a road name in the newer captures and sit directly
    below archive/ in the older ones.
    """
    parts = path.relative_to(images_dir).parts
    return "/".join(parts[: parts.index(ROUTE_IMAGES_SUBDIR)])


def build_image_index(images_dir: Path) -> dict[str, Path]:
    """Recursively index source images, failing on ambiguous duplicate stems."""
    image_index: dict[str, Path] = {}
    skipped = 0
    for image_path in sorted(images_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if not is_source_image(image_path, images_dir):
            skipped += 1
            continue
        stem = image_path.stem
        existing = image_index.get(stem)
        if existing is not None:
            raise SystemExit(
                f"duplicate image stem '{stem}': {existing} and {image_path}"
            )
        image_index[stem] = image_path
    if skipped:
        # Loud on purpose: a route rendering that is not intensity_filtered/ is
        # the one thing that could quietly change which pixels a label refers to.
        print(f"images: skipped {skipped} outside {ROUTE_IMAGES_SUBDIR}/ "
              f"under {relative_to_root(images_dir)}")
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


# A Roboflow project can hold detection-style annotations beside polygons: a
# label line with five fields is ``class xc yc w h``, not a segment. Ultralytics
# only reads a file as segmentation when some line has more than six fields (see
# verify_image_label in ultralytics/data/utils.py), so a file holding nothing but
# box lines is parsed as detection and then crashes the segmentation Format
# transform on an empty class tensor. In a mixed file the box survives as a
# two-point polygon whose rasterised mask is empty, which is worse than the
# crash: the instance still counts as mask ground truth while teaching the mask
# head nothing. Writing the four corners the box already describes fixes both,
# at the cost of those instances carrying an axis-aligned rectangle mask.
BOX_FIELD_COUNT = 5


def polygonize_boxes(text: str) -> tuple[str, int]:
    """Rewrite every box-style line in a label as its four-corner polygon."""
    lines: list[str] = []
    converted = 0
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != BOX_FIELD_COUNT:
            lines.append(line)
            continue
        class_id = fields[0]
        x, y, width, height = (float(value) for value in fields[1:])
        left, right = x - width / 2, x + width / 2
        top, bottom = y - height / 2, y + height / 2
        corners = ((left, top), (right, top), (right, bottom), (left, bottom))
        coordinates = " ".join(
            f"{min(max(value, 0.0), 1.0):.6g}"
            for corner in corners
            for value in corner
        )
        lines.append(f"{class_id} {coordinates}")
        converted += 1
    return ("\n".join(lines) + "\n" if lines else ""), converted


def report_box_labels(label_index: dict[str, Path], matched: list[str]) -> None:
    """Report how many instances are written as rectangles rather than polygons.

    Printed every run because it is a quality caveat on the labels themselves,
    not a property of this code: a report quoting mask metrics needs to know
    which fraction of the ground truth is a rectangle.
    """
    files = instances = box_only = 0
    for stem in matched:
        text = label_index[stem].read_text(encoding="utf-8")
        _, converted = polygonize_boxes(text)
        if not converted:
            continue
        files += 1
        instances += converted
        if converted == len([line for line in text.splitlines() if line.strip()]):
            box_only += 1
    if instances:
        print(
            f"box-style labels: {instances} instance(s) in {files} file(s) written "
            f"as rectangle polygons ({box_only} file(s) held no polygon at all)"
        )


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


def needs_rgb_conversion(path: Path) -> bool:
    """True if this image is not already an 8-bit, 3-channel file."""
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.mode != "RGB"
    except Exception:
        return False


def write_as_rgb8(source: Path, destination: Path) -> None:
    """Write source as an 8-bit RGB PNG, pixel-for-pixel as the trainer loads it.

    Ultralytics reads every image with ``cv2.IMREAD_COLOR`` (see
    ``BaseDataset.load_image``), which maps a 16-bit single-channel PNG to 8-bit
    across three channels — for these captures a plain divide by 256. Applying
    exactly that here changes nothing the model sees; it only settles the
    conversion on disk so the dataset is one uniform format instead of two.
    """
    import cv2

    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read image for RGB conversion: {source}")
    if not cv2.imwrite(str(destination), image):
        raise SystemExit(f"cannot write converted image: {destination}")


def copy_sample(
    stem: str,
    split: str,
    image_index: dict[str, Path],
    label_index: dict[str, Path],
    dry_run: bool,
    to_rgb: bool = False,
) -> bool:
    """Copy one matched image/label pair to its generated split."""
    image_path = image_index.get(stem)
    if image_path is None:
        print(f"skip: no image found for label {stem}.txt")
        return False

    label_path = label_index[stem]
    sample_type = "background" if label_path.stat().st_size == 0 else "labeled"
    # Only the odd ones out are re-encoded; an image already 8-bit RGB is copied
    # byte-for-byte so the untouched original survives wherever it can.
    convert = to_rgb and needs_rgb_conversion(image_path)
    action = "convert" if convert else "copy"
    print(f"{action} [{sample_type}]: {image_path.name} + {label_path.name} -> {split}")
    if dry_run:
        return True

    if convert:
        write_as_rgb8(image_path, DATA_DIR / split / "images" / f"{stem}.png")
    else:
        shutil.copy2(image_path, DATA_DIR / split / "images" / image_path.name)
    # Written rather than copied: box-style lines become rectangle polygons on
    # the way out, so the prepared split is always segmentation-shaped.
    label_text, _ = polygonize_boxes(label_path.read_text(encoding="utf-8"))
    (DATA_DIR / split / "labels" / f"{stem}.txt").write_text(
        label_text, encoding="utf-8"
    )
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match labels to the intensity_filtered/ images under archive/ and "
            "prepare a class-balanced train/valid/test dataset."
        )
    )
    parser.add_argument(
        "--source",
        choices=("local", "roboflow"),
        default=SOURCE,
        help=(
            "'local' uses data/.draft/labels; 'roboflow' checks the project for "
            "new images, generates a version when there are any, and downloads it "
            f"(default: {SOURCE})."
        ),
    )
    parser.add_argument(
        "--pin-version",
        action="store_true",
        help=(
            "Download exactly the version in ROBOFLOW_VERSION (.env). Skips the "
            "new-data check, so no version is generated: use it to rebuild a past "
            "dataset reproducibly."
        ),
    )
    parser.add_argument(
        "--rgb",
        action="store_true",
        help=(
            "Write every image as an 8-bit RGB PNG, converting the 16-bit "
            "greyscale captures with the same mapping the trainer applies. The "
            "pixels the model sees are unchanged; the dataset just stops being "
            "two formats."
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
        export_dir = download_roboflow_export(args.pin_version, args.dry_run)

    if export_dir is not None:
        label_index = build_roboflow_label_index(export_dir)
    else:
        if not SOURCE_LABELS_DIR.is_dir():
            raise SystemExit(f"labels directory not found: {SOURCE_LABELS_DIR}")
        label_index = build_label_index(SOURCE_LABELS_DIR)
    if not label_index:
        raise SystemExit("no label files found; refusing to replace the current splits")

    # The Roboflow project defines *which* images are in the dataset; archive/
    # supplies the pixels, whichever source the labels came from. Nothing is
    # fetched to cover a gap: a label with no image under archive/ is reported by
    # save_missing_image_log and kept as a leftover, so every prepared image is
    # the capture as recorded rather than a Roboflow re-encode of it.
    if not SOURCE_IMAGES_DIR.is_dir():
        raise SystemExit(f"images directory not found: {SOURCE_IMAGES_DIR}")
    image_index = build_image_index(SOURCE_IMAGES_DIR)
    if not image_index:
        raise SystemExit("no source images resolved; refusing to replace the current splits")
    resolved = sum(stem in image_index for stem in label_index)
    print(f"images: {resolved}/{len(label_index)} labels resolved from "
          f"{relative_to_root(SOURCE_IMAGES_DIR)}")

    matched = sorted(stem for stem in label_index if stem in image_index)
    missing = sorted(stem for stem in label_index if stem not in image_index)

    class_names = load_class_names(export_dir if export_dir is not None else DATA_DIR)
    report_box_labels(label_index, matched)
    save_missing_image_log(label_index, missing, class_names, args.dry_run)
    save_leftover_labels(label_index, missing, args.dry_run)

    per_split = stratified_split(matched, label_index)
    reset_output_dirs(args.dry_run)

    copied = {split: 0 for split in SPLITS}
    for split in SPLITS:
        foreground, background = per_split[split]
        for stem in foreground + background:
            if copy_sample(stem, split, image_index, label_index, args.dry_run, args.rgb):
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
        shutil.rmtree(export_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
