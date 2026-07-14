from __future__ import annotations

import argparse
import ast
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

SPLITS = ("train", "valid", "val", "test", "draft")

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
ROBOFLOW_CONFIG = REPO_ROOT / "roboflow.yaml"

# Roboflow exports label files as `<num>_png.rf.<hash>.txt`; the numeric prefix
# is the original image id used to pair labels back to archive/ frames.
ROBOfLOW_LABEL_PATTERN = re.compile(r"^(\d+)_png\.rf\.[^.]+\.txt$")

# Default source when --source is not passed. "local" scans the label folders
# already on disk; "roboflow" downloads the dataset via the Roboflow API first.
SOURCE = "roboflow"  # "local" or "roboflow"


def roboflow_stem(name: str) -> str:
    """Numeric stem for a Roboflow label file (strips `_png.rf.<hash>`)."""
    match = ROBOfLOW_LABEL_PATTERN.match(name)
    return match.group(1) if match else Path(name).stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report YOLO segmentation labels that are stored as bounding boxes "
            "instead of polygons (mixed detect/segment dataset)."
        )
    )
    parser.add_argument(
        "--source",
        choices=["local", "roboflow"],
        default=SOURCE,
        help=(
            "Where the dataset comes from. 'local' scans --root as-is; "
            "'roboflow' downloads it first via the Roboflow API "
            f"(default: {SOURCE})."
        ),
    )
    parser.add_argument(
        "--root",
        default=str(DATA_DIR),
        help="Dataset root directory containing split folders (default: <repo>/data)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        help="Split folders to scan under --root (each must contain a labels/ directory).",
    )
    parser.add_argument(
        "--dest",
        default=str(REPO_ROOT / ".draft"),
        help="Download destination for --source roboflow (default: <repo>/.draft).",
    )
    parser.add_argument(
        "--env",
        default=str(REPO_ROOT / ".env"),
        help="Path to the .env file holding ROBOFLOW_API (default: <repo>/.env).",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Path to write the text report (default: <repo>/data/check_labels_report.txt).",
    )
    return parser.parse_args()


def load_env(env_path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE parser so we avoid a python-dotenv dependency."""
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        values[key.strip()] = val.strip().strip('"').strip("'")
    return values


def load_roboflow_config() -> dict[str, str]:
    """Parse the flat `key: value` coordinates from <repo>/roboflow.yaml."""
    cfg: dict[str, str] = {}
    if not ROBOFLOW_CONFIG.exists():
        return cfg
    for line in ROBOFLOW_CONFIG.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, val = stripped.split(":", 1)  # first colon only (values may contain ':')
        cfg[key.strip()] = val.strip()
    return cfg


def download_from_roboflow(dest: Path, api_key: str) -> Path:
    """Download the dataset named in <repo>/roboflow.yaml to `dest`, return its path."""
    cfg = load_roboflow_config()
    missing = [k for k in ("workspace", "project", "version") if not cfg.get(k)]
    if missing:
        raise SystemExit(
            f"cannot download from roboflow: missing {', '.join(missing)} in {ROBOFLOW_CONFIG}"
        )
    try:
        from roboflow import Roboflow  # lazy: only needed for --source roboflow
    except ImportError:
        raise SystemExit(
            "the 'roboflow' package is required for --source roboflow "
            "(pip install roboflow)"
        )

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(cfg["workspace"]).project(cfg["project"])
    version = project.version(int(cfg["version"]))

    # Reuse a complete prior download; otherwise pull fresh with overwrite=True so a
    # stale/partial dest can't make Roboflow silently skip the download.
    already = all((dest / s / "labels").is_dir() for s in ("train", "valid", "test"))
    if already:
        print(f"Reusing existing Roboflow download at {dest}")
        return dest

    print(
        f"Downloading {cfg['workspace']}/{cfg['project']} v{cfg['version']} "
        f"from Roboflow -> {dest}"
    )
    dataset = version.download("yolov8", location=str(dest), overwrite=True)
    return Path(dataset.location)


def load_class_names(root: Path) -> dict[int, str]:
    """Best-effort parse of the `names:` list from <root>/data.yaml.

    Handles both the inline form (`names: ['a', 'b']`) and the block form that
    Roboflow exports (`names:` followed by `- a` / `- b` lines).
    """
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        return {}
    lines = yaml_path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("names:"):
            continue
        rest = stripped.split("names:", 1)[1].strip()
        if rest.startswith("["):
            try:
                names = ast.literal_eval(rest)
                return {j: str(n) for j, n in enumerate(names)}
            except (ValueError, SyntaxError):
                return {}
        # Block form: collect following "- value" lines.
        names = []
        for follow in lines[i + 1:]:
            item = follow.strip()
            if item.startswith("- "):
                names.append(item[2:].strip().strip('"').strip("'"))
            elif item and not follow.startswith((" ", "\t")):
                break  # next top-level key
        return {j: n for j, n in enumerate(names)}
    return {}


@dataclass
class FileReport:
    path: Path
    polygons: int = 0
    box_only: int = 0
    malformed: int = 0
    box_classes: list[int] = field(default_factory=list)
    malformed_lines: list[str] = field(default_factory=list)

    @property
    def is_crash_trigger(self) -> bool:
        # A file with box-only rows but no polygon row is NOT treated as a segment
        # file by Ultralytics, so it contributes boxes with zero segments. That
        # single count mismatch makes Ultralytics strip masks from the whole
        # dataset, which then crashes YOLO26-seg on the first batch.
        return self.box_only > 0 and self.polygons == 0

    @property
    def has_issue(self) -> bool:
        return self.box_only > 0 or self.malformed > 0


def scan_file(label_file: Path) -> FileReport:
    report = FileReport(path=label_file)
    for raw_line in label_file.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if not parts:
            continue
        coords = len(parts) - 1  # values after the class id
        if coords >= 6 and coords % 2 == 0:
            report.polygons += 1  # valid polygon: class + >=3 xy pairs
        elif coords == 4:
            report.box_only += 1  # bounding box: class cx cy w h
            try:
                report.box_classes.append(int(float(parts[0])))
            except ValueError:
                report.box_classes.append(-1)
        else:
            report.malformed += 1  # not a box and not a valid polygon
            report.malformed_lines.append(raw_line.strip())
    return report


def write_report(report_path: str | None, lines: list[str]) -> None:
    """Write the collected report lines to a .txt file (defaults to the data folder)."""
    default = DATA_DIR / "check_labels_report.txt"
    dest = Path(report_path) if report_path else default
    dest = dest.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written to: {dest}")


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()

    if args.source == "roboflow":
        env = load_env(Path(args.env).resolve())
        api_key = env.get("ROBOFLOW_API")
        if not api_key:
            raise SystemExit(
                f"ROBOFLOW_API not found in {args.env} (needed for --source roboflow)"
            )
        # Coordinates come from <repo>/roboflow.yaml; the scanned root becomes
        # wherever the download lands.
        root = download_from_roboflow(Path(args.dest).resolve(), api_key).resolve()

    if not root.exists() or not root.is_dir():
        raise SystemExit(f"dataset root not found: {root}")

    # Collect every report line so we can print it and also write a .txt report.
    report_lines: list[str] = []

    def emit(line: str = "") -> None:
        print(line)
        report_lines.append(line)

    class_names = load_class_names(root)

    def cls_label(idx: int) -> str:
        return f"{idx} ({class_names[idx]})" if idx in class_names else str(idx)

    split_reports: dict[str, list[FileReport]] = {}
    missing_label_dirs: list[Path] = []

    for split in args.splits:
        labels_dir = root / split / "labels"
        if not labels_dir.exists() or not labels_dir.is_dir():
            missing_label_dirs.append(labels_dir)
            continue
        reports = [scan_file(f) for f in sorted(labels_dir.glob("*.txt"))]
        split_reports[split] = [r for r in reports if r.has_issue]

    for missing in missing_label_dirs:
        emit(f"warning: labels directory not found: {missing}")

    emit("=" * 72)
    emit(f"LABEL REPORT — box-only annotations in a segmentation dataset")
    emit(f"source: {args.source}   root: {root}")
    emit("=" * 72)

    total_box = 0
    total_files = 0
    total_malformed = 0
    crash_triggers: list[Path] = []
    class_counter: Counter[int] = Counter()

    for split, reports in split_reports.items():
        split_box = sum(r.box_only for r in reports)
        split_malformed = sum(r.malformed for r in reports)
        total_box += split_box
        total_files += len(reports)
        total_malformed += split_malformed

        emit(f"\n### {split} — {len(reports)} image(s), {split_box} box-only line(s)")
        for r in sorted(reports, key=lambda r: (not r.is_crash_trigger, r.path.name)):
            class_counter.update(r.box_classes)
            classes = " ".join(cls_label(c) for c in r.box_classes)
            flag = "  <== ALL-BOX / CRASH TRIGGER" if r.is_crash_trigger else ""
            parts = []
            if r.box_only:
                parts.append(f"{r.box_only} box-only")
            if r.malformed:
                parts.append(f"{r.malformed} malformed")
            detail = f" [classes: {classes}]" if classes else ""
            emit(f"  {r.path.stem}: {', '.join(parts)} line(s){detail}{flag}")
            if r.is_crash_trigger:
                crash_triggers.append(r.path)
            for bad in r.malformed_lines:
                emit(f"      malformed: {bad}")

    emit("\n" + "-" * 72)
    emit("SUMMARY")
    emit("-" * 72)
    emit(f"Affected images : {total_files}")
    emit(f"Box-only lines  : {total_box}")
    if total_malformed:
        emit(f"Malformed lines : {total_malformed}")
    if class_counter:
        emit("By class (box-only):")
        for idx in sorted(class_counter):
            emit(f"  {cls_label(idx)}: {class_counter[idx]}")

    if crash_triggers:
        emit(f"\nCRASH TRIGGER: {len(crash_triggers)} file(s) are entirely box-only "
             "(no polygon). Ultralytics will strip all segments and YOLO26-seg "
             "crashes on batch 0:")
        for path in crash_triggers:
            emit(f"  {path.relative_to(root)}")

    if total_box or total_malformed:
        emit("\nFix: re-annotate every box-only object as a polygon (in Roboflow or "
             "otherwise), then delete data/*/labels.cache and retrain.")
        write_report(args.report, report_lines)
        raise SystemExit(1)

    emit("\nAll labels are polygons. No box-only annotations found.")
    write_report(args.report, report_lines)


if __name__ == "__main__":
    main()
