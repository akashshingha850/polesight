"""PoleSight instance-segmentation training configured by ``train.yaml``.

Usage:
    python train.py                             # every family in train.yaml
    python train.py yolo26                      # one configured family
    python train.py --eval                      # evaluate existing checkpoints
    python train.py --config train_default.yaml # a different sweep definition

Per-epoch model selection stays on the validation split. After training, the
saved best checkpoint is evaluated separately on the configured final splits
(test by default), through both YOLO26 heads when available.
"""

from __future__ import annotations

import os

# PyTorch reads this when the CUDA allocator initializes, so it must precede
# imports that load torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import csv
import gc
import json
import sys
from pathlib import Path

import torch
import wandb
import yaml
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import unwrap_model

ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "train.yaml"

# Additional segmentation-capable families can be enabled in train.yaml later.
WRAPPERS = {"yolo26": YOLO, "yolo11": YOLO, "yolov8": YOLO}

# These settings are consumed by this wrapper rather than forwarded to
# Ultralytics model.train().
_CONSUMED = {"data", "project", "device", "batch", "min_batch", "eval_splits"}
_TRAIN_ONLY = {"progloss_final_o2m", "primary_head", "train_end2end"}
_PROGLOSS: list[tuple[int, float, float]] = []


def load_config(path: str | Path | None = None) -> dict:
    """Load configuration and resolve data/project paths relative to this file."""
    config_path = Path(path) if path else CONFIG_PATH
    if not config_path.exists():
        raise SystemExit(f"config not found: {config_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    common = config.setdefault("common", {})
    for key in ("data", "project"):
        if common.get(key) and not os.path.isabs(str(common[key])):
            common[key] = str(ROOT_DIR / str(common[key]))

    unknown = [family for family in config.get("families", {}) if family not in WRAPPERS]
    if unknown:
        raise SystemExit(
            f"{config_path.name}: no model wrapper for {unknown}; "
            f"supported families are {list(WRAPPERS)}"
        )
    if not config.get("families"):
        raise SystemExit(f"{config_path.name}: no model families configured")
    return config


# ---------------------------------------------------------------------------
# Training callbacks
# ---------------------------------------------------------------------------

def _progloss_setter(final_o2m: float):
    """Return a callback that retargets YOLO26's one-to-many loss ramp."""

    def _set(trainer) -> None:
        criterion = getattr(unwrap_model(trainer.model), "criterion", None)
        if criterion is None or not hasattr(criterion, "final_o2m"):
            return
        if getattr(criterion, "_polesight_final_o2m", None) == final_o2m:
            return
        criterion.final_o2m = criterion._polesight_final_o2m = float(final_o2m)
        criterion.o2m = criterion.decay(criterion.updates)
        criterion.o2o = max(criterion.total - criterion.o2m, 0)
        LOGGER.info(
            f"[progloss] final_o2m={criterion.final_o2m} "
            f"o2m={criterion.o2m:.3f} o2o={criterion.o2o:.3f}"
        )

    return _set


def _record_progloss(trainer) -> None:
    """Record the one-to-many/one-to-one weights used for each epoch."""
    criterion = getattr(unwrap_model(trainer.model), "criterion", None)
    if criterion is not None and hasattr(criterion, "o2o"):
        _PROGLOSS.append(
            (int(trainer.epoch) + 1, float(criterion.o2m), float(criterion.o2o))
        )


def _verify(trainer) -> None:
    """Log actual trainer arguments and optimizer details at startup."""
    if wandb.run is not None:
        wandb.run.config.update(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(trainer.args).items()
            },
            allow_val_change=True,
        )

    learning_rates = [group["lr"] for group in trainer.optimizer.param_groups]
    base = min(learning_rates) if learning_rates else 0.0
    boosted = [
        index
        for index, learning_rate in enumerate(learning_rates)
        if base and round(learning_rate / base, 2) == 3.0
    ]
    parameters = sum(
        len(trainer.optimizer.param_groups[index]["params"]) for index in boosted
    )
    high = max(learning_rates) if learning_rates else 0.0
    LOGGER.info(
        f"[verify] optimizer={type(trainer.optimizer).__name__} "
        f"groups={len(learning_rates)} lr={base:g}..{high:g} "
        f"head-boost={len(boosted)} groups/{parameters} params"
    )


def _summary(trainer) -> None:
    """Write an auditable summary next to each run's weights."""
    stopper = getattr(trainer, "stopper", None)
    best_epoch = int(getattr(stopper, "best_epoch", -1))
    epochs_run = int(getattr(trainer, "epoch", -1)) + 1
    summary = {
        "name": Path(trainer.save_dir).name,
        "model": str(trainer.args.model),
        "epochs_run": epochs_run,
        "epochs_configured": int(trainer.args.epochs),
        "early_stopped": epochs_run < int(trainer.args.epochs),
        "best_epoch": best_epoch,
        "batch_actual": int(trainer.batch_size),
        "optimizer_actual": type(trainer.optimizer).__name__,
        "end2end_actual": bool(getattr(unwrap_model(trainer.model), "end2end", False)),
        "mosaic": float(trainer.args.mosaic),
        "close_mosaic": int(trainer.args.close_mosaic),
        "progloss_o2o_at_best": next(
            (o2o for epoch, _, o2o in _PROGLOSS if epoch == best_epoch), None
        ),
        "progloss_o2o_final": _PROGLOSS[-1][2] if _PROGLOSS else None,
    }
    output = Path(trainer.save_dir) / "verify.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    LOGGER.info(
        f"[verify] {epochs_run}/{summary['epochs_configured']} epochs, "
        f"best={best_epoch}, batch={summary['batch_actual']}, "
        f"optimizer={summary['optimizer_actual']}"
    )


# ---------------------------------------------------------------------------
# Final checkpoint evaluation
# ---------------------------------------------------------------------------

def _metric_block(metric, names: dict[int, str]) -> dict:
    """Serialize aggregate and per-class values from one box or mask metric."""
    per_class = {}
    for index, class_id in enumerate(metric.ap_class_index):
        class_id = int(class_id)
        per_class[str(names[class_id])] = {
            "precision": float(metric.p[index]),
            "recall": float(metric.r[index]),
            "mAP50": float(metric.ap50[index]),
            "mAP50-95": float(metric.ap[index]),
        }
    return {
        "mAP50-95": float(metric.map),
        "mAP50": float(metric.map50),
        "precision": float(metric.mp),
        "recall": float(metric.mr),
        "per_class": per_class,
    }


def _metrics(result) -> dict:
    """Serialize both detection-box and instance-mask validation metrics."""
    metrics = {
        "box": _metric_block(result.box, result.names),
        "speed_ms": {key: float(value) for key, value in result.speed.items()},
    }
    if getattr(result, "seg", None) is not None:
        metrics["mask"] = _metric_block(result.seg, result.names)
    return metrics


def _release() -> None:
    """Release model memory between variants, retries, and evaluation heads."""
    torch.cuda.empty_cache()
    gc.collect()


def _eval_checkpoint(
    weights: str | Path | None,
    name: str,
    output_root: str | Path,
    data: str | Path,
    device,
    splits=("test",),
    primary: str | None = None,
) -> dict | None:
    """Evaluate best.pt on final splits through every available YOLO26 head."""
    weights_path = Path(weights) if weights else None
    if not weights_path or not weights_path.exists():
        LOGGER.warning(f"evaluation skipped: no weights for {name}")
        return None

    probe = YOLO(str(weights_path))
    end2end = bool(getattr(probe.model.model[-1], "end2end", False))
    # model.info() only prints in this Ultralytics version and returns None, so
    # the cost figures are read straight off the module.
    try:
        from ultralytics.utils.torch_utils import get_flops, get_num_params

        layers = len(list(probe.model.modules()))
        parameters = int(get_num_params(probe.model))
        gflops = round(float(get_flops(probe.model, 640)), 2)
    except Exception:
        layers = parameters = gflops = None
    del probe
    heads = (
        (("one2one", True), ("one2many", False))
        if end2end
        else (("default", None),)
    )

    report = {
        "model": name,
        "weights": str(weights_path),
        "task": "segment",
        "end2end": end2end,
        # Cost side of the accuracy/cost table a comparison report needs.
        "layers": layers,
        "parameters": parameters,
        "gflops": gflops,
        "weights_mb": round(weights_path.stat().st_size / 1024**2, 2),
        "splits": {},
    }
    for split in splits:
        scored = {}
        for head_name, end2end_value in heads:
            model = YOLO(str(weights_path))
            if end2end_value is not None:
                model.model.model[-1].end2end = end2end_value
            result = model.val(
                data=str(data),
                split=split,
                device=device,
                plots=False,
                verbose=False,
                project=Path(output_root) / name,
                name=f"{name}-{split}-{head_name}",
                exist_ok=True,
            )
            scored[head_name] = _metrics(result)
            del model
            _release()

        report["splits"][split] = scored
        primary_head = primary if primary in scored else next(iter(scored))
        report["primary_head"] = primary_head
        chosen = scored[primary_head]
        mask = chosen.get("mask", chosen["box"])
        box = chosen["box"]
        LOGGER.info(
            f"[eval] {name} {split} [{primary_head}] "
            f"mask mAP50={mask['mAP50']:.4f} mAP50-95={mask['mAP50-95']:.4f}; "
            f"box mAP50={box['mAP50']:.4f} mAP50-95={box['mAP50-95']:.4f}"
        )

    output = Path(output_root) / name / "eval.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if wandb.run is not None:
        for split, split_results in report["splits"].items():
            for head, head_results in split_results.items():
                for kind in ("box", "mask"):
                    for metric, value in head_results.get(kind, {}).items():
                        if isinstance(value, (int, float)):
                            wandb.run.summary[f"{split}/{head}/{kind}/{metric}"] = value
    return report


# ---------------------------------------------------------------------------
# Training and OOM recovery
# ---------------------------------------------------------------------------

_OOM_MESSAGES = (
    "out of memory",
    "cublas_status_alloc_failed",
    "cudnn_status_internal_error",
    "unable to find an engine",
)


def _is_oom(error: Exception) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or any(
        message in str(error).lower() for message in _OOM_MESSAGES
    )


def _run(
    family: str,
    variant: str,
    batch: int,
    config: dict,
    arm: str = "",
    arm_overrides: dict | None = None,
) -> None:
    """Train one configured variant and evaluate its best checkpoint."""
    common = config["common"]
    family_config = config["families"][family]
    wandb_config = config.get("wandb", {})
    stem = family_config["stem"].format(v=variant)
    name = f"{stem}-{arm}" if arm else stem
    output_root = Path(common["project"])

    kwargs = {key: value for key, value in common.items() if key not in _CONSUMED}
    kwargs.update(family_config.get("overrides") or {})
    kwargs.update(arm_overrides or {})
    extras = {key: kwargs.pop(key) for key in list(kwargs) if key in _TRAIN_ONLY}
    kwargs = {key: value for key, value in kwargs.items() if value is not None}

    with wandb.init(
        project=wandb_config.get("project", "PoleSight"),
        name=f"{name}-{batch}",
        group=family,
        job_type=wandb_config.get("job_type", "recipe"),
        tags=[family, variant] + ([arm] if arm else []),
        config={"arm": arm or "default", **extras},
    ):
        _PROGLOSS.clear()
        model = WRAPPERS[family](f"{stem}.pt")
        if extras.get("train_end2end") is not None:
            # `end2end` in Ultralytics' args affects validation/prediction only.
            # model.train() rebuilds the network from model.yaml, so update both
            # the live head and that source YAML before the criterion is built.
            # The NMS arm then uses v8SegmentationLoss while the NMS-free arm uses
            # E2ELoss.
            train_end2end = bool(extras["train_end2end"])
            model.model.yaml["end2end"] = train_end2end
            model.model.end2end = train_end2end
            LOGGER.info(
                f"[model] train_end2end={model.model.end2end} "
                f"({'NMS-free' if model.model.end2end else 'with NMS'})"
            )
        model.add_callback("on_train_start", _verify)
        if extras.get("progloss_final_o2m") is not None:
            model.add_callback(
                "on_train_epoch_start",
                _progloss_setter(extras["progloss_final_o2m"]),
            )
        model.add_callback("on_train_epoch_start", _record_progloss)
        model.add_callback("on_train_end", _summary)
        model.train(
            data=common["data"],
            project=output_root / name,
            name=name,
            device=common["device"],
            batch=batch,
            **kwargs,
        )
        _eval_checkpoint(
            getattr(model.trainer, "best", None),
            name,
            output_root,
            common["data"],
            common["device"],
            splits=common.get("eval_splits") or ("test",),
            primary=extras.get("primary_head") or common.get("primary_head"),
        )
        del model


def train(
    family: str,
    variant: str,
    config: dict,
    arm: str = "",
    arm_overrides: dict | None = None,
) -> None:
    """Train one variant, halving its batch size and retrying after CUDA OOM."""
    common = config["common"]
    stem = config["families"][family]["stem"].format(v=variant)
    batch = int(common["batch"])
    minimum = int(common.get("min_batch", 4))
    while True:
        try:
            return _run(family, variant, batch, config, arm, arm_overrides)
        except Exception as error:
            _release()
            if not _is_oom(error) or batch <= minimum:
                raise
            new_batch = max(minimum, batch // 2)
            LOGGER.warning(f"{stem}: CUDA OOM at batch {batch}; retrying at {new_batch}")
            batch = new_batch


# ---------------------------------------------------------------------------
# Evaluation-only mode
# ---------------------------------------------------------------------------

def _find_checkpoint(output_root: str | Path, name: str) -> tuple[Path | None, int]:
    """Select the completed run with the most epochs for one model name."""
    candidates = []
    for directory in sorted(Path(output_root).glob(f"{name}/{name}*")):
        best = directory / "weights" / "best.pt"
        results = directory / "results.csv"
        if best.exists() and results.exists():
            epochs = max(
                sum(1 for line in results.open(encoding="utf-8") if line.strip()) - 1,
                0,
            )
            candidates.append((epochs, -best.stat().st_mtime, best))
    if not candidates:
        return None, 0
    epochs, _, best = max(candidates)
    return best, epochs


def _environment() -> dict:
    """Versions and hardware, so a result can be reproduced or explained later."""
    import platform

    import ultralytics

    device = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        device = {
            "name": properties.name,
            "memory_gb": round(properties.total_memory / 1024**3, 1),
            "capability": f"{properties.major}.{properties.minor}",
        }
    return {
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "gpu": device,
    }


def _dataset_summary(data: str | Path) -> dict:
    """Class names and split sizes for the dataset a sweep was scored on."""
    data_path = Path(data)
    summary: dict = {"data_yaml": str(data_path)}
    try:
        spec = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    except OSError:
        return summary
    summary["classes"] = spec.get("names")
    summary["images"] = {
        split: len(list((data_path.parent / split / "images").iterdir()))
        for split in ("train", "valid", "test")
        if (data_path.parent / split / "images").is_dir()
    }
    return summary


def _training_record(weights: Path | None) -> dict:
    """What the run itself reported: verify.json plus wall time from results.csv.

    Read back from disk rather than kept in memory because --eval is expected to
    run long after training, including on runs from an earlier session.
    """
    record: dict = {}
    if weights is None:
        return record
    run_dir = weights.parent.parent
    verify = run_dir / "verify.json"
    if verify.exists():
        try:
            record.update(json.loads(verify.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            pass
    results = run_dir / "results.csv"
    if results.exists():
        with results.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        # Ultralytics' `time` column is cumulative seconds since the run started,
        # so the final row is the run's wall time.
        elapsed = rows[-1].get("time") if rows else None
        if elapsed:
            record["train_seconds"] = round(float(elapsed), 1)
            record["train_hours"] = round(float(elapsed) / 3600, 3)
    record["run_dir"] = str(run_dir)
    return record


def _write_json_report(
    output_root: Path, config: dict, records: list[dict], missing: list[str]
) -> Path:
    """Write the machine-readable sweep result used as report material."""
    import datetime

    report = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "project": str(output_root),
        "environment": _environment(),
        "dataset": _dataset_summary(config["common"]["data"]),
        "config": {
            "common": config["common"],
            "families": {
                family: {
                    "variants": spec["variants"],
                    "overrides": spec.get("overrides") or {},
                    "arms": list((spec.get("arms") or {})) or None,
                }
                for family, spec in config["families"].items()
            },
        },
        "runs": records,
        "incomplete": missing,
    }
    output = output_root / "results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    LOGGER.info(f"[eval] {len(records)} run(s) -> {output}")
    return output


def eval_existing(config: dict, selected: list[str]) -> None:
    """Evaluate existing checkpoints and write a box/mask comparison table."""
    common = config["common"]
    families = config["families"]
    output_root = Path(common["project"])
    splits = list(common.get("eval_splits") or ("test",))
    rows = []
    records = []
    missing = []

    for family in selected:
        family_config = families[family]
        expected_primary = {}
        for variant in family_config["variants"]:
            base = family_config["stem"].format(v=variant)
            for arm, overrides in (family_config.get("arms") or {"": {}}).items():
                run_name = f"{base}-{arm}" if arm else base
                expected_primary[run_name] = (overrides or {}).get("primary_head")
        names = sorted(
            {
                directory.name
                for base in (
                    family_config["stem"].format(v=variant)
                    for variant in family_config["variants"]
                )
                for directory in output_root.glob(f"{base}*")
                if directory.is_dir()
            }
        )
        for name in names:
            weights, epochs = _find_checkpoint(output_root, name)
            report = _eval_checkpoint(
                weights,
                name,
                output_root,
                common["data"],
                common["device"],
                splits=splits,
                primary=expected_primary.get(name) or common.get("primary_head"),
            )
            _release()
            if not report:
                missing.append(name)
                continue

            records.append({
                "model": name,
                "family": family,
                "variant": name.replace("-seg", "").replace(family, "", 1).lstrip("-")
                           or None,
                "training": _training_record(weights),
                **{k: v for k, v in report.items() if k != "model"},
            })

            head = report.get("primary_head", "default")
            for split, scored in report["splits"].items():
                chosen = scored[head]
                row = {
                    "model": name,
                    "family": family,
                    "split": split,
                    "head": head,
                    "epochs": epochs,
                    "inference_ms": chosen["speed_ms"].get("inference"),
                    "postprocess_ms": chosen["speed_ms"].get("postprocess"),
                }
                for kind in ("box", "mask"):
                    for metric, value in chosen.get(kind, {}).items():
                        if metric != "per_class":
                            row[f"{kind}_{metric}"] = value
                    for class_name, values in chosen.get(kind, {}).get(
                        "per_class", {}
                    ).items():
                        for metric, value in values.items():
                            row[f"{class_name}_{kind}_{metric}"] = value
                rows.append(row)

    if not rows:
        raise SystemExit("no checkpoints evaluated")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    output = output_root / "eval_table.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info(f"[eval] {len(rows)} rows -> {output}")
    _write_json_report(output_root, config, records, missing)

    for split in splits:
        print(f"\n=== {split} (mask metrics) ===")
        print(
            f"{'model':<22}{'head':<10}{'mAP50':>9}"
            f"{'mAP50-95':>11}{'P':>8}{'R':>8}"
        )
        split_rows = sorted(
            (row for row in rows if row["split"] == split),
            key=lambda row: -row.get("mask_mAP50-95", 0.0),
        )
        for row in split_rows:
            print(
                f"{row['model']:<22}{row['head']:<10}"
                f"{row.get('mask_mAP50', 0.0):>9.4f}"
                f"{row.get('mask_mAP50-95', 0.0):>11.4f}"
                f"{row.get('mask_precision', 0.0):>8.3f}"
                f"{row.get('mask_recall', 0.0):>8.3f}"
            )
    if missing:
        print(f"\nno usable checkpoint for: {', '.join(missing)}")


def _take_config_path(argv: list[str]) -> str | None:
    """Pop --config PATH (or --config=PATH) out of argv and return the path.

    Sweeps are defined by their config file, so selecting one is how a stock
    baseline runs without editing the tuned recipe in place.
    """
    for index, argument in enumerate(argv):
        if argument == "--config":
            if index + 1 >= len(argv):
                raise SystemExit("--config needs a path")
            del argv[index]
            return argv.pop(index)
        if argument.startswith("--config="):
            return argv.pop(index).split("=", 1)[1]
    return None


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    config = load_config(_take_config_path(argv))
    families = config["families"]
    selected = [argument for argument in argv if not argument.startswith("-")] or list(
        families
    )
    unknown = [family for family in selected if family not in families]
    if unknown:
        raise SystemExit(
            f"unknown family/families {unknown}; train.yaml has {list(families)}"
        )

    if "--eval" in argv:
        eval_existing(config, selected)
        return

    common = config["common"]
    planned = [
        (family, variant, arm, overrides or {})
        for family in selected
        for variant in families[family]["variants"]
        for arm, overrides in (families[family].get("arms") or {"": {}}).items()
    ]
    shared = {key: value for key, value in common.items() if key not in _CONSUMED}
    LOGGER.info(
        f"=== {len(planned)} run(s) | project={common['project']} "
        f"| batch={common['batch']} ==="
    )
    LOGGER.info(f"shared: {shared}")
    for family, variant, arm, overrides in planned:
        suffix = f"-{arm}" if arm else ""
        LOGGER.info(f"{family}{variant}{suffix}: {overrides or '{}'}")

    failed = []
    for family, variant, arm, overrides in planned:
        stem = families[family]["stem"].format(v=variant)
        name = f"{stem}-{arm}" if arm else stem
        try:
            train(family, variant, config, arm, overrides)
        except Exception as error:
            LOGGER.error(f"{name} failed, continuing: {error}")
            failed.append(name)
        finally:
            _release()

    if failed:
        LOGGER.error(f"did not complete: {', '.join(failed)}")
    else:
        LOGGER.info(f"all {len(planned)} run(s) completed")


if __name__ == "__main__":
    main()
