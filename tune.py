from datetime import datetime
from ultralytics import YOLO
from ultralytics.engine.tuner import Tuner
import os
import yaml
import pandas as pd
import wandb

os.system("clear")  # clear terminal

CONFIG = "tune.yaml"


# The search algorithm this subclasses (BLX-alpha crossover + log-normal
# mutation with a decaying sigma) has no paper of its own; see the ACADEMIC
# REFERENCES block in tune.yaml for the per-mechanism citations and DOIs.
class ConfigTuner(Tuner):
    """Tuner that snaps declared hyperparameters to integers.

    Ultralytics' Tuner int-rounds only close_mosaic and epochs, so any other
    integer-valued search key (batch, nbs, mask_ratio, ...) is emitted as a
    float. A float batch reaches torch's BatchSampler and raises ValueError,
    which Tuner catches and records as fitness=0.0 — every trial fails silently.
    """

    def __init__(self, args, int_params=None, _callbacks=None):
        self.int_params = int_params or {}
        super().__init__(args=args, _callbacks=_callbacks)

    def _mutate(self, *args, **kwargs):
        hyp = super()._mutate(*args, **kwargs)
        for k, step in self.int_params.items():
            if k not in hyp:
                continue
            step = max(int(step or 1), 1)
            lo, hi = self.space[k][0], self.space[k][1]
            v = max(int(round(hyp[k] / step)) * step, step)
            hyp[k] = int(min(max(v, lo), hi))
        return hyp

with open(CONFIG) as f:
    cfg = yaml.safe_load(f)

tuner_cfg = cfg.get("tuner", {})
train_cfg = cfg.get("train", {})
space = cfg[tuner_cfg.get("use_space", "space")]

TRIALS = tuner_cfg.get("iterations", 300)
MODEL = train_cfg.get("model", "yolo26m-seg.pt")

SESSION_NAME = f"fine-tune-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{TRIALS}"
PROJECT_NAME = f"polesight-tune-{os.path.splitext(os.path.basename(MODEL))[0]}-{TRIALS}"
PROJECT_DIR = f"tune/{PROJECT_NAME}"

model = YOLO(MODEL)
tune_args = {
    **model.overrides,
    **train_cfg,
    "space": space,
    "project": PROJECT_DIR,
    "name": SESSION_NAME,
}
# strip null-valued keys so Ultralytics' own defaults apply
tune_args = {k: v for k, v in tune_args.items() if v is not None}

wandb.init(project=PROJECT_NAME, name=SESSION_NAME, config=tune_args)

# Trials train in a subprocess, so the parent's trainer callbacks never fire and
# Ultralytics' built-in W&B integration would open one run per trial. Disabling
# W&B for children (after the parent run exists) keeps a single tuning run.
if not tuner_cfg.get("per_trial_wandb", False):
    os.environ["WANDB_MODE"] = "disabled"

tuner = ConfigTuner(args=tune_args, int_params=tuner_cfg.get("int_params"))
tuner(iterations=TRIALS, cleanup=tuner_cfg.get("cleanup", False))

# log final artifacts
tune_dir = tuner.tune_dir
results_csv = os.path.join(tune_dir, "tune_results.csv")
best_yaml = os.path.join(tune_dir, "best_hyperparameters.yaml")

if os.path.exists(results_csv):
    df = pd.read_csv(results_csv)

    # replay the per-trial history into wandb
    best_so_far = float("-inf")
    for i, row in df.iterrows():
        best_so_far = max(best_so_far, row["fitness"])
        wandb.log({"trial": i + 1, "fitness": row["fitness"], "best_fitness": best_so_far,
                   **{f"hp/{k}": row[k] for k in space if k in df.columns}})

    artifact = wandb.Artifact("tune_results", type="dataset")
    artifact.add_file(results_csv)
    if os.path.exists(best_yaml):
        artifact.add_file(best_yaml)
    wandb.log_artifact(artifact)

    best_row = df.loc[df["fitness"].idxmax()]
    wandb.summary["best_fitness"] = best_row["fitness"]
    wandb.summary["best_trial"] = int(best_row.name) + 1

if os.path.exists(best_yaml):
    with open(best_yaml) as f:
        raw = "\n".join(l for l in f if not l.startswith("#"))
    for k, v in yaml.safe_load(raw).items():
        wandb.summary[f"best_hp/{k}"] = v

wandb.finish()
