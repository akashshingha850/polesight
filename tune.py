from datetime import datetime
from ultralytics import YOLO
from ultralytics.engine.tuner import Tuner
import os
import yaml
import pandas as pd
import wandb

os.system("clear")  # clear terminal


EPOCHS = 100
PATIENCE = 15
TRIALS = 300
CLOSE_MOSAIC = 10
BATCH_SIZE = 8

SESSION_NAME = f"fine-tune-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{TRIALS}"
PROJECT_NAME = f"polesight-tune-rdetr-{TRIALS}"
PROJECT_DIR = f"tune/{PROJECT_NAME}"

model = YOLO("rtdetr-l.pt")
tune_args = {
    **model.overrides,
    "mode": "train",
    "data": "data/data.yaml",
    "project": PROJECT_DIR,
    "name": SESSION_NAME,
    "epochs": EPOCHS,
    # "optimizer": "AdamW",
    "plots": True,
    "save": True,
    "val": True,
    "patience": PATIENCE,
    "close_mosaic": CLOSE_MOSAIC,
    "classes": [0, 1, 2, 4],  # skip power_pole (class index 3)
    "batch": BATCH_SIZE
}

wandb.init(project=PROJECT_NAME, name=SESSION_NAME, config=tune_args)

trial_counter = {"n": 0}

def log_trial(trainer):
    trial_counter["n"] += 1
    log_data = {"trial": trial_counter["n"], "fitness": trainer.fitness}
    log_data.update({f"metric/{k}": v for k, v in trainer.metrics.items()})
    log_data.update({f"hp/{k}": v for k, v in vars(trainer.args).items()
                     if k not in ("data", "project", "name", "mode")})
    wandb.log(log_data)

model.add_callback("on_train_end", log_trial)

tuner = Tuner(args=tune_args, _callbacks=model.callbacks)
tuner(iterations=TRIALS, cleanup=False)

# log final artifacts
tune_dir = tuner.tune_dir
results_csv = os.path.join(tune_dir, "tune_results.csv")
best_yaml = os.path.join(tune_dir, "best_hyperparameters.yaml")

if os.path.exists(results_csv):
    artifact = wandb.Artifact("tune_results", type="dataset")
    artifact.add_file(results_csv)
    if os.path.exists(best_yaml):
        artifact.add_file(best_yaml)
    wandb.log_artifact(artifact)

# log final summary
if os.path.exists(results_csv):
    df = pd.read_csv(results_csv)
    best_row = df.loc[df["fitness"].idxmax()]
    wandb.summary["best_fitness"] = best_row["fitness"]
    wandb.summary["best_trial"] = int(best_row.name) + 1

if os.path.exists(best_yaml):
    with open(best_yaml) as f:
        raw = "\n".join(l for l in f if not l.startswith("#"))
    for k, v in yaml.safe_load(raw).items():
        wandb.summary[f"best_hp/{k}"] = v

wandb.finish()

