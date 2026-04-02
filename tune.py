from datetime import datetime
from ultralytics import YOLO
from ultralytics.engine.tuner import Tuner
import os
os.system("clear")  # clear terminal


EPOCHS = 100
PATIENCE = 15
TRIALS = 300
CLOSE_MOSAIC = 10

SESSION_NAME = f"tune-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{TRIALS}"
PROJECT_NAME = f"run/polesight-tune-{TRIALS}"

model = YOLO("yolo26s-seg.pt")
tune_args = {
    **model.overrides,
    "mode": "train",
    "data": "data/data.yaml",
    "project": PROJECT_NAME,
    "name": SESSION_NAME,
    "epochs": EPOCHS,
    # "optimizer": "AdamW",
    "plots": True,
    "save": True,
    "val": True,
    "patience": PATIENCE,
    "close_mosaic": CLOSE_MOSAIC,
    "classes": [0, 1, 2, 4],  # skip power_pole (class index 3)
}

tuner = Tuner(args=tune_args, _callbacks=model.callbacks)
tuner(iterations=TRIALS, cleanup=False)
