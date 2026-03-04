from datetime import datetime
from ultralytics import YOLO
from ultralytics.engine.tuner import Tuner
import os
os.system("clear")  # clear terminal


EPOCHS = 100
PATIENCE = 15
TRIALS = 100
CLOSE_MOSAIC = 10

SESSION_NAME = f"tune-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

model = YOLO("yolo26s-seg.pt")
tune_args = {
    **model.overrides,
    "mode": "train",
    "data": "data/data.yaml",
    "project": "polesight-tune",
    "name": SESSION_NAME,
    "epochs": EPOCHS,
    "optimizer": "AdamW",
    "plots": True,
    "save": True,
    "val": True,
    "patience": PATIENCE,
    "close_mosaic": CLOSE_MOSAIC,
}

tuner = Tuner(args=tune_args, _callbacks=model.callbacks)
tuner(iterations=TRIALS, cleanup=False)
