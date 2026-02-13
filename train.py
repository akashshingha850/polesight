from ultralytics import YOLO
import os
import wandb
os.system('clear')

MODEL_NAME = "yolo26m"
EPOCHS = 100
BATCH_SIZE = 8
CLOSE_MOSAIC = 10
PATIENCE = 10


model = YOLO(f"{MODEL_NAME}-seg.pt")  # load a pretrained model (recommended for training)

wandb.init(project="polesight", name=f"{MODEL_NAME}_training")


results = model.train(data="data/data.yaml", epochs=EPOCHS, imgsz=640, batch=BATCH_SIZE, close_mosaic=CLOSE_MOSAIC, patience=PATIENCE, save=True)
