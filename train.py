from ultralytics import YOLO
import os
import wandb

EPOCHS = 100
IMGSZ = 640
BATCH_SIZE = 8
PATIENCE = 20

YOLO_MODEL = "yolo26"  # Pretrained model to start from
VARIENT = ['n', 's', 'm', 'l', 'x']

#clear terminal
os.system('clear')

# wandb.init(project="polesight")

# model = YOLO(f"{YOLO_MODEL}-seg.pt")  # load a pretrained model (recommended for training)

for var in VARIENT:
    print(f"Training {YOLO_MODEL}{var}-seg.pt")
    model = YOLO(f"{YOLO_MODEL}{var}-seg.pt")  # load a pretrained model (recommended for training)
    # Let YOLO handle wandb initialization internally
    results = model.train(data="data/data.yaml", 
                          project="polesight",
                          name=f"{YOLO_MODEL}{var}-seg", 
                          epochs=EPOCHS, imgsz=IMGSZ, 
                          batch=BATCH_SIZE, patience=PATIENCE,
                          save = True,  # Save the best model
                          )
