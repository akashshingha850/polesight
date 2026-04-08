import time
from ultralytics import YOLO
import os
import wandb

EPOCHS = 10
IMGSZ = 640
BATCH_SIZE = 16
PATIENCE = 20

YOLO_MODEL = "yolo26"  # Pretrained model to start from
# VARIENT = ['n', 's', 'm', 'l', 'x']
VARIENT = ['s']

#clear terminal
os.system('clear')

# wandb.init(project="polesight")

# model = YOLO(f"{YOLO_MODEL}-seg.pt")  # load a pretrained model (recommended for training)

for var in VARIENT:
    BATCH = BATCH_SIZE
 
    while True:
        try:
            print(f"Training {YOLO_MODEL}{var}-seg.pt")
            model = YOLO(f"{YOLO_MODEL}{var}-seg.pt")  # load a pretrained model (recommended for training)
   
            # Let YOLO handle wandb initialization internally
            results = model.train(data="data/data.yaml", 
                                  project="polesight",
                                  name=f"{YOLO_MODEL}{var}-{BATCH}_light_pole",  # Unique name for this run
                                  epochs=EPOCHS, imgsz=IMGSZ, 
                                  batch=BATCH, patience=PATIENCE,
                                  save = True,  # Save the best model
                                  )
            break  # Training successful
        except RuntimeError as e:
            wandb.finish(exit_code=1)  # Finish the current run with error status
            time.sleep(5)  # Wait a bit before retrying
            if "out of memory" in str(e).lower():
                BATCH = BATCH // 2
                print(f"CUDA OOM encountered, reducing batch size to {BATCH}")
            else:
                raise
