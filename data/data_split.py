import os
import shutil
from sklearn.model_selection import train_test_split

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Define dataset directory
data_dir = "/home/ubuntu/Documents/github/polesight/data"

# Assume images are in the images folder
all_images_dir = "/home/ubuntu/Documents/github/polesight/data/images"

# Get all image files (supporting .jpg and .png)
image_files = [f for f in os.listdir(all_images_dir) if f.endswith((".jpg", ".png"))]

# Shuffle and split using sklearn for precise ratios (80/10/10) with fixed seed
train_files, temp = train_test_split(image_files, test_size=0.2, random_state=42) # 80% train 
val_files, test_files = train_test_split(temp, test_size=0.5, random_state=42)  # 10% val, 10% test

# Create train, val, and test directories if they don't exist
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(data_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, split, "labels"), exist_ok=True)

# Function to move files
def move_files(file_list, split):
    for file in file_list:
        base_name = os.path.splitext(file)[0]
        img_src = os.path.join(all_images_dir, file)
        lbl_src = os.path.join(data_dir, "labels", base_name + ".txt")

        img_dst = os.path.join(data_dir, split, "images", file)
        lbl_dst = os.path.join(data_dir, split, "labels", base_name + ".txt")

        # Move image
        shutil.move(img_src, img_dst)

        # Move label if exists
        if os.path.exists(lbl_src):
            shutil.move(lbl_src, lbl_dst)

# Move files to respective directories
move_files(train_files, "train")
move_files(val_files, "val")
move_files(test_files, "test")

print("Dataset split and reorganization complete!")
