#!/usr/bin/env python3
"""
Annotation Preview Script
Creates a 2x4 grid visualization of specified images with their bounding box annotations.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Configuration
DATA_DIR = "/home/ubuntu/thesis-Intersection/yolo/data"

# Class names and colors
CLASS_NAMES = ["roundabout", "intersection"]
CLASS_COLORS = ['red', 'blue']

def load_image_and_annotations(image_path):
    """Load an image and its corresponding annotations from the image path."""
    # Load image
    if not os.path.exists(image_path):
        return None, None
    
    image = Image.open(image_path)
    img_width, img_height = image.size
    
    # Determine label path by replacing images with labels and .jpg with .txt
    label_path = image_path.replace('/images/', '/labels/').replace('.jpg', '.txt')
    
    # Load annotations
    annotations = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line:
                    parts = line.split()
                    class_id = int(float(parts[0]))  # Handle both int and float formats
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convert YOLO format to absolute coordinates
                    x_center_abs = x_center * img_width
                    y_center_abs = y_center * img_height
                    width_abs = width * img_width
                    height_abs = height * img_height
                    
                    # Calculate top-left corner
                    x_min = x_center_abs - width_abs / 2
                    y_min = y_center_abs - height_abs / 2
                    
                    annotations.append({
                        'class_id': class_id,
                        'class_name': CLASS_NAMES[class_id],
                        'bbox': (x_min, y_min, width_abs, height_abs),
                        'color': CLASS_COLORS[class_id]
                    })
    
    return image, annotations

def plot_image_with_annotations(ax, image, annotations):
    """Plot an image with its bounding box annotations on the given axis."""
    ax.imshow(image)
    ax.axis('off')
    
    # Draw bounding boxes
    for ann in annotations:
        x_min, y_min, width, height = ann['bbox']
        
        # Create rectangle patch
        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=2, edgecolor=ann['color'], 
            facecolor='none', alpha=0.8
        )
        ax.add_patch(rect)
        
        # Add class ID label
        ax.text(
            x_min, y_min - 5, str(ann['class_id']),
            color=ann['color'], fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8)
        )

def create_annotation_grid(image_paths, num_rows=2, num_cols=4, save_path=None, jpg_quality=85):
    """Create a grid visualization with specified images."""
    # Use the provided image paths (up to 8 images)
    selected_images = image_paths[:num_rows * num_cols]
    dataset_types = []
    
    for image_path in selected_images:
        image_name = os.path.basename(image_path)
        if 'train' in image_path:
            dataset_types.append('train')
        elif 'val' in image_path:
            dataset_types.append('val')
        elif 'test' in image_path:
            dataset_types.append('test')
        else:
            dataset_types.append('train')  # default
    
    print(f"Selected {len(selected_images)} images for visualization")
    
    # Create the plot with proper aspect ratio for 960x640 images
    # Calculate figure size to maintain image proportions without stretching
    img_aspect = 960 / 640  # 1.5 (width/height)
    cell_width = 3  # inches per cell
    cell_height = cell_width / img_aspect  # 2 inches per cell
    
    fig_width = num_cols * cell_width  # 12 inches for 4 cols
    fig_height = num_rows * cell_height  # 4 inches for 2 rows
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
    
    # Handle case where we have fewer images than grid cells
    total_cells = num_rows * num_cols
    
    for idx in range(total_cells):
        row = idx // num_cols
        col = idx % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[col]
        
        if idx < len(selected_images):
            image_path = selected_images[idx]
            image, annotations = load_image_and_annotations(image_path)
            
            if image is not None:
                plot_image_with_annotations(ax, image, annotations)
            else:
                ax.text(0.5, 0.5, f'Error loading image', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
        else:
            # Empty cell
            ax.text(0.5, 0.5, 'No image available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=8)
            ax.axis('off')
    
    # Remove all padding and spacing between subplots - negative wspace for tighter columns
    plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0.08, 
                       wspace=-0.35, hspace=0.05)
    
    # Add legend back
    legend_elements = [patches.Patch(color=color, label=f"Class {idx}: {name}") 
                      for idx, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS))]
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(CLASS_NAMES), 
              bbox_to_anchor=(0.5, 0.01), fontsize=12)
    
    # Save as JPG with compression control
    if save_path:
        # Save as JPG directly with specified quality
        jpg_path = save_path.replace('.png', '.jpg')
        plt.savefig(jpg_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', pad_inches=0,
                   format='jpg', pil_kwargs={'quality': jpg_quality, 'optimize': True})
        print(f"Grid saved to: {jpg_path}")
        
        # Check file size
        file_size = os.path.getsize(jpg_path) / 1024  # KB
        print(f"File size: {file_size:.1f} KB (quality: {jpg_quality})")
        
        if file_size > 200:
            print(f"Warning: File size ({file_size:.1f} KB) is larger than 200 KB")
            print(f"Consider reducing jpg_quality parameter (current: {jpg_quality})")
        else:
            print(f"File size is within 200 KB limit")
    
    plt.close()  # Close the figure to free memory
    
    # Print statistics
    print(f"\nGrid Statistics:")
    train_count = sum(1 for dt in dataset_types if dt == 'train')
    val_count = sum(1 for dt in dataset_types if dt == 'val')
    test_count = sum(1 for dt in dataset_types if dt == 'test')
    print(f"- Train: {train_count} images")
    print(f"- Val: {val_count} images")
    print(f"- Test: {test_count} images")
    
    # Count annotations by class for each dataset
    dataset_annotations = {'train': {name: 0 for name in CLASS_NAMES},
                          'val': {name: 0 for name in CLASS_NAMES},
                          'test': {name: 0 for name in CLASS_NAMES}}
    
    for idx, image_path in enumerate(selected_images):
        dataset_type = dataset_types[idx]
        _, annotations = load_image_and_annotations(image_path)
        if annotations:
            for ann in annotations:
                dataset_annotations[dataset_type][ann['class_name']] += 1
    
    print(f"- Annotations by dataset and class:")
    for dataset_type in ['train', 'val', 'test']:
        print(f"  {dataset_type.upper()}:")
        for class_name, count in dataset_annotations[dataset_type].items():
            print(f"    * {class_name}: {count}")

def main():
    """Main function to create the annotation preview grid."""
    print("Creating 2x4 grid with specified images...")
    
    # List of 8 custom image paths (replace with your actual paths)
    image_paths = [
        # Add your 8 image paths here, e.g.:
        # "/home/ubuntu/thesis-Intersection/yolo/data/train/images/image1.jpg",
        # "/home/ubuntu/thesis-Intersection/yolo/data/val/images/image2.jpg",
        # etc.
        "/home/ubuntu/thesis-Intersection/yolo/data/train/images/0008.jpg",
        "/home/ubuntu/thesis-Intersection/yolo/data/train/images/0643.jpg",
        "/home/ubuntu/thesis-Intersection/yolo/data/train/images/0085.jpg",
        
        "/home/ubuntu/thesis-Intersection/yolo/data/train/images/0247.jpg",
        "/home/ubuntu/thesis-Intersection/yolo/data/train/images/0276.jpg",
        "/home/ubuntu/thesis-Intersection/yolo/data/train/images/0128.jpg",
        "/home/ubuntu/thesis-Intersection/yolo/data/train/images/0336.jpg",
        "/home/ubuntu/thesis-Intersection/yolo/data/train/images/0147.jpg",
            ]
    
    # Create the grid with adjustable JPG quality (lower = smaller file)
    # Try different quality levels to find one under 200KB
    save_path = os.path.join(DATA_DIR, "annotation_preview.png")  # Will be saved as .jpg
    
    # Try progressively lower quality until we get under 200KB
    create_annotation_grid(image_paths, num_rows=2, num_cols=4, save_path=save_path, jpg_quality=20)
    
    # Check if file size is acceptable
    jpg_path = save_path.replace('.png', '.jpg')
    file_size = os.path.getsize(jpg_path) / 1024
        


if __name__ == "__main__":
    main()
