#!/usr/bin/env python3
"""
Comprehensive Dataset Analysis Script
Provides statistical analysis of the dataset including schema, instances, and bounding box analysis.
"""



import os
import numpy as np
from collections import defaultdict
import sys
from contextlib import redirect_stdout
import io

# clear the console (Linux/Unix)
os.system('clear')

# Configuration
DATA_DIR = "/home/ubuntu/thesis-Intersection/yolo/data"
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, "train/images")
TRAIN_LABELS_DIR = os.path.join(DATA_DIR, "train/labels")
VAL_IMAGES_DIR = os.path.join(DATA_DIR, "val/images")
VAL_LABELS_DIR = os.path.join(DATA_DIR, "val/labels")
TEST_IMAGES_DIR = os.path.join(DATA_DIR, "test/images")
TEST_LABELS_DIR = os.path.join(DATA_DIR, "test/labels")

# Class names and colors
CLASS_NAMES = ["roundabout", "intersection"]
CLASS_COLORS = ['red', 'blue']

# ===== DATA SCHEMA ANALYSIS =====

def count_files(directory, ext=None):
    """Counts files with a given extension in a directory and its subdirectories."""
    count = 0
    for root, _, files in os.walk(directory):
        if ext:
            count += len([f for f in files if f.endswith(ext)])
        else:
            count += len(files)
    return count

def analyze_image_file_sizes():
    """Analyze file sizes of all images in the dataset."""
    all_sizes = []
    
    # Collect file sizes from all splits
    for split in ['train', 'val', 'test']:
        images_dir = os.path.join(DATA_DIR, split, 'images')
        if os.path.exists(images_dir):
            for filename in os.listdir(images_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    filepath = os.path.join(images_dir, filename)
                    size_kb = os.path.getsize(filepath) / 1024  # Convert to KB
                    all_sizes.append(size_kb)
    
    if all_sizes:
        return {
            'min': min(all_sizes),
            'max': max(all_sizes),
            'avg': np.mean(all_sizes),
            'total_files': len(all_sizes)
        }
    return {'min': 0, 'max': 0, 'avg': 0, 'total_files': 0}

def print_tree(directory, indent=""):
    """Recursively prints the directory structure with file counts."""
    def _count_top_level(d):
        try:
            return count_files(d)
        except Exception:
            return 0

    print(f"Schema for: {directory}")
    try:
        items = sorted(os.listdir(directory))
    except FileNotFoundError:
        print(f"  (directory not found: {directory})")
        return

    for item in items:
        path = os.path.join(directory, item)
        if os.path.isdir(path):
            total = _count_top_level(path)
            print(f"- {item}/ ({total} files)")
            # show immediate children counts (useful for train/val/test and images/labels)
            try:
                subs = sorted(os.listdir(path))
            except Exception:
                subs = []
            for sub in subs:
                subpath = os.path.join(path, sub)
                if os.path.isdir(subpath):
                    subcount = _count_top_level(subpath)
                    print(f"    └─ {sub}/ ({subcount} files)")
        else:
            print(f"- {item}")

def count_annotated_images(images_dir, labels_dir=None):
    """Count images that have corresponding annotation files."""
    annotated_count = 0
    for root, _, files in os.walk(images_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".png")):
                # Build the expected label filename
                label_file = os.path.splitext(file)[0] + ".txt"

                if labels_dir:
                    # Compute relative directory from images_dir to preserve subdirectory structure
                    rel_dir = os.path.relpath(root, images_dir)
                    if rel_dir == ".":
                        rel_dir = ""
                    label_path = os.path.join(labels_dir, rel_dir, label_file) if rel_dir else os.path.join(labels_dir, label_file)
                else:
                    # Labels stored next to images
                    label_path = os.path.join(root, label_file)

                if os.path.exists(label_path):
                    annotated_count += 1
    return annotated_count

def analyze_set(set_name, images_dir, labels_dir):
    """Analyze a specific dataset set (train/val/test)"""
    if os.path.exists(images_dir) and os.path.exists(labels_dir):
        image_count = count_files(images_dir, ".jpg") + count_files(images_dir, ".png")
        label_count = count_files(labels_dir, ".txt")
        
        # Count annotated images by checking for corresponding .txt files in labels_dir
        annotated_count = count_annotated_images(images_dir, labels_dir)
        
        print(f"\n{set_name} set analysis:")
        print(f"  Images: {image_count}")
        print(f"  Labels: {label_count}")
        print(f"  Annotated images: {annotated_count}")
        print(f"  Unannotated images: {image_count - annotated_count}")
        
        return {
            'images': image_count,
            'labels': label_count,
            'annotated': annotated_count,
            'unannotated': image_count - annotated_count
        }
    return None

# ===== DATA INSTANCES ANALYSIS =====

def analyze_data_instances():
    """Analyze instances per class for each data split."""
    data_splits = ['train', 'val', 'test']
    
    # Dictionary to hold counts: subfolder -> class_id -> count
    stats = defaultdict(lambda: defaultdict(int))

    for split in data_splits:
        labels_path = os.path.join(DATA_DIR, split, 'labels')
        if not os.path.exists(labels_path):
            print(f"Labels path {labels_path} does not exist.")
            continue
        
        for filename in os.listdir(labels_path):
            if filename.endswith('.txt'):
                filepath = os.path.join(labels_path, filename)
                with open(filepath, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(float(parts[0]))
                            stats[split][class_id] += 1

    # Print the stats
    print("\nStatistics of instances per class for each subfolder:")
    print("-" * 50)
    for subfolder in data_splits:
        print(f"\n{subfolder.upper()}:")
        total_instances = 0
        for class_id in range(len(CLASS_NAMES)):
            count = stats[subfolder][class_id]
            print(f"  {CLASS_NAMES[class_id]}: {count}")
            total_instances += count
        print(f"  Total: {total_instances}")

    # Overall stats
    print("\n" + "=" * 50)
    print("OVERALL:")
    overall = defaultdict(int)
    for subfolder in data_splits:
        for class_id in range(len(CLASS_NAMES)):
            overall[class_id] += stats[subfolder][class_id]

    for class_id in range(len(CLASS_NAMES)):
        print(f"  {CLASS_NAMES[class_id]}: {overall[class_id]}")
    print(f"  Total: {sum(overall.values())}")

    return stats, overall


# ===== BOUNDING BOX ANALYSIS =====

def analyze_bounding_boxes(labels_dir):
    """Analyze bounding box statistics from YOLO format labels."""
    all_widths = []
    all_heights = []
    all_areas = []
    all_aspect_ratios = []
    class_counts = {0: 0, 1: 0}  # roundabout: 0, intersection: 1
    
    total_boxes = 0
    
    for filename in os.listdir(labels_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(labels_dir, filename)
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(float(parts[0]))
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        all_widths.append(width)
                        all_heights.append(height)
                        all_areas.append(width * height)
                        all_aspect_ratios.append(width / height if height > 0 else 1.0)
                        
                        class_counts[class_id] += 1
                        total_boxes += 1
    
    return {
        'widths': all_widths,
        'heights': all_heights,
        'areas': all_areas,
        'aspect_ratios': all_aspect_ratios,
        'class_counts': class_counts,
        'total_boxes': total_boxes
    }

def calc_stats(data):
    """Calculate basic statistics for a list of data."""
    if not data:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0}
    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'median': np.median(data)
    }

def comprehensive_bbox_analysis():
    """Perform comprehensive bounding box analysis across all splits."""
    splits = ['train', 'val', 'test']
    all_stats = {}
    
    for split in splits:
        labels_path = os.path.join(DATA_DIR, split, 'labels')
        if os.path.exists(labels_path):
            stats = analyze_bounding_boxes(labels_path)
            all_stats[split] = stats

    # Combine all splits for overall statistics
    combined_widths = []
    combined_heights = []
    combined_areas = []
    combined_aspect_ratios = []
    total_class_counts = {0: 0, 1: 0}

    for split_stats in all_stats.values():
        combined_widths.extend(split_stats['widths'])
        combined_heights.extend(split_stats['heights'])
        combined_areas.extend(split_stats['areas'])
        combined_aspect_ratios.extend(split_stats['aspect_ratios'])
        for class_id in total_class_counts:
            total_class_counts[class_id] += split_stats['class_counts'][class_id]

    # Calculate statistics
    width_stats = calc_stats(combined_widths)
    height_stats = calc_stats(combined_heights)
    area_stats = calc_stats(combined_areas)
    aspect_ratio_stats = calc_stats(combined_aspect_ratios)

    print("\n=== BOUNDING BOX ANALYSIS ===")
    print(f"Total bounding boxes: {len(combined_widths)}")
    print(f"Class distribution:")
    print(f"  Roundabout (0): {total_class_counts[0]} ({total_class_counts[0]/len(combined_widths)*100:.1f}%)" if combined_widths else "  No data")
    print(f"  Intersection (1): {total_class_counts[1]} ({total_class_counts[1]/len(combined_widths)*100:.1f}%)" if combined_widths else "  No data")

    print(f"\nBounding Box Width (normalized):")
    print(f"  Mean: {width_stats['mean']:.3f} ± {width_stats['std']:.3f}")
    print(f"  Range: {width_stats['min']:.3f} - {width_stats['max']:.3f}")
    print(f"  Median: {width_stats['median']:.3f}")

    print(f"\nBounding Box Height (normalized):")
    print(f"  Mean: {height_stats['mean']:.3f} ± {height_stats['std']:.3f}")
    print(f"  Range: {height_stats['min']:.3f} - {height_stats['max']:.3f}")
    print(f"  Median: {height_stats['median']:.3f}")

    print(f"\nBounding Box Area (normalized):")
    print(f"  Mean: {area_stats['mean']:.3f} ± {area_stats['std']:.3f}")
    print(f"  Range: {area_stats['min']:.3f} - {area_stats['max']:.3f}")
    print(f"  Median: {area_stats['median']:.3f}")

    print(f"\nAspect Ratio (width/height):")
    print(f"  Mean: {aspect_ratio_stats['mean']:.3f} ± {aspect_ratio_stats['std']:.3f}")
    print(f"  Range: {aspect_ratio_stats['min']:.3f} - {aspect_ratio_stats['max']:.3f}")
    print(f"  Median: {aspect_ratio_stats['median']:.3f}")

    # Print per-split statistics
    print(f"\n=== PER-SPLIT BBOX STATISTICS ===")
    for split in splits:
        if split in all_stats:
            stats = all_stats[split]
            print(f"\n{split.upper()} set:")
            print(f"  Total boxes: {stats['total_boxes']}")
            print(f"  Roundabout: {stats['class_counts'][0]}")
            print(f"  Intersection: {stats['class_counts'][1]}")
            if stats['aspect_ratios']:
                split_ar_stats = calc_stats(stats['aspect_ratios'])
                print(f"  Avg aspect ratio: {split_ar_stats['mean']:.3f}")
    
    # Return structured stats for LaTeX generation
    return {
        'width': width_stats,
        'height': height_stats,
        'area': area_stats,
        'aspect_ratio': aspect_ratio_stats
    }

def main():
    """Main function to run comprehensive dataset analysis and save output to text file."""
    
    # Capture all output to a string buffer
    output_buffer = io.StringIO()
    
    with redirect_stdout(output_buffer):
        print("=" * 60)
        print("COMPREHENSIVE DATASET ANALYSIS")
        print("=" * 60)
        
        # 1. Data Schema Analysis
        print("\n1. DATA SCHEMA ANALYSIS")
        print("-" * 30)
        base_file_count = count_files(DATA_DIR)
        print(f"Data ({base_file_count} files):")
        print_tree(DATA_DIR)

        # Analyze each dataset set
        train_stats = analyze_set("Train", TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR)
        val_stats = analyze_set("Validation", VAL_IMAGES_DIR, VAL_LABELS_DIR)
        test_stats = analyze_set("Test", TEST_IMAGES_DIR, TEST_LABELS_DIR)
        
        split_stats = {
            'train': train_stats,
            'val': val_stats,
            'test': test_stats
        }

        # Overall statistics
        print("\nOverall statistics:")
        total_images = 0
        total_annotated = 0
        
        if train_stats:
            total_images += train_stats['images']
            total_annotated += train_stats['annotated']
        if val_stats:
            total_images += val_stats['images']
            total_annotated += val_stats['annotated']
        if test_stats:
            total_images += test_stats['images']
            total_annotated += test_stats['annotated']
        
        print(f"Images with annotations: {total_annotated}")
        print(f"Background without annotations: {total_images - total_annotated}")

        # 2. File Size Analysis
        print("\n2. IMAGE FILE SIZE ANALYSIS")
        print("-" * 30)
        file_size_stats = analyze_image_file_sizes()
        print(f"File size statistics (KB):")
        print(f"  Min: {file_size_stats['min']:.0f}")
        print(f"  Max: {file_size_stats['max']:.0f}")
        print(f"  Average: {file_size_stats['avg']:.0f}")
        print(f"  Total files analyzed: {file_size_stats['total_files']}")

        # 3. Data Instances Analysis
        print("\n3. DATA INSTANCES ANALYSIS")
        print("-" * 30)
        stats, overall = analyze_data_instances()

        # 4. Bounding Box Analysis
        print("\n4. BOUNDING BOX ANALYSIS")
        print("-" * 30)
        bbox_stats = comprehensive_bbox_analysis()

        # 5. Summary Statistics
        print("\n5. DATASET SUMMARY FOR LATEX TABLES")
        print("-" * 40)
        
        # Create instance stats dictionary from current analysis
        instance_stats = {
            "dataset_split_instances": {sub: dict(stats[sub]) for sub in ['train', 'val', 'test']},
            "overall_dataset_instances": dict(overall),
            "class_names": CLASS_NAMES
        }
        
        # Print comprehensive summary for manual LaTeX table creation
        print_dataset_summary(file_size_stats, bbox_stats, instance_stats, split_stats)
        
        print("\n" + "=" * 60)
        print("COMPREHENSIVE ANALYSIS COMPLETE")
        print("=" * 60)
    
    # Get the captured output
    analysis_output = output_buffer.getvalue()
    
    # Save to text file
    output_file = os.path.join(DATA_DIR, 'dataset_analysis_report.txt')
    with open(output_file, 'w') as f:
        f.write(analysis_output)
    
    # Also print to console
    print(analysis_output)
    print(f"Analysis report saved to: {output_file}")

def print_dataset_summary(file_size_stats, bbox_stats, instance_stats, split_stats):
    """Print comprehensive dataset summary for LaTeX table creation."""
    
    # Calculate totals - use string keys to match the data structure
    total_images = sum(stats['images'] for stats in split_stats.values() if stats)
    roundabout_count = instance_stats['overall_dataset_instances'].get(0, 0)
    intersection_count = instance_stats['overall_dataset_instances'].get(1, 0)
    total_annotations = roundabout_count + intersection_count
    
    print("\n=== DATASET OVERVIEW TABLE DATA ===")
    print(f"Total images: {total_images}")
    print(f"Total annotations: {total_annotations:,}")
    print(f"Classes: 2 (Roundabout, Intersection)")
    print(f"Image format: JPEG")
    print(f"Annotation format: YOLO (normalized coordinates)")
    print(f"Original resolution: 960 × 640 pixels")
    print(f"Training resolution: 640 × 640 pixels (resized)")
    print(f"Aspect ratio: 3:2 (1.5:1)")
    print(f"Color space: RGB")
    print(f"Compression: Variable quality")
    print(f"File size (KB): Min: {file_size_stats['min']:.0f}, Max: {file_size_stats['max']:.0f}, Avg: {file_size_stats['avg']:.0f}")
    print(f"Dataset type: Object Detection")
    
    print("\n=== BOUNDING BOX STATISTICS TABLE DATA ===")
    print(f"Width (normalized):")
    print(f"  Mean ± Std: {bbox_stats['width']['mean']:.3f} ± {bbox_stats['width']['std']:.3f}")
    print(f"  Median: {bbox_stats['width']['median']:.3f}")
    print(f"  Range: {bbox_stats['width']['min']:.3f} - {bbox_stats['width']['max']:.3f}")
    
    print(f"Height (normalized):")
    print(f"  Mean ± Std: {bbox_stats['height']['mean']:.3f} ± {bbox_stats['height']['std']:.3f}")
    print(f"  Median: {bbox_stats['height']['median']:.3f}")
    print(f"  Range: {bbox_stats['height']['min']:.3f} - {bbox_stats['height']['max']:.3f}")
    
    print(f"Area (normalized):")
    print(f"  Mean ± Std: {bbox_stats['area']['mean']:.3f} ± {bbox_stats['area']['std']:.3f}")
    print(f"  Median: {bbox_stats['area']['median']:.3f}")
    print(f"  Range: {bbox_stats['area']['min']:.3f} - {bbox_stats['area']['max']:.3f}")
    
    print(f"Aspect Ratio (width/height):")
    print(f"  Mean ± Std: {bbox_stats['aspect_ratio']['mean']:.3f} ± {bbox_stats['aspect_ratio']['std']:.3f}")
    print(f"  Median: {bbox_stats['aspect_ratio']['median']:.3f}")
    print(f"  Range: {bbox_stats['aspect_ratio']['min']:.3f} - {bbox_stats['aspect_ratio']['max']:.3f}")
    
    print("\n=== CLASS DISTRIBUTION TABLE DATA ===")
    roundabout_pct = (roundabout_count / total_annotations * 100) if total_annotations > 0 else 0
    intersection_pct = (intersection_count / total_annotations * 100) if total_annotations > 0 else 0
    print(f"Intersection: {intersection_count} instances ({intersection_pct:.1f}%)")
    print(f"Roundabout: {roundabout_count} instances ({roundabout_pct:.1f}%)")
    print(f"Total: {total_annotations:,} instances (100.0%)")
    
    print("\n=== DATA SPLIT DISTRIBUTION TABLE DATA ===")
    
    # Calculate split statistics
    train_images = split_stats['train']['images'] if split_stats['train'] else 0
    val_images = split_stats['val']['images'] if split_stats['val'] else 0
    test_images = split_stats['test']['images'] if split_stats['test'] else 0
    
    train_intersections = instance_stats['dataset_split_instances']['train'].get(1, 0)
    train_roundabouts = instance_stats['dataset_split_instances']['train'].get(0, 0)
    val_intersections = instance_stats['dataset_split_instances']['val'].get(1, 0)
    val_roundabouts = instance_stats['dataset_split_instances']['val'].get(0, 0)
    test_intersections = instance_stats['dataset_split_instances']['test'].get(1, 0)
    test_roundabouts = instance_stats['dataset_split_instances']['test'].get(0, 0)
    
    train_total = train_intersections + train_roundabouts
    val_total = val_intersections + val_roundabouts
    test_total = test_intersections + test_roundabouts
    
    # Calculate percentages
    train_img_pct = (train_images / total_images * 100) if total_images > 0 else 0
    val_img_pct = (val_images / total_images * 100) if total_images > 0 else 0
    test_img_pct = (test_images / total_images * 100) if total_images > 0 else 0
    
    train_int_pct = (train_intersections / intersection_count * 100) if intersection_count > 0 else 0
    val_int_pct = (val_intersections / intersection_count * 100) if intersection_count > 0 else 0
    test_int_pct = (test_intersections / intersection_count * 100) if intersection_count > 0 else 0
    
    train_round_pct = (train_roundabouts / roundabout_count * 100) if roundabout_count > 0 else 0
    val_round_pct = (val_roundabouts / roundabout_count * 100) if roundabout_count > 0 else 0
    test_round_pct = (test_roundabouts / roundabout_count * 100) if roundabout_count > 0 else 0
    
    train_total_pct = (train_total / total_annotations * 100) if total_annotations > 0 else 0
    val_total_pct = (val_total / total_annotations * 100) if total_annotations > 0 else 0
    test_total_pct = (test_total / total_annotations * 100) if total_annotations > 0 else 0
    
    print(f"Training split:")
    print(f"  Images: {train_images} ({train_img_pct:.1f}%)")
    print(f"  Intersections: {train_intersections} ({train_int_pct:.1f}%)")
    print(f"  Roundabouts: {train_roundabouts} ({train_round_pct:.1f}%)")
    print(f"  Total instances: {train_total} ({train_total_pct:.1f}%)")
    
    print(f"Validation split:")
    print(f"  Images: {val_images} ({val_img_pct:.1f}%)")
    print(f"  Intersections: {val_intersections} ({val_int_pct:.1f}%)")
    print(f"  Roundabouts: {val_roundabouts} ({val_round_pct:.1f}%)")
    print(f"  Total instances: {val_total} ({val_total_pct:.1f}%)")
    
    print(f"Testing split:")
    print(f"  Images: {test_images} ({test_img_pct:.1f}%)")
    print(f"  Intersections: {test_intersections} ({test_int_pct:.1f}%)")
    print(f"  Roundabouts: {test_roundabouts} ({test_round_pct:.1f}%)")
    print(f"  Total instances: {test_total} ({test_total_pct:.1f}%)")
    
    print(f"Total:")
    print(f"  Images: {total_images} (100%)")
    print(f"  Intersections: {intersection_count} (100%)")
    print(f"  Roundabouts: {roundabout_count} (100%)")
    print(f"  Total instances: {total_annotations:,} (100%)")

if __name__ == "__main__":
    main()
