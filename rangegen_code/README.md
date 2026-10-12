# 📦 RangeImageGenerator — README

A modular, extensible Python toolkit for generating **LiDAR range images**, **RGB projections**, **intensity maps**, **classification maps**, and **pose‑aware point cloud outputs** using **LAS/LAZ** point clouds and (optionally) **trajectory files**.

This toolkit supports:

- Sensor‑frame range image generation
- Randomized pose augmentation
- Frame generation along a trajectory
- Pole‑based sampling and selection
- Object detection‑driven frame generation (COCO workflow)
- Direct sampling‑based frame generation (no trajectory required)
- Saving EXIF metadata (GPS + timestamp)
- Full spherical projection and filtering utilities
- Reprojection testing and validation
- COCO annotation generation and label expansion

The package is structured for research, dataset creation, and ML pipeline input generation.

---

## 📁 Project Structure

```
src/rangeimagegenerator/
│
├── __init__.py                    # Package initialization, exports
├── main.py                        # Main entry point for CLI pipeline
│
├── range_image_generator/
│   ├── __init__.py               # API exports, types
│   ├── io_utils.py               # LAS/LAZ reading, PCD writing, image saving with EXIF
│   ├── projection.py             # Spherical range projection + hole-filling
│   ├── transforms.py             # Coordinate transforms, pose perturbations
│   ├── frame_generator.py        # High-level transform + project pipeline
│   ├── utils.py                  # FrameIntrinsics, directory helpers
│   ├── las_helper.py             # LAS→memmap, recarray utilities
│   ├── las_helper.py             # LAS→recarray utilities
│   ├── read_exif.py              # EXIF reading utilities
│   ├── pyprojCoordinateTranformer.py  # ETRS-TM35FIN → WGS84 coordinate transforms
│   └── x200go_nmea_parser.py     # NMEA trajectory parsing
│
├── generate_from_trajectory.py      # Trajectory-based frame generation
├── generate_from_random_points.py   # Random sampling-based frame generation
├── generate_from_trajectory_with_pole_detection.py  # Pole-based generation
├── generate_around_objects.py       # Object detection-driven generation (COCO)
└── reprojection/                    # Reprojection testing module
    ├── deprojection.py
    ├── coco_parser.py
    ├── filenamematcher.py
    ├── label_expansion.py
    └── multiprocessing_workers.py
```

Configuration examples are available in `confs/` directory.

---

## 🚀 Quick Start

### **1. Prerequisites**

- Python 3.12+
- Dependencies: `numpy`, `scipy`, `pillow`, `opencv-python`, `laspy`, `toml`, `pypcd4`, `pycocotools`, `pyproj`, `scikit-learn`, `tqdm`, `matplotlib`, `open3d`

### **2. Installation**

```bash
pip install numpy scipy pillow opencv-python laspy toml pypcd4 pycocotools pyproj scikit-learn tqdm matplotlib open3d
```

To install as editable package:

```bash
pip install -e .
```

### **3. Run Example**

```bash
python -m rangeimagegenerator reprojection_testing/config.toml
```

See [`confs/`](confs/) for more configuration examples.

---

## 🛠 Installation & Setup

```bash
# Install dependencies
pip install numpy scipy pillow opencv-python laspy toml pypcd4 pycocotools pyproj scikit-learn tqdm matplotlib open3d

# Optional: install as editable
pip install -e .

# Verify installation
python -c "from rangeimagegenerator import read_laz_file; print('OK')"
```

---

## 🚀 Features

### **✓ LAZ/LAS Parsing**
- Fast memory‑mapped loading
- RGB, classification, intensity, and ID field support
- EXIF metadata extraction (GPS, timestamp)

### **✓ Spherical Range Projection**
- KITTI/Rangenet++‑style projection
- Returns range, vertex map, intensity map, class map, index map, and RGB

### **✓ Frame Generation Pipelines**
- **Random sampling** - Generate frames at random positions
- **Trajectory-based** - Generate per pose along a trajectory
- **Pole-based** - Generate around detected poles
- **Object detection** - Generate around COCO-detected objects

### **✓ Pose Transformations**
- Position + orientation perturbation (with configurable noise)
- Euler‑based noise injection
- Transform point clouds into LiDAR frame

### **✓ EXIF-enabled Image Saving**
- Location tagged based on sensor ETRS‑TM35FIN → WGS84 conversion
- Optional timestamp embedding
- GPS altitude and coordinate storage

### **✓ COCO Annotation Support**
- Full COCO detection workflow
- Label expansion for hierarchical class structures
- Automatic JSON annotation generation

### **✓ Reprojection Testing**
- Deprojection utilities for validation
- File matching and label verification
- Multiprocessing support for large datasets

### **✓ Performance**
- Memory‑mapped point clouds (efficient for 100M+ points)
- Optimized spherical projection (~1000 fps on modern CPU)
- Multi-threaded and multiprocessing support

---

## 🧩 Core API Overview

### **Load point clouds & trajectories**

```python
from rangeimagegenerator import (
    read_laz_file,
    read_trajectory_file,
    match_laz_and_trajectory,
    load_pointcloud,
)

# Load LAS file (memory-mapped by default)
points, header, memmap_file = read_laz_file("data/example.laz")

# Or use high-level loader
memmap_meta = load_pointcloud("data/example.laz", fields=["intensity", "classification"])

# Read trajectory (CSV, NMEA, or other formats)
trajectory = read_trajectory_file("data/trajectory.csv")
```

### **Match LAS files to trajectory files**

```python
from rangeimagegenerator import match_las_and_trajectory

pairs = match_las_and_trajectory("laz_folder", "trajectory_folder")
```

### **Generate frames with pose perturbation**

```python
from rangeimagegenerator import (
    FrameIntrinsics,
    frame_randomizer,
    batched_frame_randomizer,
)
from scipy.spatial.transform import Rotation as R
import numpy as np

# Example values
pos = np.array([0.0, 0.0, 0.0])
rot = R.identity()
frame_intrinsics = FrameIntrinsics(
    fov_down=-25.0,
    fov_up=3.0,
    height=64,
    width=1024,
    max_range=100.0,
)

# Create task
task = Frame_Task(
    basename="frame_0001",
    pose=(pos, rot),
    frame_intrinsics=frame_intrinsics,
    points=points,
    output_dir="output",
    position_perturbation=((0, 1), (0, 1), (0, 1)),  # x, y, z ranges
    orientation_perturbation=((0, 0), (0, 0), (0, 0)),  # roll, pitch, yaw
    number_of_pos_perturbations=3,
    number_of_orientation_perturbations=5,
)

# Generate single frame
images, annotations, output_dir = frame_randomizer(task)
```

### **Batch processing**

```python
from rangeimagegenerator import batched_frame_randomizer

# Batch multiple tasks
tasks = [task1, task2, task3]
results = batched_frame_randomizer(tasks, batch_size=10)
```

### **Reprojection testing**

```python
from rangeimagegenerator.reprojection import label_pointcloud_coco
from rangeimagegenerator.coco_parser import load_coco_annotations

# Load annotations and validate
detection_results, categories = label_pointcloud_coco(
    points, coco_file="coco_annotations.json", cfg=config
)
```

---

## ⚙️ Configuration File Example (`config.toml`)

```toml
[paths]
laz_dir = "data/laz/"
trajectory_dir = "data/trajectory/"
output_dir = "output_dataset"
coco_file = "annotations/coco.json"

[image]
width = 1024
height = 64
max_range = 100
elevation_range = [-25, 3]

[sensor]
sensor_orientation = [0, 0, 0]
use_trajectory_orientation = true

[processing]
pipeline = "random"  # Options: random, trajectory, poles, coco, detected
randomize = true
pos_pertubations = 0
orientaton_pertubations = 0
rand_range_x = [-2, 2]
rand_range_y = [-2, 2]
rand_range_z = [-1, 1]
rand_range_roll = [-1, 1]
rand_range_pitch = [-1, 1]
rand_range_yaw = [-1, 1]
skip_start = 0
skip_end = 0
stride_distance = 30
fields = ["intensity", "classification"]
number_of_processes = 1
batched = false
```

### **Configuration Options**

| Option | Default | Description |
|--------|---------|---------|
| `pipeline` | `"random"` | Generation pipeline: `random`, `trajectory`, `poles`, `coco`, `detected` |
| `randomize` | `true` | Enable pose randomization |
| `pos_pertubations` | `0` | Number of position perturbation variants |
| `orientaton_pertubations` | `0` | Number of orientation perturbation variants |
| `skip_start` | `0` | Skip first N trajectory points |
| `skip_end` | `0` | Skip last N trajectory points |
| `stride_distance` | `30` | Distance between consecutive frames (meters) |
| `use_trajectory_orientation` | `true` | Use trajectory rotation |
| `fields` | `["intensity", "classification"]` | LAS fields to extract |
| `number_of_processes` | `1` | Number of parallel processes |
| `batched` | `false` | Enable batch processing |
| `batch_size` | `0` | Batch size when batched is enabled |

See [`confs/`](confs/) for real-world configuration examples.

---

## 📊 Output Structure

Depending on script options and pipeline, you may see:

```
output/
  {las_file_basename}/
    random/                          # Random sampling output
      *.png                          # Range, RGB, intensity, classification images
      pose.txt                       # Pose information per frame
      frame_intrinsic/               # Intrinsic parameters per frame
      coco.json                      # COCO annotations (if applicable)
    trajectory/                      # Trajectory-based output
      ...
    poles/                           # Pole-based output
      ...
    detected/                        # Object detection-based output
      ...
```

Each image is saved as PNG (supports EXIF). EXIF includes:
- `gps_lat`, `gps_lon`: ETRS-TM35FIN → WGS84 converted coordinates
- `gps_alt`: altitude in meters
- `timestamp`: ISO 8601 timestamp

---

## 📁 Configuration Examples

The [`confs/`](confs/) directory contains:
- `config.toml` - Basic configuration
- `test.toml` - Test configuration
- `reprojection_testing.toml` - Reprojection testing configuration
- `vt22.toml`, `knuutilankangas.toml`, etc. - Sensor-specific configurations

Copy a config file and modify paths for your use case.

---

## 🧪 Quick Test

Generate a small sampling-based set:

```bash
python -m rangeimagegenerator reprojection_testing/config.toml
```

Then inspect images under `output_dir`.

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|-------|
| `LAS file not found` | Ensure `.las`/`.laz` files are in `laz_dir` |
| Empty output images | Check `elevation_range` matches your data |
| Slow processing | Disable randomization; process fewer frames; increase `number_of_processes` |
| EXIF not saved | Ensure `laspy` supports GPS metadata for your file |
| Import error: `rangeimagegenerator` | Ensure `src/` is in PYTHONPATH or use `pip install -e .` |
| Batch processing errors | Ensure `batch_size` divides evenly into total number of frames |

---

## 🔧 Advanced Usage

### **Coordinate Transformation**

```python
from rangeimagegenerator.pyprojCoordinateTranformer import ETRSTM35FINxy_to_WGS84lalo

# Convert coordinates
lat, lon = ETRSTM35FINxy_to_WGS84lalo(x, y, elevation)
```

### **NMEA Parsing**

```python
from rangeimagegenerator.x200go_nmea_parser import parse_nmea_file

poses = parse_nmea_file("trajectory.nmea")
```

### **Custom Frame Generation**

```python
from rangeimagegenerator import (
    transform_points,
    pertubate_position,
    pertubate_orientation,
    range_projection,
)

# Transform points to sensor frame
sensor_points = transform_points(world_points, position, orientation)

# Perturb pose
position_perturbed = pertubate_position(pos, ranges)
orientation_perturbed = pertubate_orientation(rot, ranges)
```

---

## 📚 Citation

If you use this toolkit in research, please cite:

```bibtex
@software{rangeimagegenerator,
  title = {RangeGen - LiDAR Range Image Generation Toolkit},
  author = {Timo Mäenpää},
  version = {0.1.0},
  year = {2026},
  url = {https://github.com/RangeGen/rangeimagegenerator}
}
```

---

## 📜 License

MIT License. See LICENSE file for details.

---

## 🙋 Support & Issues

Need help or found a bug? Open an issue on [GitHub](https://github.com/Aspor/RangeGen/issues).

---
---

## 📈 Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | TBD | Initial release |

---
