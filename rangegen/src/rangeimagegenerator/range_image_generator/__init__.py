"""lidar_processing

A modular LiDAR processing toolkit for:
- Reading LAS/LAZ point clouds
- Transforming and perturbing poses
- Projecting point clouds into spherical range images
- Saving RGB, intensity, classification, and pose data
- Generating augmented frames via pose perturbations

Submodules:
    utils.py            → basic helpers, FrameIntrinsics, directory creation
    io_utils.py         → LAS reading, PCD writing, image/EXIF saving
    projection.py       → spherical projection + hole‑filling filter
    transforms.py       → pose transformations and perturbations
    frame_generator.py  → high‑level frame export pipeline
"""

from .utils import (
    FrameIntrinsics,
    enumerate2,
    make_output_dirs,
    Frame_Task,
    Pose,
    MemmapMetadata,
)

from .io_utils import (
    read_laz_file,
    write_pointcloud,
    read_trajectory_file,
    save_image,
    save_classification_image,
    match_las_and_trajectory,
)


from .transforms import (
    transform_points,
    perturb_position,
    perturb_orientation,
)

from .frame_generator import (
    frame_randomizer,
    batched_frame_randomizer,
)

from .projection import range_projection_idx_only

from .coco_writer import generate_coco, write_coco_json

__all__ = [
    # utils
    "FrameIntrinsics",
    "FrameIntrinsics_dtype",
    "enumerate2",
    "make_output_dirs",
    "match_las_and_trajectory"
    # io
    "read_laz_file",
    "write_pointcloud",
    "read_trajectory_file",
    "save_image",
    "save_classification_image",
    # projection
    "range_projection",
    "filter_invalid_points",
    # transforms
    "transform_points",
    "perturb_position",
    "perturb_orientation",
    # frame generation
    "frame_randomizer",
    # coco
    "write_coco_json",
    "generate_coco",
    #
    "batched_frame_randomizer",
    # utilities
    "FrameIntrinsics",
    "enumerate2",
    "make_output_dirs",
    "Frame_Task",
    "Pose",
    "MemmapMetadata",
]
