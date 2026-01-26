# Polesight Dataset

A curated dataset for ____________

## Overview

This dataset contains ________ organized into three modalities:

- **Intensity**: Grayscale intensity images from infrared sensor
- **Range**: Depth/range images from LiDAR Seonsor
- **RGB**: Color images captured from the RGB camera


## Dataset Structure

```
data/
├── intensity/    # Grayscale intensity images from infrared sensor
└── range/        # Depth/range images from LiDAR Sensor
└── rgb/          # RGB color images
```

Each subdirectory contains PNG images with same filenames following the pattern: `{scene_id}_{sub_id}_{frame_id}.png`

- `scene_id`: 5-digit scene identifier (e.g., 02194)
- `sub_id`: Sub-scene or sequence identifier (1-8)
- `frame_id`: Frame index within the sequence (0-9)

## Channel Descriptions

- **Intensity Images**: Single-channel grayscale PNG images from infrared sensor
- **Range Images**: Single-channel grayscale depth maps from LiDAR Sensor 
- **RGB Images**: 3-channel Color images

## Labels and Annotations






## Usage 
## Citation
## License

