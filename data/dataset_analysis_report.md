# Dataset Analysis Report

Instance segmentation · 4 classes · generated 2026-09-07 16:09

## Key figures

| Metric | Value | Detail |
| :--- | ---: | :--- |
| Images | 1,088 | 1,059 annotated · 29 background |
| Annotations | 6,121 | 6,121 polygons · 0 box rows |
| Classes | 4 | fence_pole, gantry_sign_pole, light_pole, traffic_pole |
| Instances / image | 5.8 | median 5 · max 21 |
| Mask fill ratio | 0.28 | median · 10 vertices typical |
| Resolution | 1024 x 128 pixels | PNG · 16-bit grayscale (single channel) |
| Avg file size | 109 KB | 8–234 KB |
| Validation issues | 0 | clean |

## Dataset properties

| Property | Value |
| :--- | :--- |
| Dataset type | Instance Segmentation |
| Annotation format | YOLO instance segmentation (normalized polygons) |
| Image format | PNG |
| Color space | 16-bit grayscale (single channel) |
| Resolution | 1024 x 128 pixels |
| Aspect ratio | 8:1 (8.00:1) |
| Training resolution | 640 x 640 pixels (resized) |

## Image properties

Pixel modes present: I;16 (1,088)

| Resolution | Images | Share |
| :--- | ---: | ---: |
| 1024x128 | 1,088 | 100.0% |
| **total** | **1,088** | **100%** |

## Split distribution

| Split | Images | Img % | Annotated | Background | Instances | Inst % |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 870 | 80.0% | 847 | 23 | 4,900 | 80.0% |
| valid | 109 | 10.0% | 106 | 3 | 610 | 10.0% |
| test | 109 | 10.0% | 106 | 3 | 611 | 10.0% |
| **total** | **1,088** | **100%** | **1,059** | **29** | **6,121** | **100%** |

## Class distribution

| Class | Instances | Polygons | Box rows | Share |
| :--- | ---: | ---: | ---: | ---: |
| fence_pole | 961 | 961 | 0 | 15.7% |
| gantry_sign_pole | 782 | 782 | 0 | 12.8% |
| light_pole | 3,141 | 3,141 | 0 | 51.3% |
| traffic_pole | 1,237 | 1,237 | 0 | 20.2% |

## Instance geometry

Bounding boxes are derived from polygon extents and reported in pixels.

| Measure | Mean | Std | Min | Median | Max |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Width (px) | 19.0 | 30.1 | 1.0 | 10.0 | 428.0 |
| Height (px) | 28.9 | 21.7 | 1.0 | 23.0 | 127.0 |
| Bounding-box area (px²) | 775 | 2238 | 7 | 240 | 53500 |
| Aspect ratio (w/h, px) | 0.80 | 1.26 | 0.03 | 0.47 | 31.00 |
| Instances / image | 5.8 | 3.7 | 1.0 | 5.0 | 21.0 |

## Segmentation geometry

| Measure | Mean | Std | Min | Median | Max |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Mask area (px²) | 167 | 494 | 0 | 71 | 22762 |
| Mask fill ratio | 0.346 | 0.227 | 0.003 | 0.285 | 1.000 |
| Vertices per polygon | 13.5 | 12.4 | 3.0 | 10.0 | 166.0 |

Per class (medians; size split uses < 1024 px² annotation area / >= 9216 px² annotation area):

| Class | Polygons | Mask area (px²) | Fill ratio | Vertices | S / M / L |
| :--- | ---: | ---: | ---: | ---: | ---: |
| fence_pole | 961 | 38 | 0.45 | 8 | 959 / 2 / 0 |
| gantry_sign_pole | 782 | 220 | 0.20 | 18 | 695 / 84 / 3 |
| light_pole | 3,141 | 70 | 0.22 | 9 | 3,116 / 25 / 0 |
| traffic_pole | 1,237 | 58 | 0.48 | 9 | 1,218 / 19 / 0 |

## Data validation

No annotation issues found.

## Figures

### Sample annotated frames

Qualitative examples with ground-truth polygons coloured by class.

![Sample annotated frames](../paper/figures/dataset/sample_annotated_frames.png)

### Class distribution (all splits)

Instance count per class across the whole dataset.

![Class distribution (all splits)](../paper/figures/dataset/class_distribution_overall.png)

### Class distribution by split

Per-class instance counts broken down by train / validation / test.

![Class distribution by split](../paper/figures/dataset/class_distribution_by_split.png)

### Train / validation / test split

Share of images and instances in each split.

![Train / validation / test split](../paper/figures/dataset/split_distribution.png)

### Annotation coverage per split

Annotated vs. background (empty-label) images in each split.

![Annotation coverage per split](../paper/figures/dataset/annotation_coverage.png)

### Class composition across splits

For each class, how its instances are stratified across splits.

![Class composition across splits](../paper/figures/dataset/class_split_composition.png)

### Annotation density

Distribution of the number of instances per annotated image.

![Annotation density](../paper/figures/dataset/objects_per_image.png)

### Instance geometry

Width, height, area and aspect ratio in pixels, derived from polygon extents.

![Instance geometry](../paper/figures/dataset/bbox_dimension_distributions.png)

### Width vs. height by class

Instance width against height in pixels, coloured by class.

![Width vs. height by class](../paper/figures/dataset/bbox_width_height_scatter.png)

### Bounding-box area by class

Size profile of each class's bounding boxes (log scale).

![Bounding-box area by class](../paper/figures/dataset/bbox_area_by_class.png)

### Mask area by class

True polygon area per class — the segmentation size profile (log scale).

![Mask area by class](../paper/figures/dataset/mask_area_by_class.png)

### Instance size categories

Small / medium / large instances per class, using COCO's native-pixel area cutoffs.

![Instance size categories](../paper/figures/dataset/instance_size_categories.png)

### Mask fill ratio

Polygon area divided by bounding-box area — how tightly masks fit their boxes.

![Mask fill ratio](../paper/figures/dataset/mask_fill_ratio_by_class.png)

### Polygon vertex counts

How many vertices annotators used per mask; a proxy for annotation detail.

![Polygon vertex counts](../paper/figures/dataset/polygon_vertex_distribution.png)

### Mask centroid spatial distribution

Where object centroids fall within the frame.

![Mask centroid spatial distribution](../paper/figures/dataset/object_center_heatmap.png)

### Class co-occurrence

How often pairs of classes appear together in the same image.

![Class co-occurrence](../paper/figures/dataset/class_cooccurrence.png)

### Image size

Resolutions present in the dataset and their image counts.

![Image size](../paper/figures/dataset/image_size_distribution.png)

### Annotation validation issues

Invalid, malformed or suspect annotations found, by type and split.

![Annotation validation issues](../paper/figures/dataset/annotation_issues.png)

### Image file-size distribution

Distribution of image file sizes on disk (KB).

![Image file-size distribution](../paper/figures/dataset/image_file_size_distribution.png)

---

Generated by `dataset_analysis.py` · 2026-09-07 16:09
