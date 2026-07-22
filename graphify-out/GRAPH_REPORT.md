# Graph Report - .  (2026-07-22)

## Corpus Check
- Large corpus: 1065 files · ~2,003,912 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 311 nodes · 524 edges · 20 communities
- Extraction: 71% EXTRACTED · 26% INFERRED · 3% AMBIGUOUS · INFERRED: 136 edges (avg confidence: 0.83)
- Token cost: 419,241 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Label Validation CLI|Label Validation CLI]]
- [[_COMMUNITY_Dataset Analysis Script|Dataset Analysis Script]]
- [[_COMMUNITY_Tuning Config & Dataset Spec|Tuning Config & Dataset Spec]]
- [[_COMMUNITY_Annotated Frame Samples|Annotated Frame Samples]]
- [[_COMMUNITY_Box Area by Class|Box Area by Class]]
- [[_COMMUNITY_Box Dimension Distributions|Box Dimension Distributions]]
- [[_COMMUNITY_Width-Height Scatter Geometry|Width-Height Scatter Geometry]]
- [[_COMMUNITY_Class Co-occurrence Matrix|Class Co-occurrence Matrix]]
- [[_COMMUNITY_Class Distribution by Split|Class Distribution by Split]]
- [[_COMMUNITY_Object Center Spatial Bias|Object Center Spatial Bias]]
- [[_COMMUNITY_Class Split Composition|Class Split Composition]]
- [[_COMMUNITY_TrainValTest Split Sizes|Train/Val/Test Split Sizes]]
- [[_COMMUNITY_Overall Class Imbalance|Overall Class Imbalance]]
- [[_COMMUNITY_Annotation Density per Image|Annotation Density per Image]]
- [[_COMMUNITY_Annotation Coverage & Backgrounds|Annotation Coverage & Backgrounds]]
- [[_COMMUNITY_Image File Size Audit|Image File Size Audit]]
- [[_COMMUNITY_Hyperparameter Tuner|Hyperparameter Tuner]]
- [[_COMMUNITY_Roboflow Dataset Provenance|Roboflow Dataset Provenance]]
- [[_COMMUNITY_Multimodal Sensor Modalities|Multimodal Sensor Modalities]]

## God Nodes (most connected - your core abstractions)
1. `Sample Annotated Frames Montage (intensity, boxes coloured by class)` - 15 edges
2. `Class Distribution by Split (grouped bar chart)` - 13 edges
3. `Figure: Bounding-box Area by Class (box plot)` - 12 edges
4. `Bounding-box dimension distributions (normalized) figure` - 12 edges
5. `Class Composition Across Splits (100% Stacked Bar Chart)` - 12 edges
6. `main()` - 11 edges
7. `space (active 22-parameter search space)` - 11 edges
8. `Object Center Spatial Distribution Heatmap` - 11 edges
9. `build_report()` - 10 edges
10. `generate_figures()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `Native 960x640 resized to 640x640 for training` --shares_data_with--> `train (fixed training args held constant across trials)`  [INFERRED]
  data/dataset_analysis_report.html → tune.yaml
- `Figure: Object center spatial distribution` --conceptually_related_to--> `space (active 22-parameter search space)`  [INFERRED]
  data/dataset_analysis_report.html → tune.yaml
- `Severe class imbalance (power_pole = 7 instances)` --rationale_for--> `space (active 22-parameter search space)`  [INFERRED]
  data/dataset_analysis_report.html → tune.yaml
- `Task-type mismatch: report says Object Detection, tuning says segment` --conceptually_related_to--> `Segmentation fitness = mask mAP50-95 + box mAP50-95`  [AMBIGUOUS]
  data/dataset_analysis_report.html → tune.yaml
- `Gravity-aligned pole prior (flipud = 0)` --rationale_for--> `5-class pole taxonomy (fence/gantry_sign/light/power/traffic)`  [INFERRED]
  tune.yaml → data/data.yaml

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Multimodal Sensor Inputs (intensity/range/RGB)** — readme_intensity_modality, readme_range_modality, readme_rgb_modality [EXTRACTED 1.00]
- **polesight hyperparameter tuning pipeline (config, data, evaluation)** — tune_tuner_meta, tune_train_fixed_args, tune_space, tune_segmentation_fitness, data_data_dataset_config [EXTRACTED 1.00]
- **Small-dataset design constraints shaping the search space** — tune_small_dataset_strategy, tune_gravity_aligned_pole_prior, tune_rtx_2080ti_memory_budget, tune_yolo26_recipe_anchors, tune_non_searchable_params [INFERRED 0.85]
- **Dataset characterization evidence from the analysis report** — data_dataset_analysis_report_split_distribution, data_dataset_analysis_report_class_distribution, data_dataset_analysis_report_class_imbalance, data_dataset_analysis_report_bbox_statistics, data_dataset_analysis_report_background_images [EXTRACTED 1.00]
- **Train/valid/test partition of the pole dataset (412/52/51 images)** — figure_annotation_coverage_train_split, figure_annotation_coverage_valid_split, figure_annotation_coverage_test_split, figure_annotation_coverage_yolo_dataset [INFERRED 0.90]
- **Stacked-bar composition: annotated vs background images counted per split** — figure_annotation_coverage_annotated_images, figure_annotation_coverage_background_images, figure_annotation_coverage_metric_image_counts, figure_annotation_coverage_chart [EXTRACTED 1.00]
- **Five-class pole taxonomy compared on box-area distribution** — figure_bbox_area_by_class_class_fence_pole, figure_bbox_area_by_class_class_gantry_sign_pole, figure_bbox_area_by_class_class_light_pole, figure_bbox_area_by_class_class_power_pole, figure_bbox_area_by_class_class_traffic_pole, figure_bbox_area_by_class_metric_normalized_area [EXTRACTED 1.00]
- **Scale-distribution findings driving detector design choices** — figure_bbox_area_by_class_insight_size_heterogeneity, figure_bbox_area_by_class_insight_small_object_skew, figure_bbox_area_by_class_insight_light_pole_outliers, figure_bbox_area_by_class_insight_power_pole_largest, figure_bbox_area_by_class_multiscale_training_implication [INFERRED 0.75]
- **Four normalized box-geometry histogram panels sharing one annotation pool** — figure_bbox_dimension_distributions_width_panel, figure_bbox_dimension_distributions_height_panel, figure_bbox_dimension_distributions_area_panel, figure_bbox_dimension_distributions_aspect_ratio_panel [INFERRED 0.95]
- **All five pole classes occupy the same width-height region, making box geometry alone non-discriminative** — figure_bbox_width_height_scatter_class_fence_pole, figure_bbox_width_height_scatter_class_gantry_sign_pole, figure_bbox_width_height_scatter_class_light_pole, figure_bbox_width_height_scatter_class_power_pole, figure_bbox_width_height_scatter_class_traffic_pole, figure_bbox_width_height_scatter_class_overlap_insight [INFERRED 0.85]
- **Dataset-statistics pipeline: YOLO labels parsed by dataset_analysis.py into normalized w/h scatter figure** — figure_bbox_width_height_scatter_figure, figure_bbox_width_height_scatter_dataset_analysis_script, figure_bbox_width_height_scatter_yolo_label_source, figure_bbox_width_height_scatter_bbox_dimension_metric [INFERRED 0.85]
- **Five pole classes forming the co-occurrence matrix axes** — figure_class_cooccurrence_fence_pole, figure_class_cooccurrence_gantry_sign_pole, figure_class_cooccurrence_light_pole, figure_class_cooccurrence_power_pole, figure_class_cooccurrence_traffic_pole [EXTRACTED 1.00]
- **light_pole co-occurs with every other class, acting as the dataset hub class** — figure_class_cooccurrence_light_pole, figure_class_cooccurrence_fence_pole, figure_class_cooccurrence_gantry_sign_pole, figure_class_cooccurrence_traffic_pole, figure_class_cooccurrence_multilabel_scenes [INFERRED 0.85]
- **Five-class pole taxonomy of the PoleSight dataset** — figure_class_distribution_by_split_class_fence_pole, figure_class_distribution_by_split_class_gantry_sign_pole, figure_class_distribution_by_split_class_light_pole, figure_class_distribution_by_split_class_power_pole, figure_class_distribution_by_split_class_traffic_pole [EXTRACTED 1.00]
- **Train/valid/test partition of the dataset** — figure_class_distribution_by_split_split_train, figure_class_distribution_by_split_split_valid, figure_class_distribution_by_split_split_test [EXTRACTED 1.00]
- **Five-class pole taxonomy of the PoleSight detection dataset** — figure_class_distribution_overall_light_pole, figure_class_distribution_overall_fence_pole, figure_class_distribution_overall_traffic_pole, figure_class_distribution_overall_gantry_sign_pole, figure_class_distribution_overall_power_pole [EXTRACTED 1.00]
- **Long-tail minority classes under 11% of annotations** — figure_class_distribution_overall_traffic_pole, figure_class_distribution_overall_gantry_sign_pole, figure_class_distribution_overall_power_pole, figure_class_distribution_overall_class_imbalance [INFERRED 0.85]
- **Five-class pole taxonomy of the detection dataset** — figure_class_split_composition_traffic_pole, figure_class_split_composition_power_pole, figure_class_split_composition_light_pole, figure_class_split_composition_gantry_sign_pole, figure_class_split_composition_fence_pole [EXTRACTED 1.00]
- **Train/valid/test partition summing to 100% of instances per class** — figure_class_split_composition_train_split, figure_class_split_composition_valid_split, figure_class_split_composition_test_split [EXTRACTED 1.00]
- **Classes following the ~80/10/10 train/valid/test ratio** — figure_class_split_composition_traffic_pole, figure_class_split_composition_light_pole, figure_class_split_composition_gantry_sign_pole, figure_class_split_composition_fence_pole [INFERRED 0.85]
- **Spatial Prior Evidence: Vertical Band, Empty Bottom, Bimodal Lateral Hotspots** — figure_object_center_heatmap_upper_middle_band_concentration, figure_object_center_heatmap_bottom_third_empty, figure_object_center_heatmap_bimodal_lateral_hotspots, figure_object_center_heatmap_left_edge_column_spike, figure_object_center_heatmap_positional_bias_risk [INFERRED 0.85]
- **Annotation Density Profile of the Pole Dataset** — figure_objects_per_image_metric_objects_per_image, figure_objects_per_image_median_three, figure_objects_per_image_right_skewed_distribution, figure_objects_per_image_crowded_scene_tail, figure_objects_per_image_sparse_scene_dominance [INFERRED 0.85]
- **Five-class pole taxonomy jointly annotated across sampled frames** — figure_sample_annotated_frames_class_fence_pole, figure_sample_annotated_frames_class_gantry_sign_pole, figure_sample_annotated_frames_class_light_pole, figure_sample_annotated_frames_class_power_pole, figure_sample_annotated_frames_class_traffic_pole [EXTRACTED 1.00]
- **Six sampled LiDAR intensity frames composing the montage** — figure_sample_annotated_frames_frame_05371, figure_sample_annotated_frames_frame_1550, figure_sample_annotated_frames_frame_0053, figure_sample_annotated_frames_frame_0085, figure_sample_annotated_frames_frame_1544, figure_sample_annotated_frames_frame_0419 [EXTRACTED 1.00]
- **Annotation-quality signals visible in the montage** — figure_sample_annotated_frames_oversized_box_artifact, figure_sample_annotated_frames_panorama_seam_wraparound, figure_sample_annotated_frames_label_text_occlusion, figure_sample_annotated_frames_sparse_point_density [INFERRED 0.85]
- **Split composition measured jointly by image count and instance count** — figure_split_distribution_train_split, figure_split_distribution_valid_split, figure_split_distribution_test_split, figure_split_distribution_eighty_ten_ten_ratio [EXTRACTED 1.00]

## Communities (20 total, 0 thin omitted)

### Community 0 - "Label Validation CLI"
Cohesion: 0.11
Nodes (35): download_from_roboflow(), FileReport, load_class_names(), load_env(), load_roboflow_config(), main(), parse_args(), Namespace (+27 more)

### Community 1 - "Dataset Analysis Script"
Cohesion: 0.09
Nodes (37): add_sample_montage(), analyze_bounding_boxes(), analyze_image_file_sizes(), analyze_set(), _apply_style(), build_report(), calc_stats(), class_color() (+29 more)

### Community 2 - "Tuning Config & Dataset Spec"
Cohesion: 0.08
Nodes (37): polesight dataset config (data.yaml), 5-class pole taxonomy (fence/gantry_sign/light/power/traffic), Roboflow provenance (polesight-7kgwj v9), train/valid/test split paths, YOLO normalized-coordinate annotation format, Background (empty-label) images: 28 of 515, Bounding-box statistics (aspect ratio mean 2.37, max 125.88), Figure: Object center spatial distribution (+29 more)

### Community 3 - "Annotated Frame Samples"
Cohesion: 0.20
Nodes (24): Per-Class Colour Legend (5 pole classes), Class: fence_pole (blue), Class: gantry_sign_pole (teal), Class: light_pole (orange), Class: power_pole (green), Class: traffic_pole (purple), dataset_analysis.py Figure Generator, Sample Annotated Frames Montage (intensity, boxes coloured by class) (+16 more)

### Community 4 - "Box Area by Class"
Cohesion: 0.18
Nodes (16): Box-plot Encoding (median, IQR, whiskers, outliers), Figure: Bounding-box Area by Class (box plot), Class: fence_pole, Class: gantry_sign_pole, Class: light_pole, Class: power_pole, Class: traffic_pole, Generator: dataset_analysis.py (+8 more)

### Community 5 - "Box Dimension Distributions"
Cohesion: 0.19
Nodes (16): Implication for anchor sizing, input resolution and scale augmentation, Panel: normalized box area (median 0.14), Panel: aspect ratio w/h (median 1.29, long tail to 25), Bounding-box dimension distributions (normalized) figure, Class- and split-agnostic pooling of all annotation instances, dataset_analysis.py figure generator, Panel: normalized box height (median 0.35), Histogram small-multiples chart type (2x2 panel grid) (+8 more)

### Community 6 - "Width-Height Scatter Geometry"
Cohesion: 0.17
Nodes (16): Anchor / Aspect-ratio Prior Selection for Detector, Normalized Bounding-box Width and Height Metric, Class: fence_pole, Class: gantry_sign_pole, Class: light_pole, Insight: Heavy Inter-class Overlap in Box Geometry, Class: power_pole, Class: traffic_pole (+8 more)

### Community 7 - "Class Co-occurrence Matrix"
Cohesion: 0.30
Nodes (14): Blues Sequential Colormap with 'Images' Colorbar (0-467), Class Imbalance in the Pole Detection Dataset, dataset_analysis.py (figure generator), fence_pole (class), Class Co-occurrence Heatmap Figure, gantry_sign_pole (class), light_pole (class), light_pole Dominates the Dataset (467 images, diagonal maximum) (+6 more)

### Community 8 - "Class Distribution by Split"
Cohesion: 0.36
Nodes (14): Class Distribution by Split (grouped bar chart), Class: fence_pole, Class: gantry_sign_pole, Rationale: severe class imbalance (light_pole dominates, power_pole near-absent), Class: light_pole, Class: power_pole, Class: traffic_pole, dataset_analysis.py (figure generator) (+6 more)

### Community 9 - "Object Center Spatial Bias"
Cohesion: 0.21
Nodes (14): 2D Histogram Heatmap of Normalized Bounding Box Centers, Bimodal Lateral Hotspots at x approx 0.15-0.30 and x approx 0.70-0.90, Bottom Third of Frame (y > 0.65) Is Almost Empty of Object Centers, dataset_analysis.py Heatmap Generator, Object Center Spatial Distribution Heatmap, Elevated Counts Hugging the Left Frame Edge (x approx 0.0), Objects Count Colorbar, Blues Colormap 0 to 30, Utility Pole Object Detection Dataset (+6 more)

### Community 10 - "Class Split Composition"
Cohesion: 0.38
Nodes (13): Class Composition Across Splits (100% Stacked Bar Chart), dataset_analysis.py (figure generator), fence_pole (class), gantry_sign_pole (class), light_pole (class), power_pole (class), power_pole split imbalance insight (57/29/14 vs ~80/10/10 norm), Share of Instances (%) per Class per Split (+5 more)

### Community 11 - "Train/Val/Test Split Sizes"
Cohesion: 0.35
Nodes (12): Train/Validation/Test Split Distribution Chart, dataset_analysis.py (Figure Generator), Small Dataset Scale (515 images, 1909 pole instances), 80/10/10 Split Ratio Convention, Images-per-Split Bar Panel, Instance-Density Imbalance Across Splits, Instances-per-Split Bar Panel, Small Holdout Sets Limit Evaluation Reliability (+4 more)

### Community 12 - "Overall Class Imbalance"
Cohesion: 0.31
Nodes (11): Class Distribution (All Splits) Bar Chart, Severe Class Imbalance: 154:1 head-to-tail ratio (light_pole vs power_pole), dataset_analysis.py (matplotlib figure generator), fence_pole (486 instances, 25.5%), gantry_sign_pole (139 instances, 7.3%), Instance Count per Class Metric (1909 total annotations), light_pole (1077 instances, 56.4%), Rare tail classes may need oversampling, class weighting, or merging before training (+3 more)

### Community 13 - "Annotation Density per Image"
Cohesion: 0.29
Nodes (11): Annotation Density Histogram (objects per image), Rare Crowded Scenes Up To ~17 Objects, dataset_analysis.py (Figure Generator), Image Count (Y Axis, ~0-100 Images Per Bin), Median 3 Objects Per Image, Metric: Objects Per Image, Rationale: Density Informs max_det / NMS and Mosaic Augmentation Settings, Utility Pole Infrastructure Detection Domain (+3 more)

### Community 14 - "Annotation Coverage & Backgrounds"
Cohesion: 0.31
Nodes (10): Annotated images category (blue series), Background images category (red series, empty label files), Annotation coverage per split (stacked bar chart), dataset_analysis.py (chart generator), Insight: ~80/10/10 train/valid/test split with high annotation coverage (~95%), Metric: image count per split by annotation status, Test split (48 annotated, 3 background), Train split (390 annotated, 22 background) (+2 more)

### Community 15 - "Image File Size Audit"
Cohesion: 0.31
Nodes (10): Average File Size Marker at 100 KB, Insight: Right-Skewed Bimodal Peaks near 100 KB and 125 KB, Image File-Size Distribution Histogram, dataset_analysis.py Chart Generator, Dataset Homogeneity and Quality Audit, File Size in KB (x-axis metric), Image Count per Size Bin (y-axis metric), Matplotlib Histogram Rendering (green bars, dashed red reference line) (+2 more)

### Community 16 - "Hyperparameter Tuner"
Cohesion: 0.33
Nodes (3): ConfigTuner, Tuner that snaps declared hyperparameters to integers.      Ultralytics' Tuner i, Tuner

### Community 17 - "Roboflow Dataset Provenance"
Cohesion: 0.60
Nodes (5): ROBOFLOW_API key in .env, data/ directory (regenerable dataset), Roboflow Dataset Coordinates, Roboflow Project polesight-7kgwj, Roboflow Workspace polesight

### Community 18 - "Multimodal Sensor Modalities"
Cohesion: 0.50
Nodes (4): Intensity Modality (grayscale infrared images), Polesight Dataset (multimodal pole-detection dataset), Range Modality (LiDAR depth images), RGB Modality (color camera images)

## Ambiguous Edges - Review These
- `Segmentation fitness = mask mAP50-95 + box mAP50-95` → `Task-type mismatch: report says Object Detection, tuning says segment`  [AMBIGUOUS]
  data/dataset_analysis_report.html · relation: conceptually_related_to
- `YOLO normalized-coordinate annotation format` → `Task-type mismatch: report says Object Detection, tuning says segment`  [AMBIGUOUS]
  data/dataset_analysis_report.html · relation: references
- `Figure: Bounding-box Area by Class (box plot)` → `Implication: Multi-scale Augmentation / Anchor Sizing for Detector Training`  [AMBIGUOUS]
  .figure/bbox_area_by_class.png · relation: conceptually_related_to
- `Bounding-box dimension distributions (normalized) figure` → `Implication for anchor sizing, input resolution and scale augmentation`  [AMBIGUOUS]
  .figure/bbox_dimension_distributions.png · relation: rationale_for
- `Insight: width distribution is broad/multi-modal with spikes near 0.0, 0.3 and 0.9` → `Class- and split-agnostic pooling of all annotation instances`  [AMBIGUOUS]
  .figure/bbox_dimension_distributions.png · relation: conceptually_related_to
- `Bounding-box Width vs Height Scatter (Figure)` → `Split Coverage Unlabeled (no train/val/test breakdown shown)`  [AMBIGUOUS]
  .figure/bbox_width_height_scatter.png · relation: conceptually_related_to
- `Normalized Bounding-box Width and Height Metric` → `Ambiguity: Axes Span Full 0-1 Range (min-max vs raw YOLO fraction)`  [AMBIGUOUS]
  .figure/bbox_width_height_scatter.png · relation: rationale_for
- `Blues Sequential Colormap with 'Images' Colorbar (0-467)` → `light_pole Dominates the Dataset (467 images, diagonal maximum)`  [AMBIGUOUS]
  .figure/class_cooccurrence.png · relation: conceptually_related_to
- `Class: power_pole` → `Split: test`  [AMBIGUOUS]
  .figure/class_distribution_by_split.png · relation: shares_data_with
- `Class: power_pole` → `Split: train`  [AMBIGUOUS]
  .figure/class_distribution_by_split.png · relation: shares_data_with
- `Class: power_pole` → `Split: valid`  [AMBIGUOUS]
  .figure/class_distribution_by_split.png · relation: shares_data_with
- `fence_pole (486 instances, 25.5%)` → `gantry_sign_pole (139 instances, 7.3%)`  [AMBIGUOUS]
  .figure/class_distribution_overall.png · relation: semantically_similar_to
- `Image File-Size Distribution Histogram` → `Insight: Right-Skewed Bimodal Peaks near 100 KB and 125 KB`  [AMBIGUOUS]
  .figure/image_file_size_distribution.png · relation: conceptually_related_to
- `Bimodal Lateral Hotspots at x approx 0.15-0.30 and x approx 0.70-0.90` → `Elevated Counts Hugging the Left Frame Edge (x approx 0.0)`  [AMBIGUOUS]
  .figure/object_center_heatmap.png · relation: conceptually_related_to
- `Annotation Density Histogram (objects per image)` → `Rationale: Density Informs max_det / NMS and Mosaic Augmentation Settings`  [AMBIGUOUS]
  .figure/objects_per_image.png · relation: conceptually_related_to
- `Annotation Issue: Boxes Far Larger Than Thin Pole Structures` → `Annotation Issue: Boxes Running Off Image Edge (panorama seam wrap-around)`  [AMBIGUOUS]
  .figure/sample_annotated_frames.png · relation: rationale_for

## Knowledge Gaps
- **30 isolated node(s):** `Intensity Modality (grayscale infrared images)`, `Range Modality (LiDAR depth images)`, `RGB Modality (color camera images)`, `dataset_analysis.py (report generator)`, `Background (empty-label) images: 28 of 515` (+25 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Segmentation fitness = mask mAP50-95 + box mAP50-95` and `Task-type mismatch: report says Object Detection, tuning says segment`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `YOLO normalized-coordinate annotation format` and `Task-type mismatch: report says Object Detection, tuning says segment`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **What is the exact relationship between `Figure: Bounding-box Area by Class (box plot)` and `Implication: Multi-scale Augmentation / Anchor Sizing for Detector Training`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Bounding-box dimension distributions (normalized) figure` and `Implication for anchor sizing, input resolution and scale augmentation`?**
  _Edge tagged AMBIGUOUS (relation: rationale_for) - confidence is low._
- **What is the exact relationship between `Insight: width distribution is broad/multi-modal with spikes near 0.0, 0.3 and 0.9` and `Class- and split-agnostic pooling of all annotation instances`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Bounding-box Width vs Height Scatter (Figure)` and `Split Coverage Unlabeled (no train/val/test breakdown shown)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Normalized Bounding-box Width and Height Metric` and `Ambiguity: Axes Span Full 0-1 Range (min-max vs raw YOLO fraction)`?**
  _Edge tagged AMBIGUOUS (relation: rationale_for) - confidence is low._