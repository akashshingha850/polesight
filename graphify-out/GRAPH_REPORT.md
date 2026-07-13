# Graph Report - .  (2026-07-13)

## Corpus Check
- 9 files · ~4,059 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 77 nodes · 115 edges · 11 communities (10 shown, 1 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 6 edges (avg confidence: 0.83)
- Token cost: 0 input · 27,776 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Dataset Config & Pole Taxonomy|Dataset Config & Pole Taxonomy]]
- [[_COMMUNITY_TrainValTest Splitting|Train/Val/Test Splitting]]
- [[_COMMUNITY_Draft Restoration|Draft Restoration]]
- [[_COMMUNITY_Label Validation & Cleanup|Label Validation & Cleanup]]
- [[_COMMUNITY_Dataset InstanceSize Analysis|Dataset Instance/Size Analysis]]
- [[_COMMUNITY_Bounding-Box Statistics|Bounding-Box Statistics]]
- [[_COMMUNITY_Annotation Counting|Annotation Counting]]
- [[_COMMUNITY_Label Renaming|Label Renaming]]
- [[_COMMUNITY_Analysis Entry & Tree Printing|Analysis Entry & Tree Printing]]
- [[_COMMUNITY_LaTeX Summary Export|LaTeX Summary Export]]
- [[_COMMUNITY_Class Loading from YAML|Class Loading from YAML]]

## God Nodes (most connected - your core abstractions)
1. `main()` - 9 edges
2. `data.yaml Dataset Configuration` - 9 edges
3. `main()` - 7 edges
4. `comprehensive_bbox_analysis()` - 7 edges
5. `Dataset Analysis Report` - 7 edges
6. `main()` - 5 edges
7. `analyze_set()` - 5 edges
8. `analyze_data_instances()` - 5 edges
9. `parse_args()` - 4 edges
10. `resolve_split_dir()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `Image Specification (960x640 orig, 640x640 train, RGB, JPEG)` --semantically_similar_to--> `RGB Modality (color camera images)`  [INFERRED] [semantically similar]
  data/dataset_analysis_report.txt → README.md
- `Dataset Analysis Report` --conceptually_related_to--> `Polesight Dataset (multimodal pole-detection dataset)`  [INFERRED]
  data/dataset_analysis_report.txt → README.md
- `data.yaml Dataset Configuration` --references--> `Test Split (51 images, 203 instances, 10%)`  [EXTRACTED]
  data/data.yaml → data/dataset_analysis_report.txt
- `data.yaml Dataset Configuration` --references--> `Train Split (412 images, 1518 instances, 80%)`  [EXTRACTED]
  data/data.yaml → data/dataset_analysis_report.txt
- `data.yaml Dataset Configuration` --references--> `Validation Split (52 images, 188 instances, 10%)`  [EXTRACTED]
  data/data.yaml → data/dataset_analysis_report.txt

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Pole Class Taxonomy (5 detection classes)** — data_data_yaml_fence_pole, data_data_yaml_gantry_sign_pole, data_data_yaml_light_pole, data_data_yaml_power_pole, data_data_yaml_traffic_pole [EXTRACTED 1.00]
- **Train/Valid/Test Dataset Partition** — data_dataset_analysis_report_train_split, data_dataset_analysis_report_valid_split, data_dataset_analysis_report_test_split [EXTRACTED 1.00]
- **Multimodal Sensor Inputs (intensity/range/RGB)** — readme_intensity_modality, readme_range_modality, readme_rgb_modality [EXTRACTED 1.00]

## Communities (11 total, 1 thin omitted)

### Community 0 - "Dataset Config & Pole Taxonomy"
Cohesion: 0.14
Nodes (19): data.yaml Dataset Configuration, fence_pole (class 0), gantry_sign_pole (class 1), light_pole (class 2), power_pole (class 3), Roboflow Project Source (polesight-7kgwj v9), traffic_pole (class 4), Bounding Box Statistics (normalized width/height/area/aspect ratio) (+11 more)

### Community 1 - "Train/Val/Test Splitting"
Cohesion: 0.38
Nodes (9): build_image_index(), build_label_index(), copy_sample(), main(), parse_args(), Namespace, Path, reset_output_dirs() (+1 more)

### Community 2 - "Draft Restoration"
Cohesion: 0.39
Nodes (7): collect_images(), ensure_destination_dirs(), main(), parse_args(), Namespace, Path, restore_split()

### Community 3 - "Label Validation & Cleanup"
Cohesion: 0.57
Nodes (6): has_too_few_values(), main(), parse_args(), Namespace, Path, remove_problematic_lines()

### Community 4 - "Dataset Instance/Size Analysis"
Cohesion: 0.47
Nodes (5): analyze_data_instances(), analyze_image_file_sizes(), Analyze instances per class for each data split., Analyze file sizes of all images in the dataset., resolve_split_dir()

### Community 5 - "Bounding-Box Statistics"
Cohesion: 0.33
Nodes (6): analyze_bounding_boxes(), calc_stats(), comprehensive_bbox_analysis(), Analyze bounding box statistics from YOLO format labels., Calculate basic statistics for a list of data., Perform comprehensive bounding box analysis across all splits.

### Community 6 - "Annotation Counting"
Cohesion: 0.33
Nodes (6): analyze_set(), count_annotated_images(), count_files(), Count images that have corresponding non-empty annotation files., Analyze a specific dataset set (train/val/test), Counts files with a given extension in a directory and its subdirectories.

### Community 7 - "Label Renaming"
Cohesion: 0.53
Nodes (5): main(), parse_args(), Namespace, Path, rename_labels()

### Community 8 - "Analysis Entry & Tree Printing"
Cohesion: 0.50
Nodes (4): main(), print_tree(), Recursively prints the directory structure with file counts., Main function to run comprehensive dataset analysis and save output to text file

### Community 9 - "LaTeX Summary Export"
Cohesion: 0.67
Nodes (3): class_name(), print_dataset_summary(), Print comprehensive dataset summary for LaTeX table creation.

## Knowledge Gaps
- **6 isolated node(s):** `Intensity Modality (grayscale infrared images)`, `Range Modality (LiDAR depth images)`, `fence_pole (class 0)`, `gantry_sign_pole (class 1)`, `traffic_pole (class 4)` (+1 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `Analysis Entry & Tree Printing` to `LaTeX Summary Export`, `Dataset Instance/Size Analysis`, `Bounding-Box Statistics`, `Annotation Counting`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._
- **What connects `Load class names from data.yaml.`, `Counts files with a given extension in a directory and its subdirectories.`, `Analyze file sizes of all images in the dataset.` to the rest of the system?**
  _18 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Dataset Config & Pole Taxonomy` be split into smaller, more focused modules?**
  _Cohesion score 0.14035087719298245 - nodes in this community are weakly interconnected._