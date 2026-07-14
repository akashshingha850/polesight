# Graph Report - .  (2026-07-14)

## Corpus Check
- Large corpus: 1064 files · ~2,000,633 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 81 nodes · 145 edges · 14 communities (11 shown, 3 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 2 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Split Builder & Roboflow Download|Split Builder & Roboflow Download]]
- [[_COMMUNITY_Figure Generation|Figure Generation]]
- [[_COMMUNITY_Bounding-Box Statistics|Bounding-Box Statistics]]
- [[_COMMUNITY_Split Coverage Analysis|Split Coverage Analysis]]
- [[_COMMUNITY_Label Sanity Checker|Label Sanity Checker]]
- [[_COMMUNITY_Class-Name Loading (checker)|Class-Name Loading (checker)]]
- [[_COMMUNITY_Report Writing & Stem Parsing|Report Writing & Stem Parsing]]
- [[_COMMUNITY_Dataset Analysis Entry & IO|Dataset Analysis Entry & I/O]]
- [[_COMMUNITY_Image Size & Sample Selection|Image Size & Sample Selection]]
- [[_COMMUNITY_Roboflow Config & Credentials|Roboflow Config & Credentials]]
- [[_COMMUNITY_Roboflow Config Loading|Roboflow Config Loading]]
- [[_COMMUNITY_Multimodal Dataset Concepts|Multimodal Dataset Concepts]]
- [[_COMMUNITY_.env Parsing|.env Parsing]]
- [[_COMMUNITY_data.yaml Class Loading|data.yaml Class Loading]]

## God Nodes (most connected - your core abstractions)
1. `main()` - 11 edges
2. `build_report()` - 10 edges
3. `generate_figures()` - 10 edges
4. `main()` - 8 edges
5. `add_sample_montage()` - 8 edges
6. `download_from_roboflow()` - 7 edges
7. `load_env()` - 6 edges
8. `load_class_names()` - 6 edges
9. `download_roboflow_export()` - 6 edges
10. `write_data_yaml()` - 6 edges

## Surprising Connections (you probably didn't know these)
- `build_roboflow_label_index()` --calls--> `roboflow_stem()`  [EXTRACTED]
  data_split.py → check_labels.py
- `download_roboflow_export()` --calls--> `load_env()`  [EXTRACTED]
  data_split.py → check_labels.py
- `write_data_yaml()` --calls--> `load_roboflow_config()`  [EXTRACTED]
  data_split.py → check_labels.py
- `download_roboflow_export()` --calls--> `download_from_roboflow()`  [EXTRACTED]
  data_split.py → check_labels.py
- `write_data_yaml()` --calls--> `load_class_names()`  [EXTRACTED]
  data_split.py → check_labels.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Multimodal Sensor Inputs (intensity/range/RGB)** — readme_intensity_modality, readme_range_modality, readme_rgb_modality [EXTRACTED 1.00]

## Communities (14 total, 3 thin omitted)

### Community 0 - "Split Builder & Roboflow Download"
Cohesion: 0.22
Nodes (17): build_image_index(), build_label_index(), build_roboflow_label_index(), copy_sample(), download_roboflow_export(), main(), parse_args(), Namespace (+9 more)

### Community 1 - "Figure Generation"
Cohesion: 0.32
Nodes (8): add_sample_montage(), _apply_style(), class_color(), generate_figures(), Save a figure as PNG plus a companion JSON of the plotted data., Render all figures + companion JSON into data/.figure/., Qualitative Figure 1: real annotated sample frames with drawn boxes., _save()

### Community 2 - "Bounding-Box Statistics"
Cohesion: 0.29
Nodes (7): analyze_bounding_boxes(), build_report(), calc_stats(), class_name(), Analyze bounding-box statistics from YOLO-format labels for one split.      Also, Basic descriptive statistics for a list of values., Run the full analysis and return (report_dict, raw_arrays).

### Community 3 - "Split Coverage Analysis"
Cohesion: 0.33
Nodes (6): analyze_set(), count_annotated_images(), count_files(), Analyze a specific dataset split (images/labels/annotation coverage)., Counts files with a given extension in a directory and its subdirectories., Count images that have a corresponding non-empty annotation file.

### Community 5 - "Class-Name Loading (checker)"
Cohesion: 0.40
Nodes (5): load_class_names(), main(), parse_args(), Namespace, Best-effort parse of the `names:` list from <root>/data.yaml.      Handles both

### Community 6 - "Report Writing & Stem Parsing"
Cohesion: 0.40
Nodes (5): Path, Write the collected report lines to a .txt file (defaults to the data folder)., Numeric stem for a Roboflow label file (strips `_png.rf.<hash>`)., roboflow_stem(), write_report()

### Community 7 - "Dataset Analysis Entry & I/O"
Cohesion: 0.50
Nodes (4): _load_intensity_display(), main(), Load an image (incl. 16-bit ``I;16`` intensity strips) as a 0-1 float     array,, write_json_report()

### Community 8 - "Image Size & Sample Selection"
Cohesion: 0.40
Nodes (5): analyze_image_file_sizes(), Analyze file sizes (KB) of all images in the dataset., Pick sample images that together cover as many classes as possible,     preferri, resolve_split_dir(), _select_sample_images()

### Community 9 - "Roboflow Config & Credentials"
Cohesion: 0.60
Nodes (5): ROBOFLOW_API key in .env, data/ directory (regenerable dataset), Roboflow Dataset Coordinates, Roboflow Project polesight-7kgwj, Roboflow Workspace polesight

### Community 10 - "Roboflow Config Loading"
Cohesion: 0.50
Nodes (4): download_from_roboflow(), load_roboflow_config(), Download the dataset named in <repo>/roboflow.yaml to `dest`, return its path., Parse the flat `key: value` coordinates from <repo>/roboflow.yaml.

### Community 11 - "Multimodal Dataset Concepts"
Cohesion: 0.50
Nodes (4): Intensity Modality (grayscale infrared images), Polesight Dataset (multimodal pole-detection dataset), Range Modality (LiDAR depth images), RGB Modality (color camera images)

## Knowledge Gaps
- **3 isolated node(s):** `Intensity Modality (grayscale infrared images)`, `Range Modality (LiDAR depth images)`, `RGB Modality (color camera images)`
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `download_from_roboflow()` connect `Roboflow Config Loading` to `Split Builder & Roboflow Download`, `Label Sanity Checker`, `Class-Name Loading (checker)`, `Report Writing & Stem Parsing`?**
  _High betweenness centrality (0.019) - this node is a cross-community bridge._
- **Why does `build_report()` connect `Bounding-Box Statistics` to `Image Size & Sample Selection`, `Figure Generation`, `Split Coverage Analysis`, `Dataset Analysis Entry & I/O`?**
  _High betweenness centrality (0.018) - this node is a cross-community bridge._
- **What connects `Numeric stem for a Roboflow label file (strips `_png.rf.<hash>`).`, `Minimal KEY=VALUE parser so we avoid a python-dotenv dependency.`, `Parse the flat `key: value` coordinates from <repo>/roboflow.yaml.` to the rest of the system?**
  _26 weakly-connected nodes found - possible documentation gaps or missing edges._