from pathlib import Path
import logging


def find_matches(coco, dataset_root):

    # Root directory containing the dataset
    dataset_root = Path(dataset_root)

    # ------------------------------------------------------------------
    # Find all pose files under directories named "pose"
    # ------------------------------------------------------------------
    pose_files = {}

    for pose_dir in dataset_root.rglob("pose"):
        if pose_dir.is_dir():
            for txt_file in pose_dir.glob("*.txt"):
                pose_files[txt_file.stem] = txt_file.resolve()

    # ------------------------------------------------------------------
    # Build mapping
    # ------------------------------------------------------------------
    mapping = {}
    for id in coco.getImgIds():
        image = coco.loadImgs(id)[0]
        original_name = image.get("extra", {}).get("name")
        if original_name:
            stem = Path(original_name).stem
        else:
            stem = image["file_name"].split("_png.rf.")[0]

        if stem in pose_files:
            mapping[image["id"]] = str(pose_files[stem])

    logging.debug((f"Created {len(mapping)} mappings"))
    return mapping
