import math
import json
import cv2
import numpy as np
from pycocotools import mask as maskUtils
import logging

import datetime


# -----------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------


def encode_mask(binary_mask):
    """Encode binary mask using COCO RLE."""
    rle = maskUtils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def get_bbox_from_mask(mask):
    """Compute bounding box [x,y,w,h] from binary mask."""
    ys, xs = np.where(mask)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return [int(x_min), int(y_min), math.ceil(x_max - x_min), math.ceil(y_max - y_min)]


# -----------------------------------------
# MAIN COCO GENERATION
# -----------------------------------------


def generate_coco_annotation(image_id, mask, classification, id):

    category_id = classification
    # Encode segmentation
    rle = encode_mask(mask)

    # Bounding box
    bbox = get_bbox_from_mask(mask)
    area = int(maskUtils.area(rle))

    w, h = bbox[2], bbox[3]

    if w < 3 or h < 3 or (w * h) < 30:
        return ()

    return (
        {
            "id": int(id),
            "image_id": int(image_id),
            "category_id": int(category_id),
            "segmentation": rle,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0,
        },
    )


def generate_coco(classification_image, annotation_id=0, image_id=0, id_image=None):
    # classification_image=classification_image.T

    # classification_image = np.array(255.0*classification_image/
    # np.max(classification_image)),dtype=np.uint8)
    annotations = []
    if id_image is not None:
        ids = np.unique(id_image)
        for i in ids:
            if i == 0:
                continue
            mask = id_image == i
            c = np.max(classification_image[mask])

            annotation = generate_coco_annotation(image_id, mask, c - 1, annotation_id)
            if annotation:
                annotation_id += 1
                annotations += annotation
    else:
        classes = np.unique(classification_image)
        for c in classes:
            if c == 0 or c == 1:
                continue
            binary_image = np.array(classification_image == c, dtype=np.uint8) * 255
            binary_image = cv2.dilate(binary_image, (3, 3))
            binary_image = cv2.erode(binary_image, (3, 3))

            binary_image = cv2.dilate(binary_image, (5, 5))

            (
                totalLabels,
                label_ids,
            ) = cv2.connectedComponentsWithAlgorithm(
                binary_image, connectivity=8, ltype=cv2.CV_32S, ccltype=cv2.CCL_DEFAULT
            )
            for i in range(1, totalLabels):
                mask = label_ids == i

                annotation = generate_coco_annotation(
                    image_id, mask, c - 1, annotation_id
                )
                if annotation:
                    annotation_id += 1
                    annotations += annotation

    (
        height,
        width,
    ) = classification_image.shape[:2]

    return annotations


def write_coco_json(
    filename,
    images,
    annotations,
    categories,
    info={
        "year": 2026,
        "version": "0.0.1",
        "description": "automatically created",
        "contributor": "",
        "url": "",
        "date_created": str(datetime.datetime.now()),
    },
):
    coco = {
        "info": info,
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    annotation_ids = set()
    for annotation in annotations:
        if annotation["id"] in annotation_ids:
            logging.debug(("DUPLICATE ANNOTATION????"))
        annotation_ids.add(annotation["id"])

    image_ids = set()
    for image in images:
        if image["id"] in image_ids:
            logging.debug(("DUPLICATE IMAGE????"))
        image_ids.add(image["id"])

    with open(filename, "w") as file:
        json.dump(coco, file, indent=2)
