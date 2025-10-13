#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

"""
Dataset utilities for COCO format manipulation.

This module provides helper functions for:
- Image validation
- Annotation validation and filtering
- ID remapping
- Statistics calculation
- COCO JSON manipulation
"""

import json
import os
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from collections import defaultdict


def validate_image(img_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that an image can be read and is not corrupted.

    Args:
        img_path: Path to image file

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not os.path.exists(img_path):
        return False, f"File not found: {img_path}"

    try:
        img = cv2.imread(img_path)
        if img is None:
            return False, "Failed to read image (corrupted or unsupported format)"

        if img.size == 0:
            return False, "Image has zero size"

        height, width = img.shape[:2]
        if height < 1 or width < 1:
            return False, f"Invalid dimensions: {height}x{width}"

        return True, None

    except Exception as e:
        return False, f"Exception during validation: {str(e)}"


def validate_bbox(
    bbox: List[float],
    img_width: int,
    img_height: int,
    min_area: float = 1.0
) -> Tuple[bool, Optional[str]]:
    """
    Validate a COCO bbox [x, y, width, height].

    Args:
        bbox: COCO format bbox [x, y, w, h]
        img_width: Image width
        img_height: Image height
        min_area: Minimum bbox area

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(bbox) != 4:
        return False, f"Invalid bbox format: {bbox}"

    x, y, w, h = bbox

    # Check for negative or zero dimensions
    if w <= 0 or h <= 0:
        return False, f"Invalid bbox dimensions: w={w}, h={h}"

    # Check for negative coordinates
    if x < 0 or y < 0:
        return False, f"Negative coordinates: x={x}, y={y}"

    # Check if bbox is completely outside image
    if x >= img_width or y >= img_height:
        return False, f"Bbox outside image: x={x}, y={y}, img={img_width}x{img_height}"

    # Check area
    area = w * h
    if area < min_area:
        return False, f"Bbox area {area} < minimum {min_area}"

    return True, None


def clip_bbox(
    bbox: List[float],
    img_width: int,
    img_height: int
) -> List[float]:
    """
    Clip a COCO bbox to image boundaries.

    Args:
        bbox: COCO format bbox [x, y, w, h]
        img_width: Image width
        img_height: Image height

    Returns:
        Clipped bbox [x, y, w, h]
    """
    x, y, w, h = bbox

    # Clip coordinates
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_width, x + w)
    y2 = min(img_height, y + h)

    # Calculate new width and height
    new_w = max(0, x2 - x1)
    new_h = max(0, y2 - y1)

    return [x1, y1, new_w, new_h]


def scale_bbox(bbox: List[float], scale_x: float, scale_y: float) -> List[float]:
    """
    Scale a COCO bbox by given factors.

    Args:
        bbox: COCO format bbox [x, y, w, h]
        scale_x: Horizontal scale factor
        scale_y: Vertical scale factor

    Returns:
        Scaled bbox [x, y, w, h]
    """
    x, y, w, h = bbox
    return [x * scale_x, y * scale_y, w * scale_x, h * scale_y]


def letterbox_image(
    img: np.ndarray,
    target_size: Tuple[int, int],
    fill_value: int = 114
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image with letterbox (maintain aspect ratio, pad with fill_value).

    This matches YOLOX's preprocessing strategy.

    Args:
        img: Input image (H, W, C)
        target_size: Target size (height, width)
        fill_value: Value to fill padding areas (default: 114, YOLOX standard)

    Returns:
        Tuple of (letterboxed_image, scale_ratio, (pad_width, pad_height))
    """
    target_h, target_w = target_size
    img_h, img_w = img.shape[:2]

    # Calculate scale ratio (same as YOLOX)
    scale = min(target_h / img_h, target_w / img_w)

    # Calculate new size
    new_h = int(img_h * scale)
    new_w = int(img_w * scale)

    # Resize image
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create padded image
    padded_img = np.full((target_h, target_w, 3), fill_value, dtype=np.uint8)

    # Calculate padding offsets (center the image)
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2

    # Place resized image in center
    padded_img[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized_img

    return padded_img, scale, (pad_left, pad_top)


def adjust_bbox_for_letterbox(
    bbox: List[float],
    scale: float,
    pad_left: int,
    pad_top: int
) -> List[float]:
    """
    Adjust bbox coordinates for letterbox transformation.

    Args:
        bbox: Original COCO bbox [x, y, w, h]
        scale: Scale ratio from letterbox
        pad_left: Left padding pixels
        pad_top: Top padding pixels

    Returns:
        Adjusted bbox [x, y, w, h]
    """
    x, y, w, h = bbox

    # Scale bbox
    x_scaled = x * scale
    y_scaled = y * scale
    w_scaled = w * scale
    h_scaled = h * scale

    # Add padding offset
    x_adjusted = x_scaled + pad_left
    y_adjusted = y_scaled + pad_top

    return [x_adjusted, y_adjusted, w_scaled, h_scaled]


def remap_ids(
    dataset: Dict,
    img_id_offset: int = 0,
    ann_id_offset: int = 0,
    cat_id_mapping: Optional[Dict[int, int]] = None
) -> Dict:
    """
    Remap image IDs, annotation IDs, and category IDs in a COCO dataset.

    Args:
        dataset: COCO format dataset dict
        img_id_offset: Offset to add to all image IDs
        ann_id_offset: Offset to add to all annotation IDs
        cat_id_mapping: Optional mapping from old category IDs to new ones

    Returns:
        New dataset with remapped IDs
    """
    import copy
    new_dataset = copy.deepcopy(dataset)

    # Create image ID mapping
    img_id_map = {}
    for img in new_dataset.get("images", []):
        old_id = img["id"]
        new_id = old_id + img_id_offset
        img_id_map[old_id] = new_id
        img["id"] = new_id

    # Remap annotations
    for ann in new_dataset.get("annotations", []):
        # Remap annotation ID
        ann["id"] = ann["id"] + ann_id_offset

        # Remap image ID
        if ann["image_id"] in img_id_map:
            ann["image_id"] = img_id_map[ann["image_id"]]

        # Remap category ID if mapping provided
        if cat_id_mapping is not None and ann["category_id"] in cat_id_mapping:
            ann["category_id"] = cat_id_mapping[ann["category_id"]]

    # Remap category IDs in categories
    if cat_id_mapping is not None:
        for cat in new_dataset.get("categories", []):
            if cat["id"] in cat_id_mapping:
                cat["id"] = cat_id_mapping[cat["id"]]

    return new_dataset


def calculate_dataset_statistics(dataset: Dict) -> Dict:
    """
    Calculate statistics about a COCO dataset.

    Args:
        dataset: COCO format dataset dict

    Returns:
        Dictionary with statistics
    """
    stats = {
        "num_images": len(dataset.get("images", [])),
        "num_annotations": len(dataset.get("annotations", [])),
        "num_categories": len(dataset.get("categories", [])),
        "annotations_per_image": {},
        "category_distribution": defaultdict(int),
        "bbox_areas": [],
        "image_sizes": [],
    }

    # Count annotations per image
    img_ann_count = defaultdict(int)
    for ann in dataset.get("annotations", []):
        img_id = ann["image_id"]
        img_ann_count[img_id] += 1
        stats["category_distribution"][ann["category_id"]] += 1
        stats["bbox_areas"].append(ann.get("area", 0))

    # Calculate image-level stats
    for img in dataset.get("images", []):
        stats["image_sizes"].append((img["height"], img["width"]))

    stats["annotations_per_image"] = {
        "min": min(img_ann_count.values()) if img_ann_count else 0,
        "max": max(img_ann_count.values()) if img_ann_count else 0,
        "mean": sum(img_ann_count.values()) / len(img_ann_count) if img_ann_count else 0,
        "images_with_zero_annotations": stats["num_images"] - len(img_ann_count),
    }

    # Calculate bbox area stats
    if stats["bbox_areas"]:
        stats["bbox_area_stats"] = {
            "min": min(stats["bbox_areas"]),
            "max": max(stats["bbox_areas"]),
            "mean": np.mean(stats["bbox_areas"]),
            "median": np.median(stats["bbox_areas"]),
        }

    # Convert defaultdict to regular dict
    stats["category_distribution"] = dict(stats["category_distribution"])

    return stats


def load_coco_json(json_path: str) -> Dict:
    """
    Load a COCO format JSON file.

    Args:
        json_path: Path to JSON file

    Returns:
        COCO dataset dict
    """
    with open(json_path, "r") as f:
        return json.load(f)


def save_coco_json(dataset: Dict, json_path: str) -> None:
    """
    Save a COCO format dataset to JSON file.

    Args:
        dataset: COCO dataset dict
        json_path: Output path for JSON file
    """
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(dataset, f, indent=2)


def create_empty_coco_dataset(
    categories: List[Dict],
    info: Optional[Dict] = None,
    licenses: Optional[List[Dict]] = None
) -> Dict:
    """
    Create an empty COCO format dataset.

    Args:
        categories: List of category dicts with 'id', 'name', 'supercategory'
        info: Optional dataset info dict
        licenses: Optional list of license dicts

    Returns:
        Empty COCO dataset dict
    """
    dataset = {
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    if info is not None:
        dataset["info"] = info

    if licenses is not None:
        dataset["licenses"] = licenses

    return dataset


def filter_annotations(
    dataset: Dict,
    min_area: float = 1.0,
    max_aspect_ratio: float = 50.0
) -> Dict:
    """
    Filter out invalid annotations from a COCO dataset.

    Args:
        dataset: COCO dataset dict
        min_area: Minimum bbox area
        max_aspect_ratio: Maximum width/height or height/width ratio

    Returns:
        Filtered dataset
    """
    import copy
    filtered_dataset = copy.deepcopy(dataset)

    valid_annotations = []
    for ann in filtered_dataset.get("annotations", []):
        bbox = ann.get("bbox", [])
        if len(bbox) != 4:
            continue

        x, y, w, h = bbox

        # Check area
        area = w * h
        if area < min_area:
            continue

        # Check aspect ratio
        if w > 0 and h > 0:
            aspect_ratio = max(w / h, h / w)
            if aspect_ratio > max_aspect_ratio:
                continue

        valid_annotations.append(ann)

    filtered_dataset["annotations"] = valid_annotations
    return filtered_dataset
