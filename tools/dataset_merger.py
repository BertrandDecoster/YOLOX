#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

"""
COCO dataset merger.

This module provides functionality to merge multiple COCO format datasets into one,
handling ID conflicts and category remapping.
"""

import os
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import copy

from .dataset_utils import (
    load_coco_json,
    save_coco_json,
    create_empty_coco_dataset,
    validate_bbox,
    clip_bbox,
)


class COCOMerger:
    """
    Merge multiple COCO datasets into a single dataset.

    Handles:
    - Image ID conflicts
    - Annotation ID conflicts
    - Category ID conflicts/merging
    - Path remapping
    """

    def __init__(
        self,
        merge_categories: bool = True,
        validate_annotations: bool = True,
        clip_bboxes: bool = True,
    ):
        """
        Initialize COCO merger.

        Args:
            merge_categories: If True, merge categories by name. If False, keep separate.
            validate_annotations: If True, validate and filter invalid annotations.
            clip_bboxes: If True, clip bboxes to image boundaries.
        """
        self.merge_categories = merge_categories
        self.validate_annotations = validate_annotations
        self.clip_bboxes = clip_bboxes

        self.next_img_id = 0
        self.next_ann_id = 0
        self.next_cat_id = 1

        self.category_map = {}  # Maps (dataset_name, old_cat_id) -> new_cat_id
        self.merged_categories = {}  # Maps category_name -> category_dict
        self.dataset_img_dirs = {}  # Maps dataset_name -> image base directory

    def merge_datasets(
        self,
        dataset_paths: List[Tuple[str, str, str]],
        output_json: Optional[str] = None,
    ) -> Dict:
        """
        Merge multiple COCO datasets.

        Args:
            dataset_paths: List of (dataset_name, json_path, img_dir) tuples
            output_json: Optional path to save merged dataset

        Returns:
            Merged COCO dataset dict
        """
        print(f"Merging {len(dataset_paths)} datasets...")

        # Initialize merged dataset
        merged = create_empty_coco_dataset(categories=[])
        merged["info"] = {
            "description": "Merged dataset",
            "version": "1.0",
            "year": 2024,
        }

        # Track statistics
        stats = {
            "total_images": 0,
            "total_annotations": 0,
            "datasets": {},
        }

        # Process each dataset
        for dataset_name, json_path, img_dir in dataset_paths:
            print(f"\nProcessing dataset: {dataset_name}")
            print(f"  Loading from: {json_path}")
            print(f"  Images base: {img_dir}")

            # Store image directory for this dataset
            self.dataset_img_dirs[dataset_name] = img_dir

            dataset = load_coco_json(json_path)
            dataset_stats = self._merge_single_dataset(
                merged, dataset, dataset_name, img_dir
            )

            stats["datasets"][dataset_name] = dataset_stats
            stats["total_images"] += dataset_stats["images"]
            stats["total_annotations"] += dataset_stats["annotations"]

            print(f"  Added {dataset_stats['images']} images, "
                  f"{dataset_stats['annotations']} annotations")

        # Finalize categories
        merged["categories"] = list(self.merged_categories.values())

        # Add dataset image directory mapping to merged dataset
        merged["dataset_img_dirs"] = self.dataset_img_dirs

        print(f"\nMerge complete:")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Total annotations: {stats['total_annotations']}")
        print(f"  Total categories: {len(merged['categories'])}")

        # Save if output path provided
        if output_json:
            save_coco_json(merged, output_json)
            print(f"\nSaved merged dataset to: {output_json}")

        return merged

    def _merge_single_dataset(
        self,
        merged: Dict,
        dataset: Dict,
        dataset_name: str,
        dataset_base_path: str,
    ) -> Dict:
        """
        Merge a single dataset into the merged dataset.

        Args:
            merged: The merged dataset being built
            dataset: The dataset to merge in
            dataset_name: Name identifier for this dataset
            dataset_base_path: Base path for resolving image paths

        Returns:
            Statistics dict for this dataset
        """
        stats = {"images": 0, "annotations": 0, "annotations_filtered": 0}

        # First, identify which categories are actually used in annotations
        used_cat_ids = set()
        for ann in dataset.get("annotations", []):
            used_cat_ids.add(ann.get("category_id"))

        # Filter categories to only include those that are used
        original_categories = dataset.get("categories", [])
        used_categories = [cat for cat in original_categories if cat["id"] in used_cat_ids]

        if len(used_categories) < len(original_categories):
            unused_cats = [cat for cat in original_categories if cat["id"] not in used_cat_ids]
            print(f"  Filtering out {len(unused_cats)} unused categories:")
            for cat in unused_cats:
                print(f"    - Category {cat['id']}: '{cat['name']}' (0 annotations)")

        # Process only the used categories
        old_to_new_cat_id = self._process_categories_filtered(
            used_categories, dataset_name
        )

        # Process images
        old_to_new_img_id = {}
        for img in dataset.get("images", []):
            old_img_id = img["id"]
            new_img_id = self.next_img_id
            self.next_img_id += 1

            old_to_new_img_id[old_img_id] = new_img_id

            # Create new image entry
            new_img = {
                "id": new_img_id,
                "file_name": img["file_name"],
                "height": img["height"],
                "width": img["width"],
                "dataset_source": dataset_name,  # Track source dataset
            }

            # Preserve other fields
            for key in ["date_captured", "license", "coco_url", "flickr_url"]:
                if key in img:
                    new_img[key] = img[key]

            merged["images"].append(new_img)
            stats["images"] += 1

        # Process annotations
        for ann in dataset.get("annotations", []):
            # Remap IDs
            new_ann_id = self.next_ann_id
            self.next_ann_id += 1

            old_img_id = ann["image_id"]
            if old_img_id not in old_to_new_img_id:
                print(f"Warning: Annotation {ann['id']} references "
                      f"non-existent image {old_img_id}, skipping")
                stats["annotations_filtered"] += 1
                continue

            new_img_id = old_to_new_img_id[old_img_id]

            old_cat_id = ann["category_id"]
            if old_cat_id not in old_to_new_cat_id:
                print(f"Warning: Annotation {ann['id']} references "
                      f"non-existent category {old_cat_id}, skipping")
                stats["annotations_filtered"] += 1
                continue

            new_cat_id = old_to_new_cat_id[old_cat_id]

            # Validate bbox
            bbox = ann.get("bbox", [])
            if len(bbox) != 4:
                print(f"Warning: Invalid bbox format in annotation {ann['id']}, skipping")
                stats["annotations_filtered"] += 1
                continue

            # Get corresponding image for validation
            img = next((img for img in merged["images"] if img["id"] == new_img_id), None)
            if img is None:
                stats["annotations_filtered"] += 1
                continue

            # Validate and clip bbox
            if self.validate_annotations:
                is_valid, error = validate_bbox(bbox, img["width"], img["height"])
                if not is_valid:
                    print(f"Warning: Invalid annotation {ann['id']}: {error}, skipping")
                    stats["annotations_filtered"] += 1
                    continue

            if self.clip_bboxes:
                bbox = clip_bbox(bbox, img["width"], img["height"])
                # Check if bbox still has area after clipping
                if bbox[2] <= 0 or bbox[3] <= 0:
                    stats["annotations_filtered"] += 1
                    continue

            # Create new annotation
            new_ann = {
                "id": new_ann_id,
                "image_id": new_img_id,
                "category_id": new_cat_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],  # width * height
                "iscrowd": ann.get("iscrowd", 0),
            }

            # Preserve segmentation if present
            if "segmentation" in ann:
                new_ann["segmentation"] = ann["segmentation"]

            merged["annotations"].append(new_ann)
            stats["annotations"] += 1

        return stats

    def _process_categories(
        self,
        dataset: Dict,
        dataset_name: str,
    ) -> Dict[int, int]:
        """
        Process categories and create mapping from old to new IDs.

        Args:
            dataset: Dataset being processed
            dataset_name: Name of the dataset

        Returns:
            Mapping from old category IDs to new category IDs
        """
        old_to_new = {}

        for cat in dataset.get("categories", []):
            old_cat_id = cat["id"]
            cat_name = cat["name"]

            if self.merge_categories:
                # Merge categories with same name
                if cat_name in self.merged_categories:
                    # Category already exists
                    new_cat_id = self.merged_categories[cat_name]["id"]
                else:
                    # New category
                    new_cat_id = self.next_cat_id
                    self.next_cat_id += 1

                    self.merged_categories[cat_name] = {
                        "id": new_cat_id,
                        "name": cat_name,
                        "supercategory": cat.get("supercategory", "object"),
                    }
            else:
                # Keep categories separate (add dataset prefix)
                prefixed_name = f"{dataset_name}_{cat_name}"

                if prefixed_name in self.merged_categories:
                    new_cat_id = self.merged_categories[prefixed_name]["id"]
                else:
                    new_cat_id = self.next_cat_id
                    self.next_cat_id += 1

                    self.merged_categories[prefixed_name] = {
                        "id": new_cat_id,
                        "name": prefixed_name,
                        "supercategory": cat.get("supercategory", "object"),
                    }

            old_to_new[old_cat_id] = new_cat_id
            self.category_map[(dataset_name, old_cat_id)] = new_cat_id

        return old_to_new

    def _process_categories_filtered(
        self,
        categories: List[Dict],
        dataset_name: str,
    ) -> Dict[int, int]:
        """
        Process a filtered list of categories and create mapping from old to new IDs.

        Args:
            categories: List of category dicts to process (already filtered)
            dataset_name: Name of the dataset

        Returns:
            Mapping from old category IDs to new category IDs
        """
        old_to_new = {}

        for cat in categories:
            old_cat_id = cat["id"]
            cat_name = cat["name"]

            if self.merge_categories:
                # Merge categories with same name
                if cat_name in self.merged_categories:
                    # Category already exists
                    new_cat_id = self.merged_categories[cat_name]["id"]
                else:
                    # New category
                    new_cat_id = self.next_cat_id
                    self.next_cat_id += 1

                    self.merged_categories[cat_name] = {
                        "id": new_cat_id,
                        "name": cat_name,
                        "supercategory": cat.get("supercategory", "object"),
                    }
            else:
                # Keep categories separate (add dataset prefix)
                prefixed_name = f"{dataset_name}_{cat_name}"

                if prefixed_name in self.merged_categories:
                    new_cat_id = self.merged_categories[prefixed_name]["id"]
                else:
                    new_cat_id = self.next_cat_id
                    self.next_cat_id += 1

                    self.merged_categories[prefixed_name] = {
                        "id": new_cat_id,
                        "name": prefixed_name,
                        "supercategory": cat.get("supercategory", "object"),
                    }

            old_to_new[old_cat_id] = new_cat_id
            self.category_map[(dataset_name, old_cat_id)] = new_cat_id

        return old_to_new


def merge_coco_datasets(
    dataset_configs: List[Dict],
    output_json: str,
    merge_categories: bool = True,
) -> Dict:
    """
    Convenience function to merge COCO datasets.

    Args:
        dataset_configs: List of dicts with 'name', 'json_path', and optionally 'img_dir' keys
        output_json: Path to save merged dataset
        merge_categories: Whether to merge categories by name

    Returns:
        Merged dataset dict

    Example:
        >>> configs = [
        ...     {"name": "dataset1", "json_path": "path/to/dataset1.json"},
        ...     {"name": "dataset2", "json_path": "path/to/dataset2.json", "img_dir": "custom/images"},
        ... ]
        >>> merged = merge_coco_datasets(configs, "merged.json")
    """
    dataset_paths = [
        (
            cfg["name"],
            cfg["json_path"],
            cfg.get("img_dir", os.path.dirname(cfg["json_path"]))
        )
        for cfg in dataset_configs
    ]

    merger = COCOMerger(merge_categories=merge_categories)
    merged = merger.merge_datasets(dataset_paths, output_json)

    return merged
