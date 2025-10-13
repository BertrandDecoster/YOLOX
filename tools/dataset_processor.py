#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

"""
COCO dataset processor.

This module processes COCO datasets:
- Splits into train/val/test with anti-leakage grouping
- Resizes images (letterbox)
- Adjusts annotations for resized images
- Saves processed images and annotations
"""

import os
import shutil
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import cv2
import numpy as np
from tqdm import tqdm

from .dataset_utils import (
    load_coco_json,
    save_coco_json,
    letterbox_image,
    adjust_bbox_for_letterbox,
    validate_image,
    validate_bbox,
    clip_bbox,
    create_empty_coco_dataset,
    calculate_dataset_statistics,
)
from .dataset_splitter import split_dataset_by_groups, get_split_statistics


class DatasetProcessor:
    """
    Process COCO dataset with splitting, resizing, and annotation adjustment.
    """

    def __init__(
        self,
        input_json: str,
        input_img_dir: Optional[str] = None,
        output_dir: str = "",
        target_size: Tuple[int, int] = (640, 640),
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        separator: str = "_jpg.rf",
        seed: int = 42,
        use_weighted_split: bool = True,
        copy_images: bool = True,
        validate_images: bool = True,
        max_images: Optional[int] = None,
    ):
        """
        Initialize dataset processor.

        Args:
            input_json: Path to input COCO JSON
            input_img_dir: Directory containing input images (optional if JSON contains dataset_img_dirs)
            output_dir: Output directory for processed dataset
            target_size: Target image size (height, width)
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            separator: Separator for grouping files (anti-leakage)
            seed: Random seed
            use_weighted_split: Use weighted splitting algorithm
            copy_images: If True, copy images. If False, create symlinks.
            validate_images: Validate images before processing
            max_images: Maximum number of images to process (randomly sampled). None = process all.
        """
        self.input_json = input_json
        self.input_img_dir = input_img_dir
        self.output_dir = output_dir
        self.target_size = target_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.separator = separator
        self.seed = seed
        self.use_weighted_split = use_weighted_split
        self.copy_images = copy_images
        self.validate_images = validate_images
        self.max_images = max_images
        self.dataset_img_dirs = {}  # Will be populated from JSON if available

        # Create output directories
        self.splits = ["train2017", "val2017", "test2017"]
        self.output_img_dirs = {}
        self.output_json_paths = {}

        for split in self.splits:
            img_dir = os.path.join(output_dir, split)
            os.makedirs(img_dir, exist_ok=True)
            self.output_img_dirs[split] = img_dir

        # Create annotations directory
        ann_dir = os.path.join(output_dir, "annotations")
        os.makedirs(ann_dir, exist_ok=True)

        for split in self.splits:
            json_name = f"instances_{split}.json"
            self.output_json_paths[split] = os.path.join(ann_dir, json_name)

    def process(self) -> Dict:
        """
        Run the full processing pipeline.

        Returns:
            Statistics dict
        """
        print("=" * 80)
        print("DATASET PROCESSING PIPELINE")
        print("=" * 80)
        print(f"Input JSON: {self.input_json}")
        print(f"Input images: {self.input_img_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Target size: {self.target_size}")
        print(f"Split ratios: {self.train_ratio}/{self.val_ratio}/{self.test_ratio}")
        print()

        # Load dataset
        print("Loading dataset...")
        dataset = load_coco_json(self.input_json)
        print(f"Loaded {len(dataset['images'])} images, "
              f"{len(dataset['annotations'])} annotations")

        # Filter out unused categories
        print("Filtering categories...")
        dataset = self._filter_unused_categories(dataset)

        # Limit dataset size if requested
        if self.max_images is not None and len(dataset['images']) > self.max_images:
            print(f"Limiting dataset to {self.max_images} images (randomly sampled)...")
            dataset = self._sample_dataset(dataset, self.max_images)
            print(f"Sampled {len(dataset['images'])} images, "
                  f"{len(dataset['annotations'])} annotations")

        # Load dataset image directories if available (from merger)
        if "dataset_img_dirs" in dataset:
            self.dataset_img_dirs = dataset["dataset_img_dirs"]
            print(f"Found {len(self.dataset_img_dirs)} source dataset(s)")
        elif self.input_img_dir:
            # Single dataset, use provided directory
            self.dataset_img_dirs = {"default": self.input_img_dir}
        else:
            # Try to infer from JSON location
            self.dataset_img_dirs = {"default": os.path.dirname(self.input_json)}

        print(f"Image base directories: {list(self.dataset_img_dirs.values())}")
        print()

        # Split dataset
        print("Splitting dataset...")
        file_to_split = self._split_dataset(dataset)
        print()

        # Process each split
        stats = {}
        for split_name in ["train", "val", "test"]:
            split_key = f"{split_name}2017"
            print(f"Processing {split_name} split...")

            split_stats = self._process_split(
                dataset,
                file_to_split,
                split_name,
                split_key,
            )
            stats[split_name] = split_stats
            print()

        # Save overall statistics
        stats_path = os.path.join(self.output_dir, "annotations", "dataset_stats.json")
        import json
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved statistics to: {stats_path}")
        print()

        print("=" * 80)
        print("PROCESSING COMPLETE")
        print("=" * 80)
        for split_name in ["train", "val", "test"]:
            s = stats[split_name]
            print(f"{split_name.upper()}:")
            print(f"  Images: {s['num_images']}")
            print(f"  Annotations: {s['num_annotations']}")
            print(f"  Output: {self.output_img_dirs[f'{split_name}2017']}")

        return stats

    def _split_dataset(self, dataset: Dict) -> Dict[str, str]:
        """
        Split dataset into train/val/test.

        Args:
            dataset: COCO dataset

        Returns:
            Mapping from filename to split name ('train', 'val', or 'test')
        """
        # Get all filenames
        filenames = [img["file_name"] for img in dataset["images"]]

        # Split using weighted algorithm
        train_files, val_files, test_files = split_dataset_by_groups(
            filenames,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            separator=self.separator,
            seed=self.seed,
            use_weighted=self.use_weighted_split,
        )

        # Get statistics
        split_stats = get_split_statistics(
            train_files, val_files, test_files, self.separator
        )

        print(f"Split statistics:")
        print(f"  Train: {split_stats['train']['files']} files "
              f"({split_stats['train']['ratio']:.1%}) "
              f"in {split_stats['train']['groups']} groups")
        print(f"  Val:   {split_stats['val']['files']} files "
              f"({split_stats['val']['ratio']:.1%}) "
              f"in {split_stats['val']['groups']} groups")
        print(f"  Test:  {split_stats['test']['files']} files "
              f"({split_stats['test']['ratio']:.1%}) "
              f"in {split_stats['test']['groups']} groups")

        # Create filename to split mapping
        file_to_split = {}
        for f in train_files:
            file_to_split[f] = "train"
        for f in val_files:
            file_to_split[f] = "val"
        for f in test_files:
            file_to_split[f] = "test"

        return file_to_split

    def _sample_dataset(self, dataset: Dict, max_images: int) -> Dict:
        """
        Randomly sample a subset of images from the dataset.

        Args:
            dataset: COCO dataset
            max_images: Maximum number of images to keep

        Returns:
            Sampled COCO dataset
        """
        import random
        random.seed(self.seed)

        # Get all image IDs
        all_images = dataset["images"]
        if len(all_images) <= max_images:
            return dataset

        # Randomly sample images
        sampled_images = random.sample(all_images, max_images)
        sampled_img_ids = {img["id"] for img in sampled_images}

        # Filter annotations to only include those for sampled images
        sampled_annotations = [
            ann for ann in dataset["annotations"]
            if ann["image_id"] in sampled_img_ids
        ]

        # Create new dataset with sampled data
        sampled_dataset = {
            "images": sampled_images,
            "annotations": sampled_annotations,
            "categories": dataset.get("categories", []),
        }

        # Preserve metadata
        if "info" in dataset:
            sampled_dataset["info"] = dataset["info"]
        if "licenses" in dataset:
            sampled_dataset["licenses"] = dataset["licenses"]
        if "dataset_img_dirs" in dataset:
            sampled_dataset["dataset_img_dirs"] = dataset["dataset_img_dirs"]

        return sampled_dataset

    def _filter_unused_categories(self, dataset: Dict) -> Dict:
        """
        Filter out categories that have no annotations.

        Args:
            dataset: COCO dataset

        Returns:
            Dataset with only used categories
        """
        # Find which categories are actually used
        used_cat_ids = set()
        for ann in dataset.get("annotations", []):
            used_cat_ids.add(ann.get("category_id"))

        # Filter categories
        original_categories = dataset.get("categories", [])
        used_categories = [cat for cat in original_categories if cat["id"] in used_cat_ids]

        if len(used_categories) < len(original_categories):
            unused_cats = [cat for cat in original_categories if cat["id"] not in used_cat_ids]
            print(f"  Filtering out {len(unused_cats)} unused categories:")
            for cat in unused_cats:
                print(f"    - Category {cat['id']}: '{cat['name']}' (0 annotations)")

        # Update dataset
        dataset["categories"] = used_categories
        print(f"  Kept {len(used_categories)} used categories")

        return dataset

    def _process_split(
        self,
        dataset: Dict,
        file_to_split: Dict[str, str],
        split_name: str,
        split_key: str,
    ) -> Dict:
        """
        Process a single split.

        Args:
            dataset: Full COCO dataset
            file_to_split: Mapping from filename to split name
            split_name: Name of split ('train', 'val', 'test')
            split_key: Key for split ('train2017', 'val2017', 'test2017')

        Returns:
            Statistics dict
        """
        # Create empty dataset for this split
        split_dataset = create_empty_coco_dataset(
            categories=dataset.get("categories", []),
            info=dataset.get("info"),
            licenses=dataset.get("licenses"),
        )

        # Filter images for this split
        split_images = [
            img for img in dataset["images"]
            if file_to_split.get(img["file_name"]) == split_name
        ]

        # Create image ID mapping
        old_to_new_img_id = {}
        for new_id, img in enumerate(split_images):
            old_to_new_img_id[img["id"]] = new_id

        # Process images
        stats = {
            "num_images": 0,
            "num_annotations": 0,
            "images_skipped": 0,
            "annotations_skipped": 0,
        }

        print(f"  Processing {len(split_images)} images...")

        for img in tqdm(split_images, desc=f"  {split_name}"):
            # Get correct image path based on source dataset
            input_path = self._get_image_path(img)

            if self.validate_images:
                is_valid, error = validate_image(input_path)
                if not is_valid:
                    print(f"  Skipping invalid image {img['file_name']}: {error}")
                    stats["images_skipped"] += 1
                    continue

            # Process image
            processed_img, new_annotations = self._process_single_image(
                img, input_path, old_to_new_img_id[img["id"]], split_key, dataset
            )

            if processed_img is not None:
                split_dataset["images"].append(processed_img)
                split_dataset["annotations"].extend(new_annotations)
                stats["num_images"] += 1
                stats["num_annotations"] += len(new_annotations)

        # Save split
        save_coco_json(split_dataset, self.output_json_paths[split_key])

        print(f"  Saved {stats['num_images']} images, {stats['num_annotations']} annotations")
        print(f"  JSON: {self.output_json_paths[split_key]}")

        return stats

    def _get_image_path(self, img: Dict) -> str:
        """
        Get full path to an image, handling multiple source datasets.

        Args:
            img: Image dict from COCO

        Returns:
            Full path to image file
        """
        filename = img["file_name"]
        dataset_source = img.get("dataset_source", "default")

        # Get base directory for this dataset
        base_dir = self.dataset_img_dirs.get(dataset_source)
        if base_dir is None:
            # Fallback to default or first available
            base_dir = self.dataset_img_dirs.get("default") or list(self.dataset_img_dirs.values())[0]

        return os.path.join(base_dir, filename)

    def _process_single_image(
        self,
        img: Dict,
        input_path: str,
        new_img_id: int,
        split_key: str,
        full_dataset: Dict,
    ) -> Tuple[Optional[Dict], List[Dict]]:
        """
        Process a single image and its annotations.

        Args:
            img: Image dict from COCO
            input_path: Path to input image
            new_img_id: New image ID for output
            split_key: Split key (train2017, val2017, test2017)
            full_dataset: Full COCO dataset (for getting annotations)

        Returns:
            Tuple of (processed_image_dict, list_of_processed_annotations)
        """
        # Read image
        cv_img = cv2.imread(input_path)
        if cv_img is None:
            return None, []

        orig_h, orig_w = cv_img.shape[:2]

        # Letterbox resize
        resized_img, scale, (pad_left, pad_top) = letterbox_image(
            cv_img, self.target_size
        )

        # Save resized image
        output_filename = img["file_name"]
        output_path = os.path.join(self.output_img_dirs[split_key], output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, resized_img)

        # Create new image dict
        new_img = {
            "id": new_img_id,
            "file_name": output_filename,
            "height": self.target_size[0],
            "width": self.target_size[1],
            "original_height": orig_h,
            "original_width": orig_w,
            "scale": scale,
            "pad_left": pad_left,
            "pad_top": pad_top,
        }

        # Get and process annotations
        annotations = [
            ann for ann in full_dataset["annotations"]
            if ann["image_id"] == img["id"]
        ]

        new_annotations = []
        for ann_idx, ann in enumerate(annotations):
            # Adjust bbox for letterbox
            bbox = ann["bbox"]
            new_bbox = adjust_bbox_for_letterbox(bbox, scale, pad_left, pad_top)

            # Clip to image boundaries
            new_bbox = clip_bbox(new_bbox, self.target_size[1], self.target_size[0])

            # Validate
            is_valid, _ = validate_bbox(
                new_bbox, self.target_size[1], self.target_size[0], min_area=1.0
            )

            if not is_valid or new_bbox[2] <= 0 or new_bbox[3] <= 0:
                continue

            # Create new annotation
            new_ann = {
                "id": len(new_annotations),  # Will be remapped globally later if needed
                "image_id": new_img_id,
                "category_id": ann["category_id"],
                "bbox": new_bbox,
                "area": new_bbox[2] * new_bbox[3],
                "iscrowd": ann.get("iscrowd", 0),
            }

            if "segmentation" in ann:
                # TODO: Adjust segmentation polygons if needed
                new_ann["segmentation"] = []

            new_annotations.append(new_ann)

        return new_img, new_annotations
