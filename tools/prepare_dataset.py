#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

"""
Complete dataset preparation pipeline.

This script orchestrates the full dataset preparation process:
1. Merge multiple COCO datasets from datasets_raw/
2. Split into train/val/test with anti-leakage grouping
3. Resize images and adjust annotations
4. Save to datasets/ in YOLOX-compatible format
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.dataset_merger import COCOMerger
from tools.dataset_processor import DatasetProcessor
from tools.dataset_utils import calculate_dataset_statistics, save_coco_json


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare YOLOX dataset from raw COCO annotations"
    )

    # Input/Output
    parser.add_argument(
        "--input-datasets",
        nargs="+",
        required=True,
        help="List of input dataset JSON files. Images are auto-inferred from JSON location. "
             "Format: path.json OR name:path.json OR name:path.json:img_dir (custom override)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory (will be created in datasets/)"
    )

    # Merging options
    parser.add_argument(
        "--merge-categories",
        action="store_true",
        default=True,
        help="Merge categories with same name across datasets (default: True)"
    )
    parser.add_argument(
        "--no-merge-categories",
        action="store_false",
        dest="merge_categories",
        help="Keep categories separate across datasets"
    )

    # Splitting options
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test set ratio (default: 0.15)"
    )
    parser.add_argument(
        "--separator",
        type=str,
        default="_jpg.rf",
        help="Separator for grouping files (anti-leakage) (default: '_jpg.rf')"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting (default: 42)"
    )
    parser.add_argument(
        "--use-uniform-split",
        action="store_true",
        help="Use uniform splitting instead of weighted (less accurate)"
    )

    # Processing options
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        default=[640, 640],
        help="Target image size [height width] (default: 640 640)"
    )
    parser.add_argument(
        "--skip-resize",
        action="store_true",
        help="Skip image resizing (copy original images)"
    )
    parser.add_argument(
        "--symlink-images",
        action="store_true",
        help="Create symlinks instead of copying images (faster but less portable)"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip image validation (faster but may include corrupted images)"
    )

    # Other
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without actually doing it"
    )

    return parser.parse_args()


def parse_dataset_input(dataset_str: str) -> tuple:
    """
    Parse dataset input string with optional image directory override.

    Args:
        dataset_str: String in one of these formats:
            - "path.json" (name and img_dir auto-inferred)
            - "name:path.json" (img_dir auto-inferred)
            - "name:path.json:img_dir" (custom img_dir)

    Returns:
        Tuple of (name, json_path, img_dir)
    """
    parts = dataset_str.split(":")

    if len(parts) == 1:
        # Just path: infer name and img_dir
        json_path = parts[0]
        name = os.path.splitext(os.path.basename(json_path))[0]
        img_dir = os.path.dirname(json_path)

    elif len(parts) == 2:
        # name:path: infer img_dir
        name, json_path = parts
        img_dir = os.path.dirname(json_path)

    elif len(parts) == 3:
        # name:path:img_dir: all specified
        name, json_path, img_dir = parts

    else:
        raise ValueError(f"Invalid dataset format: {dataset_str}")

    # If img_dir is empty (JSON at root), use current directory
    if not img_dir:
        img_dir = "."

    return name, json_path, img_dir


def main():
    """Main pipeline function."""
    args = parse_args()

    print("=" * 80)
    print("YOLOX DATASET PREPARATION PIPELINE")
    print("=" * 80)
    print()

    # Parse input datasets
    dataset_configs = []
    for dataset_str in args.input_datasets:
        name, json_path, img_dir = parse_dataset_input(dataset_str)

        dataset_configs.append({
            "name": name,
            "json_path": json_path,
            "img_dir": img_dir,
        })

    print("Input datasets:")
    for cfg in dataset_configs:
        print(f"  - {cfg['name']}")
        print(f"    JSON: {cfg['json_path']}")
        print(f"    Images: {cfg['img_dir']}")
    print()

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"ERROR: Ratios must sum to 1.0, got {total_ratio}")
        return 1

    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
        print()
        return 0

    # Step 1: Merge datasets
    print("STEP 1: Merging datasets")
    print("-" * 80)

    merged_json = os.path.join(args.output_dir, "merged_annotations.json")
    os.makedirs(os.path.dirname(merged_json), exist_ok=True)

    if len(dataset_configs) == 1:
        print("Single dataset - skipping merge")
        merged_json = dataset_configs[0]["json_path"]
        merged_img_dir = dataset_configs[0]["img_dir"]
    else:
        merger = COCOMerger(
            merge_categories=args.merge_categories,
            validate_annotations=True,
            clip_bboxes=True,
        )

        # Pass dataset paths with image directories
        dataset_paths = [
            (cfg["name"], cfg["json_path"], cfg["img_dir"])
            for cfg in dataset_configs
        ]
        merged_dataset = merger.merge_datasets(dataset_paths, merged_json)

        # Merger now tracks image directories per dataset
        # Processor will handle multiple base directories
        merged_img_dir = None  # Not needed - processor uses source info

    print()

    # Step 2: Process dataset (split + resize + adjust annotations)
    print("STEP 2: Processing dataset")
    print("-" * 80)

    processor = DatasetProcessor(
        input_json=merged_json,
        input_img_dir=merged_img_dir,
        output_dir=args.output_dir,
        target_size=tuple(args.target_size),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        separator=args.separator,
        seed=args.seed,
        use_weighted_split=not args.use_uniform_split,
        copy_images=not args.symlink_images,
        validate_images=not args.skip_validation,
    )

    stats = processor.process()

    print()
    print("=" * 80)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print()
    print("Directory structure:")
    print(f"  {args.output_dir}/")
    print("    ├── train2017/")
    print("    ├── val2017/")
    print("    ├── test2017/")
    print("    └── annotations/")
    print("        ├── instances_train2017.json")
    print("        ├── instances_val2017.json")
    print("        ├── instances_test2017.json")
    print("        └── dataset_stats.json")
    print()
    print("Next steps:")
    print(f"  1. Create symlink: ln -s {os.path.abspath(args.output_dir)} datasets/YourDataset")
    print("  2. Create experiment file in exps/")
    print("  3. Train: python -m yolox.tools.train -f your_exp.py")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
