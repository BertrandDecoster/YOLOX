#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

"""
Debugging script for dataset splitter.

This script helps verify the grouping logic by analyzing actual dataset files
and showing how they would be grouped to prevent data leakage.
"""

import os
import sys
from collections import Counter
from pathlib import Path

# Add parent directory to path to import tools
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.dataset_splitter import (
    extract_group_name,
    get_split_statistics,
    group_files_by_prefix,
    split_dataset_by_groups,
)


def test_extract_group_name():
    """Test the group name extraction logic."""
    print("=" * 80)
    print("Testing Group Name Extraction")
    print("=" * 80)

    test_cases = [
        "yoto00494_jpg.rf.3db065df7214ee61d389b9251b9fe16d.jpg",
        "video18_1561_jpg.rf.adbe31d78a75e071e7ba16e016c6aafb.jpg",
        "scene00931_jpg.rf.ab66bc3d36e503e9717549597913b3ee.jpg",
        "pic_371_jpg.rf.b3136ae8b6bbf3b459c06221646c7115.jpg",
        "yoto11108_jpg.rf.aba9c2b5010687a31d0195d40dc8f963.jpg",
        "video16_451_jpg.rf.d91a4b708f7b95ad1785661a981583a9.jpg",
        "0283_jpg.rf.67a118ac1fc5df4397e15395ad2b6c9d.jpg",
        "video12_jpg.rf.b7b9062659698b9a1418f211d0eca393.jpg",
        "zrenjanin-serbia-october-2015-image-260nw-334404434_jpg.rf.cab65e64fce268fd11438a2a9e0660dc.jpg",
    ]

    for filename in test_cases:
        group = extract_group_name(filename)
        print(f"  {filename[:60]:<60} -> '{group}'")

    print()


def analyze_dataset_directory(directory_path: str, separator: str = "_jpg.rf"):
    """Analyze a dataset directory and show grouping statistics."""
    print("=" * 80)
    print(f"Analyzing Dataset Directory: {directory_path}")
    print("=" * 80)

    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"ERROR: Directory does not exist: {directory_path}")
        print()
        return

    # Get all files
    all_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".jpg"):
                all_files.append(os.path.join(root, file))

    if not all_files:
        print(f"No .jpg files found in {directory_path}")
        print()
        return

    print(f"Total files found: {len(all_files)}")
    print()

    # Count files with separator
    files_with_separator = [f for f in all_files if separator in f]
    print(f"Files with separator '{separator}': {len(files_with_separator)}")
    print(f"Files without separator: {len(all_files) - len(files_with_separator)}")
    print()

    # Group files
    groups = group_files_by_prefix(all_files, separator)

    print(f"Total groups: {len(groups)}")
    print()

    # Get group size statistics
    group_sizes = [len(files) for group_name, files in groups.items()]
    group_size_counter = Counter(group_sizes)

    print("Group Size Distribution:")
    for size in sorted(group_size_counter.keys()):
        count = group_size_counter[size]
        print(f"  {count:4} groups with {size:4} files")
    print()

    # Show top groups by file count
    print("Top 20 Groups by File Count:")
    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
    for i, (group_name, files) in enumerate(sorted_groups[:20], 1):
        print(f"  {i:2}. '{group_name}': {len(files)} files")
        # Show first 3 file examples
        for j, filepath in enumerate(files[:3], 1):
            basename = os.path.basename(filepath)
            print(f"      {j}. {basename[:70]}")
        if len(files) > 3:
            print(f"      ... and {len(files) - 3} more files")
    print()


def test_dataset_split(directory_path: str, separator: str = "_jpg.rf"):
    """Test the dataset splitting functionality."""
    print("=" * 80)
    print("Testing Dataset Split")
    print("=" * 80)

    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"ERROR: Directory does not exist: {directory_path}")
        print()
        return

    # Get all files
    all_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".jpg"):
                all_files.append(os.path.join(root, file))

    if not all_files:
        print(f"No .jpg files found in {directory_path}")
        print()
        return

    # Perform split
    train_files, val_files, test_files = split_dataset_by_groups(
        all_files,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        separator=separator,
        seed=42,
    )

    # Get statistics
    stats = get_split_statistics(train_files, val_files, test_files, separator)

    print(f"Total files: {stats['total_files']}")
    print()
    print("Train Set:")
    print(f"  Files:  {stats['train']['files']:5} ({stats['train']['ratio']:.1%})")
    print(f"  Groups: {stats['train']['groups']:5}")
    print()
    print("Validation Set:")
    print(f"  Files:  {stats['val']['files']:5} ({stats['val']['ratio']:.1%})")
    print(f"  Groups: {stats['val']['groups']:5}")
    print()
    print("Test Set:")
    print(f"  Files:  {stats['test']['files']:5} ({stats['test']['ratio']:.1%})")
    print(f"  Groups: {stats['test']['groups']:5}")
    print()

    # Verify no group leakage
    train_groups = set(group_files_by_prefix(train_files, separator).keys())
    val_groups = set(group_files_by_prefix(val_files, separator).keys())
    test_groups = set(group_files_by_prefix(test_files, separator).keys())

    train_val_overlap = train_groups & val_groups
    train_test_overlap = train_groups & test_groups
    val_test_overlap = val_groups & test_groups

    print("Data Leakage Check:")
    print(f"  Train/Val overlap:  {len(train_val_overlap)} groups")
    print(f"  Train/Test overlap: {len(train_test_overlap)} groups")
    print(f"  Val/Test overlap:   {len(val_test_overlap)} groups")

    if train_val_overlap or train_test_overlap or val_test_overlap:
        print("  WARNING: Group leakage detected!")
        if train_val_overlap:
            print(f"    Overlapping groups (train/val): {list(train_val_overlap)[:5]}")
        if train_test_overlap:
            print(
                f"    Overlapping groups (train/test): {list(train_test_overlap)[:5]}"
            )
        if val_test_overlap:
            print(f"    Overlapping groups (val/test): {list(val_test_overlap)[:5]}")
    else:
        print("  ✓ No group leakage detected! All groups are in separate splits.")
    print()


def test_randomization_modes(directory_path: str, separator: str = "_jpg.rf"):
    """Test different randomization modes for weighted splitting."""
    print("=" * 80)
    print("Testing Weighted Split Randomization Modes")
    print("=" * 80)

    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"ERROR: Directory does not exist: {directory_path}")
        print()
        return

    # Get all files
    all_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".jpg"):
                all_files.append(os.path.join(root, file))

    if not all_files:
        print(f"No .jpg files found in {directory_path}")
        print()
        return

    print(f"Total files: {len(all_files)}")
    print()

    for shuffle_mode in ["none", "within_size_buckets", "full"]:
        print(f"Shuffle mode: {shuffle_mode}")

        train, val, test = split_dataset_by_groups(
            all_files,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            separator=separator,
            seed=42,
            use_weighted=True,
            shuffle_mode=shuffle_mode,
        )

        stats = get_split_statistics(train, val, test, separator)

        print(
            f"  Train:      {stats['train']['files']:5} files ({stats['train']['ratio']:.1%})"
        )
        print(
            f"  Validation: {stats['val']['files']:5} files ({stats['val']['ratio']:.1%})"
        )
        print(
            f"  Test:       {stats['test']['files']:5} files ({stats['test']['ratio']:.1%})"
        )

        error = (
            abs(stats["train"]["ratio"] - 0.7)
            + abs(stats["val"]["ratio"] - 0.15)
            + abs(stats["test"]["ratio"] - 0.15)
        ) / 3
        print(f"  MAE: {error:.3%}")
        print()


def main():
    """Main function to run all tests."""
    # Test group name extraction with examples
    test_extract_group_name()

    # Default dataset path
    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "datasets_raw", "DroneOrig"
    )

    # Allow custom path from command line
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]

    # Analyze the dataset directory
    analyze_dataset_directory(dataset_path)

    # Test the splitting functionality
    test_dataset_split(dataset_path)

    # Test randomization modes
    test_randomization_modes(dataset_path)


if __name__ == "__main__":
    main()
