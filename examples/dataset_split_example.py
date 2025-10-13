#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Example demonstrating weighted dataset splitting.

This script shows how to use the weighted partition optimization to split
datasets with varying group sizes while maintaining accurate train/val/test ratios.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.dataset_splitter import split_dataset_by_groups, get_split_statistics


def example_basic_usage():
    """Basic example with simulated dataset."""
    print("=" * 80)
    print("Example 1: Basic Usage with Simulated Dataset")
    print("=" * 80)
    print()

    # Simulate a dataset with different group sizes
    files = []
    # Large groups (e.g., from videos with many frames)
    for video_id in range(3):
        for frame in range(100):
            files.append(f"video{video_id}_{frame:04d}_jpg.rf.hash{frame}.jpg")

    # Medium groups
    for scene_id in range(10):
        for img in range(20):
            files.append(f"scene{scene_id}_{img:03d}_jpg.rf.hash{img}.jpg")

    # Small groups
    for photo_id in range(50):
        for aug in range(5):
            files.append(f"photo{photo_id}_{aug}_jpg.rf.hash{aug}.jpg")

    print(f"Total files: {len(files)}")
    print()

    # Split with default weighted method
    train, val, test = split_dataset_by_groups(
        files,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        separator="_jpg.rf",
        seed=42,
        use_weighted=True  # Use optimization (default)
    )

    stats = get_split_statistics(train, val, test, separator="_jpg.rf")

    print("Results with WEIGHTED splitting (optimized):")
    print(f"  Train:      {stats['train']['files']:5} files ({stats['train']['ratio']:.1%}) "
          f"in {stats['train']['groups']:3} groups")
    print(f"  Validation: {stats['val']['files']:5} files ({stats['val']['ratio']:.1%}) "
          f"in {stats['val']['groups']:3} groups")
    print(f"  Test:       {stats['test']['files']:5} files ({stats['test']['ratio']:.1%}) "
          f"in {stats['test']['groups']:3} groups")
    print()

    # Compare with uniform splitting
    train_u, val_u, test_u = split_dataset_by_groups(
        files,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        separator="_jpg.rf",
        seed=42,
        use_weighted=False  # Legacy uniform method
    )

    stats_u = get_split_statistics(train_u, val_u, test_u, separator="_jpg.rf")

    print("Results with UNIFORM splitting (legacy):")
    print(f"  Train:      {stats_u['train']['files']:5} files ({stats_u['train']['ratio']:.1%}) "
          f"in {stats_u['train']['groups']:3} groups")
    print(f"  Validation: {stats_u['val']['files']:5} files ({stats_u['val']['ratio']:.1%}) "
          f"in {stats_u['val']['groups']:3} groups")
    print(f"  Test:       {stats_u['test']['files']:5} files ({stats_u['test']['ratio']:.1%}) "
          f"in {stats_u['test']['groups']:3} groups")
    print()


def example_randomization():
    """Example showing randomization options."""
    print("=" * 80)
    print("Example 2: Randomization Modes")
    print("=" * 80)
    print()

    # Create simple dataset
    files = []
    for group_id in range(20):
        size = 10 + group_id * 5  # Varying group sizes
        for i in range(size):
            files.append(f"group{group_id:03d}_{i:04d}_jpg.rf.hash{i}.jpg")

    print(f"Total files: {len(files)}")
    print()

    for shuffle_mode in ["none", "within_size_buckets", "full"]:
        train, val, test = split_dataset_by_groups(
            files,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            separator="_jpg.rf",
            seed=42,
            use_weighted=True,
            shuffle_mode=shuffle_mode
        )

        stats = get_split_statistics(train, val, test, separator="_jpg.rf")

        print(f"Shuffle mode: {shuffle_mode}")
        print(f"  Train:      {stats['train']['ratio']:.1%}")
        print(f"  Validation: {stats['val']['ratio']:.1%}")
        print(f"  Test:       {stats['test']['ratio']:.1%}")
        print()


def example_with_perturbation():
    """Example showing perturbation for additional randomness."""
    print("=" * 80)
    print("Example 3: Using Perturbation for More Randomness")
    print("=" * 80)
    print()

    # Create dataset
    files = []
    for group_id in range(30):
        size = 15 + (group_id % 5) * 3
        for i in range(size):
            files.append(f"data{group_id:03d}_{i:04d}_jpg.rf.hash{i}.jpg")

    print(f"Total files: {len(files)}")
    print()

    # Try different perturbation levels
    for perturb_prob in [0.0, 0.3, 0.5]:
        train, val, test = split_dataset_by_groups(
            files,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            separator="_jpg.rf",
            seed=42,
            use_weighted=True,
            shuffle_mode="within_size_buckets",
            perturbation_prob=perturb_prob
        )

        stats = get_split_statistics(train, val, test, separator="_jpg.rf")

        print(f"Perturbation probability: {perturb_prob:.1f}")
        print(f"  Train:      {stats['train']['ratio']:.1%} ({stats['train']['groups']} groups)")
        print(f"  Validation: {stats['val']['ratio']:.1%} ({stats['val']['groups']} groups)")
        print(f"  Test:       {stats['test']['ratio']:.1%} ({stats['test']['groups']} groups)")
        print()


def main():
    """Run all examples."""
    example_basic_usage()
    example_randomization()
    example_with_perturbation()


if __name__ == "__main__":
    main()
