#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

"""
Dataset splitting utilities with anti-leakage grouping.

This module provides functions to split datasets into train/val/test sets
while avoiding data leakage by keeping related images together in the same split.
"""

import os
import random
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Literal

from .weighted_partition import (
    assign_weighted_items_to_partitions,
    get_partition_statistics as get_optimization_stats,
)


def extract_group_name(filename: str, separator: str = "_jpg.rf") -> Optional[str]:
    """
    Extract group name from filename using a heuristic to avoid data leakage.

    The heuristic:
    1. Find the separator in the filename
    2. Take everything before the separator
    3. Remove all trailing digits
    4. The result is the group name

    Examples:
        yoto00494_jpg.rf.3db065df7214ee61d389b9251b9fe16d.jpg -> 'yoto'
        video18_1561_jpg.rf.adbe31d78a75e071e7ba16e016c6aafb.jpg -> 'video'
        scene00931_jpg.rf.ab66bc3d36e503e9717549597913b3ee.jpg -> 'scene'
        pic_371_jpg.rf.b3136ae8b6bbf3b459c06221646c7115.jpg -> 'pic_'

    Args:
        filename: The filename to extract group name from
        separator: The separator string to look for (default: '_jpg.rf')

    Returns:
        Group name if separator found, None otherwise
    """
    # Get just the filename without path
    basename = os.path.basename(filename)

    # Check if separator exists in filename
    if separator not in basename:
        return None

    # Extract prefix before separator
    prefix = basename.split(separator)[0]

    # Remove trailing digits (and underscores before digits)
    # This regex removes all digit sequences from the end,
    group_name = re.sub(r"[\d]+$", "", prefix)

    # Handle edge case where everything was digits
    if not group_name:
        return prefix

    return group_name


def group_files_by_prefix(
    file_list: List[str], separator: str = "_jpg.rf"
) -> Dict[str, List[str]]:
    """
    Group files by their extracted group names.

    Files without the separator are grouped under a special '__ungrouped__' key.

    Args:
        file_list: List of file paths
        separator: The separator string to look for (default: '_jpg.rf')

    Returns:
        Dictionary mapping group_name -> list of files in that group
    """
    groups = defaultdict(list)

    for filepath in file_list:
        group_name = extract_group_name(filepath, separator)

        if group_name is None:
            # Files without the separator go to ungrouped
            groups["__ungrouped__"].append(filepath)
        else:
            groups[group_name].append(filepath)

    return dict(groups)


def split_dataset_by_groups(
    file_list: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    separator: str = "_jpg.rf",
    seed: int = 42,
    use_weighted: bool = True,
    shuffle_mode: Literal["none", "within_size_buckets", "full"] = "within_size_buckets",
    random_tie_breaking: bool = True,
    perturbation_prob: float = 0.0,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split dataset into train/val/test sets by grouping related files together.

    This function ensures that all files from the same group stay in the same split,
    preventing data leakage when files with similar names are related (e.g., frames
    from the same video or augmented versions of the same image).

    Args:
        file_list: List of file paths to split
        train_ratio: Ratio of data for training (default: 0.7)
        val_ratio: Ratio of data for validation (default: 0.15)
        test_ratio: Ratio of data for testing (default: 0.15)
        separator: The separator string to identify related files (default: '_jpg.rf')
        seed: Random seed for reproducibility (default: 42)
        use_weighted: If True, use weighted assignment algorithm to match ratios more
            closely. If False, use simple uniform group splitting (default: True)
        shuffle_mode: Randomization mode for weighted assignment (default: 'within_size_buckets')
            - "none": Process groups largest-first
            - "within_size_buckets": Randomize within logarithmic size buckets
            - "full": Fully random order
        random_tie_breaking: Use random tie-breaking in weighted assignment (default: True)
        perturbation_prob: Probability of random perturbation in weighted assignment (default: 0.0)

    Returns:
        Tuple of (train_files, val_files, test_files)

    Raises:
        ValueError: If ratios don't sum to 1.0
    """
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )

    # Group files
    groups = group_files_by_prefix(file_list, separator)

    if use_weighted:
        # Use weighted assignment for better ratio matching
        train_files, val_files, test_files = _split_weighted(
            groups,
            train_ratio,
            val_ratio,
            test_ratio,
            seed,
            shuffle_mode,
            random_tie_breaking,
            perturbation_prob,
        )
    else:
        # Use legacy uniform splitting
        train_files, val_files, test_files = _split_uniform(
            groups, train_ratio, val_ratio, test_ratio, seed
        )

    return train_files, val_files, test_files


def _split_uniform(
    groups: Dict[str, List[str]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Legacy uniform splitting: split groups by count, not by weight.

    This method ignores group sizes and simply divides groups uniformly.
    """
    # Set random seed for reproducibility
    random.seed(seed)

    # Get list of group names
    group_names = list(groups.keys())

    # Shuffle groups (not individual files!)
    random.shuffle(group_names)

    # Calculate split indices
    n_groups = len(group_names)
    train_idx = int(n_groups * train_ratio)
    val_idx = int(n_groups * (train_ratio + val_ratio))

    # Split groups
    train_groups = group_names[:train_idx]
    val_groups = group_names[train_idx:val_idx]
    test_groups = group_names[val_idx:]

    # Collect files from each split
    train_files = []
    val_files = []
    test_files = []

    for group_name in train_groups:
        train_files.extend(groups[group_name])

    for group_name in val_groups:
        val_files.extend(groups[group_name])

    for group_name in test_groups:
        test_files.extend(groups[group_name])

    return train_files, val_files, test_files


def _split_weighted(
    groups: Dict[str, List[str]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    shuffle_mode: Literal["none", "within_size_buckets", "full"],
    random_tie_breaking: bool,
    perturbation_prob: float,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Weighted splitting: assign groups based on their sizes to match target ratios.

    This method uses the weighted partition optimization to achieve more accurate
    train/val/test ratios when groups have varying sizes.
    """
    # Convert groups to weighted items (name -> file count)
    weighted_items = [(group_name, len(files)) for group_name, files in groups.items()]

    # Assign groups to partitions using optimization
    target_ratios = [train_ratio, val_ratio, test_ratio]
    assignments = assign_weighted_items_to_partitions(
        weighted_items,
        target_ratios,
        shuffle_mode=shuffle_mode,
        random_tie_breaking=random_tie_breaking,
        perturbation_prob=perturbation_prob,
        seed=seed,
    )

    # Collect files based on assignments
    train_files = []
    val_files = []
    test_files = []

    for group_name, partition_idx in assignments.items():
        if partition_idx == 0:
            train_files.extend(groups[group_name])
        elif partition_idx == 1:
            val_files.extend(groups[group_name])
        elif partition_idx == 2:
            test_files.extend(groups[group_name])

    return train_files, val_files, test_files


def get_split_statistics(
    train_files: List[str],
    val_files: List[str],
    test_files: List[str],
    separator: str = "_jpg.rf",
) -> Dict:
    """
    Get statistics about a dataset split.

    Args:
        train_files: List of training files
        val_files: List of validation files
        test_files: List of test files
        separator: The separator used for grouping

    Returns:
        Dictionary containing split statistics
    """
    train_groups = group_files_by_prefix(train_files, separator)
    val_groups = group_files_by_prefix(val_files, separator)
    test_groups = group_files_by_prefix(test_files, separator)

    total_files = len(train_files) + len(val_files) + len(test_files)

    return {
        "total_files": total_files,
        "train": {
            "files": len(train_files),
            "groups": len(train_groups),
            "ratio": len(train_files) / total_files if total_files > 0 else 0,
        },
        "val": {
            "files": len(val_files),
            "groups": len(val_groups),
            "ratio": len(val_files) / total_files if total_files > 0 else 0,
        },
        "test": {
            "files": len(test_files),
            "groups": len(test_groups),
            "ratio": len(test_files) / total_files if total_files > 0 else 0,
        },
    }
