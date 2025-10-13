#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

"""
Weighted partition assignment optimization.

This module solves the problem of assigning weighted items to partition buckets
to match target ratios as closely as possible. This is a pure optimization problem
independent of any machine learning context.

Problem Statement:
    Given:
    - Target partition ratios: [r1, r2, ..., rn] where sum(ri) = 1.0
    - Weighted items: [(name1, w1), (name2, w2), ...]

    Assign each item to exactly one partition bucket such that the actual ratios
    are as close as possible to the target ratios.

Constraints:
    - Each item must be assigned to exactly one bucket
    - If an item's weight exceeds a partition's target capacity, it can only be
      assigned to buckets with sufficient capacity
"""

import random
from typing import Dict, List, Tuple, Optional, Literal


def assign_weighted_items_to_partitions(
    items: List[Tuple[str, float]],
    target_ratios: List[float],
    shuffle_mode: Literal["none", "within_size_buckets", "full"] = "none",
    random_tie_breaking: bool = True,
    perturbation_prob: float = 0.0,
    seed: Optional[int] = None,
) -> Dict[str, int]:
    """
    Assign weighted items to partition buckets to match target ratios.

    This function uses a greedy algorithm with constraint handling:
    1. Sort items by weight (largest first) to handle constraints early
    2. For each item, assign it to the bucket that is furthest from its target ratio
    3. Handle oversized items by assigning to the only valid bucket(s)

    Args:
        items: List of (name, weight) tuples. Weights must be non-negative.
        target_ratios: List of target ratios for each partition. Must sum to 1.0.
        shuffle_mode: How to randomize item order before assignment:
            - "none": Process items in order (largest first)
            - "within_size_buckets": Randomize within logarithmic size buckets
            - "full": Completely random order (may violate constraints more often)
        random_tie_breaking: If True, randomly choose among buckets when multiple
            are equally far from target. If False, choose first bucket.
        perturbation_prob: Probability (0.0-1.0) of randomly perturbing an assignment
            by trying to assign to a random bucket instead of the optimal one.
            Perturbation is only applied if it doesn't violate constraints.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary mapping item name -> partition index (0-indexed)

    Raises:
        ValueError: If ratios don't sum to 1.0, if any weight is negative,
                   or if an item is too large to fit in any partition.

    Examples:
        >>> items = [("A", 100), ("B", 50), ("C", 50)]
        >>> ratios = [0.7, 0.15, 0.15]
        >>> result = assign_weighted_items_to_partitions(items, ratios)
        >>> # Result might be: {"A": 0, "B": 1, "C": 2}
        >>> # Actual ratios: [100/200, 50/200, 50/200] = [0.5, 0.25, 0.25]

        >>> # With oversized item
        >>> items = [("Large", 150), ("Small1", 25), ("Small2", 25)]
        >>> ratios = [0.7, 0.15, 0.15]
        >>> result = assign_weighted_items_to_partitions(items, ratios)
        >>> # Large item (75% of total) can only go to partition 0 (target 70%)
    """
    # Validate inputs
    if abs(sum(target_ratios) - 1.0) > 1e-6:
        raise ValueError(
            f"Target ratios must sum to 1.0, got {sum(target_ratios):.6f}"
        )

    if not items:
        return {}

    if any(weight < 0 for _, weight in items):
        raise ValueError("All weights must be non-negative")

    # Set random seed
    if seed is not None:
        random.seed(seed)

    # Calculate total weight
    total_weight = sum(weight for _, weight in items)

    if total_weight == 0:
        # All weights are zero, distribute arbitrarily
        return {name: i % len(target_ratios) for i, (name, _) in enumerate(items)}

    # Initialize buckets
    n_buckets = len(target_ratios)
    bucket_weights = [0.0] * n_buckets
    assignments = {}

    # Sort items by weight (largest first for constraint handling)
    sorted_items = sorted(items, key=lambda x: x[1], reverse=True)

    # Apply shuffle mode
    if shuffle_mode == "within_size_buckets":
        sorted_items = _shuffle_within_size_buckets(sorted_items, seed)
    elif shuffle_mode == "full":
        random.shuffle(sorted_items)

    # Assign each item
    for item_name, item_weight in sorted_items:
        # Calculate current bucket ratios
        current_ratios = [
            w / total_weight if total_weight > 0 else 0 for w in bucket_weights
        ]

        # Calculate target weights for each bucket
        target_weights = [r * total_weight for r in target_ratios]

        # Find valid buckets (those that can accommodate this item)
        valid_buckets = _find_valid_buckets(
            item_weight, bucket_weights, target_weights, total_weight
        )

        if not valid_buckets:
            # No valid bucket - this item is too large for any partition
            raise ValueError(
                f"Item '{item_name}' with weight {item_weight} ({item_weight/total_weight:.1%} "
                f"of total) cannot fit in any partition. Largest partition target is "
                f"{max(target_ratios):.1%}."
            )

        # Apply perturbation with given probability
        if perturbation_prob > 0 and random.random() < perturbation_prob:
            if len(valid_buckets) > 1:
                # Try random bucket
                bucket_idx = random.choice(valid_buckets)
            else:
                # Only one valid bucket, no perturbation possible
                bucket_idx = valid_buckets[0]
        else:
            # Choose bucket that is furthest from its target (greedy)
            bucket_idx = _choose_best_bucket(
                valid_buckets,
                bucket_weights,
                target_weights,
                item_weight,
                random_tie_breaking,
            )

        # Assign item to chosen bucket
        assignments[item_name] = bucket_idx
        bucket_weights[bucket_idx] += item_weight

    return assignments


def _shuffle_within_size_buckets(
    sorted_items: List[Tuple[str, float]], seed: Optional[int]
) -> List[Tuple[str, float]]:
    """
    Shuffle items within logarithmic size buckets.

    This maintains approximate size ordering while adding randomness.
    Items are grouped into buckets based on log2(weight), then shuffled within each bucket.
    """
    if not sorted_items:
        return []

    # Group by log bucket
    import math

    buckets: Dict[int, List[Tuple[str, float]]] = {}
    for name, weight in sorted_items:
        if weight > 0:
            log_bucket = int(math.log2(weight))
        else:
            log_bucket = -1000  # Special bucket for zero weights

        if log_bucket not in buckets:
            buckets[log_bucket] = []
        buckets[log_bucket].append((name, weight))

    # Shuffle within each bucket and reconstruct
    result = []
    for log_bucket in sorted(buckets.keys(), reverse=True):
        bucket_items = buckets[log_bucket]
        random.shuffle(bucket_items)
        result.extend(bucket_items)

    return result


def _find_valid_buckets(
    item_weight: float,
    bucket_weights: List[float],
    target_weights: List[float],
    total_weight: float,
) -> List[int]:
    """
    Find buckets that can accommodate the item without exceeding reasonable bounds.

    A bucket is valid if adding the item wouldn't cause it to exceed its target
    by more than the item's weight. This ensures large items can still be assigned.
    """
    valid = []
    item_ratio = item_weight / total_weight if total_weight > 0 else 0

    for i in range(len(bucket_weights)):
        # Calculate what the ratio would be after adding this item
        new_weight = bucket_weights[i] + item_weight
        new_ratio = new_weight / total_weight if total_weight > 0 else 0

        # A bucket is valid if:
        # 1. The new ratio doesn't exceed 1.0 (obviously)
        # 2. The new ratio doesn't exceed (target + item_ratio)
        #    This allows large items to be assigned even if they exceed the target
        target_ratio = target_weights[i] / total_weight if total_weight > 0 else 0

        if new_ratio <= target_ratio + item_ratio:
            valid.append(i)

    # If no bucket passes the constraint, check if item is truly too large
    if not valid:
        # An item is too large if it alone (plus current bucket contents) would
        # exceed the sum of the target ratio + the item's own ratio
        # Only relax constraint if this is a first-item scenario or item can physically fit
        max_target_ratio = max(target_weights) / total_weight if total_weight > 0 else 0

        # Check if ANY bucket is still empty enough to potentially accept this item
        # The item needs: target_ratio + item_ratio >= new_ratio for at least one bucket
        for i in range(len(bucket_weights)):
            current_ratio = bucket_weights[i] / total_weight if total_weight > 0 else 0
            # If this bucket is empty or nearly empty, and it's the largest target, use it
            if current_ratio < 0.01 and target_weights[i] >= max(target_weights) - 1e-6:
                valid.append(i)

        # If still no valid buckets, the item is genuinely too large
        # Return empty list to trigger error in caller

    return valid


def _choose_best_bucket(
    valid_buckets: List[int],
    bucket_weights: List[float],
    target_weights: List[float],
    item_weight: float,
    random_tie_breaking: bool,
) -> int:
    """
    Choose the best bucket from valid options using a greedy heuristic.

    The best bucket is the one that is currently furthest below its target.
    This helps balance the distribution across buckets.
    """
    # Calculate deficit for each valid bucket
    deficits = []
    for i in valid_buckets:
        deficit = target_weights[i] - bucket_weights[i]
        deficits.append((deficit, i))

    # Sort by deficit (largest deficit first)
    deficits.sort(reverse=True, key=lambda x: x[0])

    # Check for ties
    max_deficit = deficits[0][0]
    tied_buckets = [i for deficit, i in deficits if abs(deficit - max_deficit) < 1e-6]

    if len(tied_buckets) > 1 and random_tie_breaking:
        return random.choice(tied_buckets)
    else:
        return tied_buckets[0]


def get_partition_statistics(
    assignments: Dict[str, int],
    items: List[Tuple[str, float]],
    target_ratios: List[float],
) -> Dict:
    """
    Calculate statistics about a partition assignment.

    Args:
        assignments: Dictionary mapping item name -> partition index
        items: Original list of (name, weight) tuples
        target_ratios: Target ratios for each partition

    Returns:
        Dictionary with statistics including actual ratios, deviations, and errors
    """
    # Create item weight lookup
    item_weights = {name: weight for name, weight in items}
    total_weight = sum(item_weights.values())

    # Calculate actual weights per partition
    n_partitions = len(target_ratios)
    partition_weights = [0.0] * n_partitions
    partition_counts = [0] * n_partitions

    for item_name, partition_idx in assignments.items():
        partition_weights[partition_idx] += item_weights[item_name]
        partition_counts[partition_idx] += 1

    # Calculate actual ratios
    actual_ratios = [
        w / total_weight if total_weight > 0 else 0 for w in partition_weights
    ]

    # Calculate deviations
    deviations = [actual - target for actual, target in zip(actual_ratios, target_ratios)]

    # Calculate error metrics
    mean_absolute_error = sum(abs(d) for d in deviations) / len(deviations)
    max_absolute_error = max(abs(d) for d in deviations)
    rmse = (sum(d ** 2 for d in deviations) / len(deviations)) ** 0.5

    return {
        "total_weight": total_weight,
        "total_items": len(assignments),
        "partitions": [
            {
                "index": i,
                "target_ratio": target_ratios[i],
                "actual_ratio": actual_ratios[i],
                "deviation": deviations[i],
                "weight": partition_weights[i],
                "item_count": partition_counts[i],
            }
            for i in range(n_partitions)
        ],
        "mean_absolute_error": mean_absolute_error,
        "max_absolute_error": max_absolute_error,
        "rmse": rmse,
    }
