#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

"""
Tests for weighted partition assignment optimization.
"""

import sys
from pathlib import Path

# Add parent directory to path to import tools
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.weighted_partition import (
    assign_weighted_items_to_partitions,
    get_partition_statistics,
)


def test_basic_assignment():
    """Test basic assignment with uniform weights."""
    print("=" * 80)
    print("Test: Basic Assignment (Uniform Weights)")
    print("=" * 80)

    items = [("A", 10), ("B", 10), ("C", 10), ("D", 10), ("E", 10)]
    ratios = [0.6, 0.2, 0.2]

    result = assign_weighted_items_to_partitions(items, ratios, seed=42)

    stats = get_partition_statistics(result, items, ratios)

    print(f"Total items: {stats['total_items']}")
    print(f"Total weight: {stats['total_weight']}")
    print()

    for p in stats["partitions"]:
        print(
            f"Partition {p['index']}: target={p['target_ratio']:.1%}, "
            f"actual={p['actual_ratio']:.1%}, deviation={p['deviation']:+.1%}, "
            f"items={p['item_count']}, weight={p['weight']}"
        )

    print()
    print(f"Mean Absolute Error: {stats['mean_absolute_error']:.3%}")
    print(f"Max Absolute Error: {stats['max_absolute_error']:.3%}")
    print(f"RMSE: {stats['rmse']:.3%}")
    print()


def test_varied_weights():
    """Test assignment with varied weights."""
    print("=" * 80)
    print("Test: Varied Weights")
    print("=" * 80)

    # Simulate realistic scenario: few large groups, many small groups
    items = [
        ("Large1", 100),
        ("Large2", 80),
        ("Medium1", 30),
        ("Medium2", 25),
        ("Medium3", 20),
    ] + [(f"Small{i}", 5) for i in range(10)]

    ratios = [0.7, 0.15, 0.15]

    result = assign_weighted_items_to_partitions(items, ratios, seed=42)

    stats = get_partition_statistics(result, items, ratios)

    print(f"Total items: {stats['total_items']}")
    print(f"Total weight: {stats['total_weight']}")
    print()

    for p in stats["partitions"]:
        print(
            f"Partition {p['index']}: target={p['target_ratio']:.1%}, "
            f"actual={p['actual_ratio']:.1%}, deviation={p['deviation']:+.1%}, "
            f"items={p['item_count']}, weight={p['weight']}"
        )

    print()
    print(f"Mean Absolute Error: {stats['mean_absolute_error']:.3%}")
    print(f"Max Absolute Error: {stats['max_absolute_error']:.3%}")
    print(f"RMSE: {stats['rmse']:.3%}")
    print()


def test_oversized_item():
    """Test handling of item larger than partition target."""
    print("=" * 80)
    print("Test: Oversized Item (Item > Partition Target)")
    print("=" * 80)

    # Item "Huge" is 60% of total, but test partition is only 15%
    items = [
        ("Huge", 150),
        ("Small1", 50),
        ("Small2", 50),
    ]
    ratios = [0.7, 0.15, 0.15]

    result = assign_weighted_items_to_partitions(items, ratios, seed=42)

    stats = get_partition_statistics(result, items, ratios)

    print(f"Total items: {stats['total_items']}")
    print(f"Total weight: {stats['total_weight']}")
    print()

    print("Item assignments:")
    for item_name, partition_idx in result.items():
        weight = next(w for n, w in items if n == item_name)
        print(f"  {item_name}: partition {partition_idx} (weight={weight})")
    print()

    for p in stats["partitions"]:
        print(
            f"Partition {p['index']}: target={p['target_ratio']:.1%}, "
            f"actual={p['actual_ratio']:.1%}, deviation={p['deviation']:+.1%}, "
            f"items={p['item_count']}, weight={p['weight']}"
        )

    print()
    print(f"Mean Absolute Error: {stats['mean_absolute_error']:.3%}")
    print(f"Max Absolute Error: {stats['max_absolute_error']:.3%}")
    print()


def test_impossible_oversized_item():
    """Test handling of very large items (actually possible but with high deviation)."""
    print("=" * 80)
    print("Test: Very Large Item (90% of total, largest target is 70%)")
    print("=" * 80)

    # Item is 90% of total, but largest partition is only 70%
    # This is actually solvable - the item goes to the 70% partition,
    # causing deviation but not violating hard constraints
    items = [
        ("VeryLarge", 900),
        ("Tiny1", 50),
        ("Tiny2", 50),
    ]
    ratios = [0.7, 0.15, 0.15]

    result = assign_weighted_items_to_partitions(items, ratios, seed=42)
    stats = get_partition_statistics(result, items, ratios)

    print(f"Total items: {stats['total_items']}")
    print(f"Total weight: {stats['total_weight']}")
    print()

    print("Note: Large item goes to partition 0 (causing deviation from target)")
    for p in stats["partitions"]:
        print(
            f"Partition {p['index']}: target={p['target_ratio']:.1%}, "
            f"actual={p['actual_ratio']:.1%}, deviation={p['deviation']:+.1%}"
        )

    print()
    print(f"MAE: {stats['mean_absolute_error']:.3%}")
    print()


def test_randomization_modes():
    """Test different randomization modes produce different results."""
    print("=" * 80)
    print("Test: Randomization Modes")
    print("=" * 80)

    items = [(f"Item{i}", 10 + i) for i in range(20)]
    ratios = [0.7, 0.15, 0.15]

    print("Testing different shuffle modes:")
    print()

    for shuffle_mode in ["none", "within_size_buckets", "full"]:
        result = assign_weighted_items_to_partitions(
            items, ratios, shuffle_mode=shuffle_mode, seed=42
        )
        stats = get_partition_statistics(result, items, ratios)

        print(f"Shuffle mode: {shuffle_mode}")
        for p in stats["partitions"]:
            print(
                f"  Partition {p['index']}: actual={p['actual_ratio']:.1%}, "
                f"deviation={p['deviation']:+.1%}"
            )
        print(f"  MAE: {stats['mean_absolute_error']:.3%}")
        print()


def test_perturbation():
    """Test perturbation randomization."""
    print("=" * 80)
    print("Test: Perturbation Randomization")
    print("=" * 80)

    items = [(f"Item{i}", 10) for i in range(30)]
    ratios = [0.7, 0.15, 0.15]

    print("Comparing no perturbation vs high perturbation:")
    print()

    for perturb_prob in [0.0, 0.5]:
        result = assign_weighted_items_to_partitions(
            items, ratios, perturbation_prob=perturb_prob, seed=42
        )
        stats = get_partition_statistics(result, items, ratios)

        print(f"Perturbation probability: {perturb_prob}")
        for p in stats["partitions"]:
            print(
                f"  Partition {p['index']}: actual={p['actual_ratio']:.1%}, "
                f"deviation={p['deviation']:+.1%}, items={p['item_count']}"
            )
        print(f"  MAE: {stats['mean_absolute_error']:.3%}")
        print()


def test_edge_cases():
    """Test edge cases."""
    print("=" * 80)
    print("Test: Edge Cases")
    print("=" * 80)

    # Empty list
    print("1. Empty list:")
    result = assign_weighted_items_to_partitions([], [0.7, 0.15, 0.15])
    print(f"   Result: {result}")
    print()

    # Single item
    print("2. Single item:")
    result = assign_weighted_items_to_partitions([("Only", 100)], [0.7, 0.15, 0.15])
    print(f"   Result: {result}")
    print()

    # All zero weights
    print("3. All zero weights:")
    items = [("A", 0), ("B", 0), ("C", 0)]
    result = assign_weighted_items_to_partitions(items, [0.7, 0.15, 0.15])
    print(f"   Result: {result}")
    print()

    # Single partition
    print("4. Single partition:")
    items = [("A", 10), ("B", 20), ("C", 30)]
    result = assign_weighted_items_to_partitions(items, [1.0])
    print(f"   Result: {result}")
    stats = get_partition_statistics(result, items, [1.0])
    print(f"   Actual ratio: {stats['partitions'][0]['actual_ratio']:.1%}")
    print()


def test_comparison_uniform_vs_weighted():
    """Compare uniform splitting vs weighted splitting."""
    print("=" * 80)
    print("Test: Comparison - Uniform vs Weighted Splitting")
    print("=" * 80)

    # Create realistic scenario: power-law distribution
    items = (
        [(f"VeryLarge{i}", 100) for i in range(3)]
        + [(f"Large{i}", 50) for i in range(5)]
        + [(f"Medium{i}", 20) for i in range(10)]
        + [(f"Small{i}", 5) for i in range(30)]
    )

    ratios = [0.7, 0.15, 0.15]
    total_weight = sum(w for _, w in items)

    print(f"Total items: {len(items)}")
    print(f"Total weight: {total_weight}")
    print(f"Target ratios: {ratios}")
    print()

    # Simulate uniform splitting (assign by group count, not weight)
    print("Method 1: UNIFORM SPLITTING (by item count)")
    n_items = len(items)
    train_idx = int(n_items * 0.7)
    val_idx = int(n_items * 0.85)

    uniform_assignments = {}
    for i, (name, _) in enumerate(items):
        if i < train_idx:
            uniform_assignments[name] = 0
        elif i < val_idx:
            uniform_assignments[name] = 1
        else:
            uniform_assignments[name] = 2

    uniform_stats = get_partition_statistics(uniform_assignments, items, ratios)

    for p in uniform_stats["partitions"]:
        print(
            f"  Partition {p['index']}: target={p['target_ratio']:.1%}, "
            f"actual={p['actual_ratio']:.1%}, deviation={p['deviation']:+.1%}, "
            f"items={p['item_count']}"
        )
    print(f"  MAE: {uniform_stats['mean_absolute_error']:.3%}")
    print(f"  Max Error: {uniform_stats['max_absolute_error']:.3%}")
    print()

    # Weighted splitting
    print("Method 2: WEIGHTED SPLITTING (by weight)")
    weighted_assignments = assign_weighted_items_to_partitions(
        items, ratios, seed=42
    )
    weighted_stats = get_partition_statistics(weighted_assignments, items, ratios)

    for p in weighted_stats["partitions"]:
        print(
            f"  Partition {p['index']}: target={p['target_ratio']:.1%}, "
            f"actual={p['actual_ratio']:.1%}, deviation={p['deviation']:+.1%}, "
            f"items={p['item_count']}"
        )
    print(f"  MAE: {weighted_stats['mean_absolute_error']:.3%}")
    print(f"  Max Error: {weighted_stats['max_absolute_error']:.3%}")
    print()

    # Calculate improvement
    improvement = (
        (uniform_stats["mean_absolute_error"] - weighted_stats["mean_absolute_error"])
        / uniform_stats["mean_absolute_error"]
        * 100
    )
    print(f"Improvement in MAE: {improvement:.1f}%")
    print()


def main():
    """Run all tests."""
    tests = [
        test_basic_assignment,
        test_varied_weights,
        test_oversized_item,
        test_impossible_oversized_item,
        test_randomization_modes,
        test_perturbation,
        test_edge_cases,
        test_comparison_uniform_vs_weighted,
    ]

    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"ERROR in {test_func.__name__}: {e}")
            import traceback

            traceback.print_exc()
            print()


if __name__ == "__main__":
    main()
