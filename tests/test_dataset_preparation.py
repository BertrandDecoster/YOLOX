#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

"""
Tests for dataset preparation pipeline.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
import numpy as np
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.dataset_utils import (
    validate_bbox,
    clip_bbox,
    scale_bbox,
    letterbox_image,
    adjust_bbox_for_letterbox,
    create_empty_coco_dataset,
    remap_ids,
)
from tools.dataset_merger import COCOMerger
from tools.dataset_processor import DatasetProcessor


def test_bbox_validation():
    """Test bbox validation functions."""
    print("=" * 80)
    print("Test: Bbox Validation")
    print("=" * 80)

    # Valid bbox
    is_valid, error = validate_bbox([10, 20, 50, 60], 640, 480)
    assert is_valid, f"Should be valid: {error}"
    print("✓ Valid bbox passed")

    # Negative dimensions
    is_valid, error = validate_bbox([10, 20, -50, 60], 640, 480)
    assert not is_valid, "Should be invalid (negative width)"
    print("✓ Negative width rejected")

    # Out of bounds
    is_valid, error = validate_bbox([700, 20, 50, 60], 640, 480)
    assert not is_valid, "Should be invalid (out of bounds)"
    print("✓ Out of bounds rejected")

    # Zero area
    is_valid, error = validate_bbox([10, 20, 0, 60], 640, 480, min_area=1.0)
    assert not is_valid, "Should be invalid (zero width)"
    print("✓ Zero area rejected")

    print()


def test_bbox_clipping():
    """Test bbox clipping."""
    print("=" * 80)
    print("Test: Bbox Clipping")
    print("=" * 80)

    # Bbox extending outside image
    bbox = [600, 400, 100, 100]  # Extends beyond 640x480
    clipped = clip_bbox(bbox, 640, 480)
    print(f"Original: {bbox}")
    print(f"Clipped: {clipped}")
    assert clipped[0] == 600
    assert clipped[1] == 400
    assert clipped[2] == 40  # 640 - 600
    assert clipped[3] == 80  # 480 - 400
    print("✓ Bbox clipped correctly")

    # Negative coordinates
    bbox = [-10, -20, 100, 100]
    clipped = clip_bbox(bbox, 640, 480)
    print(f"Negative coords: {bbox} -> {clipped}")
    assert clipped[0] == 0
    assert clipped[1] == 0
    assert clipped[2] == 90  # 100 - 10
    assert clipped[3] == 80  # 100 - 20
    print("✓ Negative coordinates handled")

    print()


def test_letterbox_image():
    """Test letterbox image resizing."""
    print("=" * 80)
    print("Test: Letterbox Image Resizing")
    print("=" * 80)

    # Create test image (640x480, landscape)
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Resize to 640x640 (square)
    resized, scale, (pad_left, pad_top) = letterbox_image(img, (640, 640))

    print(f"Original shape: {img.shape}")
    print(f"Resized shape: {resized.shape}")
    print(f"Scale: {scale:.4f}")
    print(f"Padding: left={pad_left}, top={pad_top}")

    assert resized.shape == (640, 640, 3), "Output shape incorrect"
    assert abs(scale - 640/640) < 0.01, "Scale incorrect"  # Limited by width
    print("✓ Letterbox resize correct")

    # Test bbox adjustment
    original_bbox = [100, 100, 200, 150]
    adjusted = adjust_bbox_for_letterbox(original_bbox, scale, pad_left, pad_top)
    print(f"Original bbox: {original_bbox}")
    print(f"Adjusted bbox: {adjusted}")
    print("✓ Bbox adjustment complete")

    print()


def test_id_remapping():
    """Test ID remapping."""
    print("=" * 80)
    print("Test: ID Remapping")
    print("=" * 80)

    dataset = {
        "images": [
            {"id": 1, "file_name": "img1.jpg", "height": 480, "width": 640},
            {"id": 2, "file_name": "img2.jpg", "height": 480, "width": 640},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40]},
            {"id": 2, "image_id": 2, "category_id": 2, "bbox": [50, 60, 70, 80]},
        ],
        "categories": [
            {"id": 1, "name": "cat1"},
            {"id": 2, "name": "cat2"},
        ]
    }

    # Remap with offsets
    remapped = remap_ids(
        dataset,
        img_id_offset=1000,
        ann_id_offset=2000,
        cat_id_mapping={1: 10, 2: 20}
    )

    print("Original image IDs:", [img["id"] for img in dataset["images"]])
    print("Remapped image IDs:", [img["id"] for img in remapped["images"]])

    assert remapped["images"][0]["id"] == 1001
    assert remapped["images"][1]["id"] == 1002
    print("✓ Image IDs remapped")

    assert remapped["annotations"][0]["id"] == 2001
    assert remapped["annotations"][0]["image_id"] == 1001
    assert remapped["annotations"][0]["category_id"] == 10
    print("✓ Annotation IDs remapped")

    assert remapped["categories"][0]["id"] == 10
    assert remapped["categories"][1]["id"] == 20
    print("✓ Category IDs remapped")

    print()


def test_dataset_merger():
    """Test merging multiple datasets."""
    print("=" * 80)
    print("Test: Dataset Merger")
    print("=" * 80)

    # Create two simple datasets
    dataset1 = {
        "images": [
            {"id": 1, "file_name": "d1_img1.jpg", "height": 480, "width": 640},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40], "area": 1200, "iscrowd": 0},
        ],
        "categories": [
            {"id": 1, "name": "drone", "supercategory": "object"},
        ]
    }

    dataset2 = {
        "images": [
            {"id": 1, "file_name": "d2_img1.jpg", "height": 480, "width": 640},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [50, 60, 70, 80], "area": 5600, "iscrowd": 0},
        ],
        "categories": [
            {"id": 1, "name": "drone", "supercategory": "object"},
        ]
    }

    # Create temp files
    with tempfile.TemporaryDirectory() as tmpdir:
        json1 = os.path.join(tmpdir, "dataset1.json")
        json2 = os.path.join(tmpdir, "dataset2.json")

        with open(json1, "w") as f:
            json.dump(dataset1, f)
        with open(json2, "w") as f:
            json.dump(dataset2, f)

        # Merge
        merger = COCOMerger(merge_categories=True)
        merged = merger.merge_datasets([
            ("dataset1", json1, tmpdir),
            ("dataset2", json2, tmpdir),
        ])

        print(f"Merged images: {len(merged['images'])}")
        print(f"Merged annotations: {len(merged['annotations'])}")
        print(f"Merged categories: {len(merged['categories'])}")

        assert len(merged["images"]) == 2, "Should have 2 images"
        assert len(merged["annotations"]) == 2, "Should have 2 annotations"
        assert len(merged["categories"]) == 1, "Should merge same category"
        print("✓ Datasets merged correctly")

        # Check IDs are unique
        img_ids = [img["id"] for img in merged["images"]]
        ann_ids = [ann["id"] for ann in merged["annotations"]]
        assert len(img_ids) == len(set(img_ids)), "Image IDs not unique"
        assert len(ann_ids) == len(set(ann_ids)), "Annotation IDs not unique"
        print("✓ All IDs are unique")

    print()


def test_end_to_end():
    """Test end-to-end pipeline with synthetic data."""
    print("=" * 80)
    print("Test: End-to-End Pipeline")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create synthetic dataset
        input_dir = os.path.join(tmpdir, "input")
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(input_dir)

        # Create some test images
        images = []
        annotations = []
        for i in range(10):
            filename = f"video1_{i:04d}_jpg.rf.hash{i}.jpg"
            img_path = os.path.join(input_dir, filename)

            # Create dummy image
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(img_path, img)

            images.append({
                "id": i,
                "file_name": filename,
                "height": 480,
                "width": 640,
            })

            # Add 1-2 annotations per image
            for j in range(1 + (i % 2)):
                annotations.append({
                    "id": len(annotations),
                    "image_id": i,
                    "category_id": 1,
                    "bbox": [100 + j*50, 100, 80, 60],
                    "area": 4800,
                    "iscrowd": 0,
                })

        dataset = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "drone", "supercategory": "object"}]
        }

        # Save dataset
        json_path = os.path.join(input_dir, "annotations.json")
        with open(json_path, "w") as f:
            json.dump(dataset, f)

        print(f"Created {len(images)} images, {len(annotations)} annotations")

        # Process dataset
        processor = DatasetProcessor(
            input_json=json_path,
            input_img_dir=input_dir,
            output_dir=output_dir,
            target_size=(640, 640),
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            separator="_jpg.rf",
            seed=42,
            use_weighted_split=True,
            copy_images=True,
            validate_images=False,  # Skip validation for synthetic images
        )

        stats = processor.process()

        print("\nResults:")
        for split in ["train", "val", "test"]:
            print(f"  {split}: {stats[split]['num_images']} images, "
                  f"{stats[split]['num_annotations']} annotations")

        # Verify outputs exist
        for split in ["train2017", "val2017", "test2017"]:
            img_dir = os.path.join(output_dir, split)
            json_file = os.path.join(output_dir, "annotations", f"instances_{split}.json")

            assert os.path.exists(img_dir), f"Missing {img_dir}"
            assert os.path.exists(json_file), f"Missing {json_file}"

            # Load and verify JSON
            with open(json_file) as f:
                split_data = json.load(f)
                print(f"  {split}: {len(split_data['images'])} images in JSON")

        print("✓ End-to-end pipeline successful")

    print()


def main():
    """Run all tests."""
    tests = [
        test_bbox_validation,
        test_bbox_clipping,
        test_letterbox_image,
        test_id_remapping,
        test_dataset_merger,
        test_end_to_end,
    ]

    print("Running dataset preparation tests...\n")

    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"ERROR in {test_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
            print()

    print("=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
