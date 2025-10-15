#!/usr/bin/env python3
"""Visualize ground truth annotations to compare with model predictions"""

import json

import cv2
import numpy as np


def visualize_ground_truth():
    # Load image
    img_path = "datasets/trashcan_test/train2017/frame_0000.jpg"
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # Load YOLO annotations
    txt_path = "datasets/trashcan_test/train2017/frame_0000.txt"

    # Draw YOLO format boxes
    img_yolo = img.copy()
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1]) * w
            y_center = float(parts[2]) * h
            box_width = float(parts[3]) * w
            box_height = float(parts[4]) * h

            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            # Draw box
            cv2.rectangle(img_yolo, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                img_yolo,
                f"GT cls:{class_id}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

    cv2.imwrite("ground_truth_yolo.jpg", img_yolo)

    # Load COCO annotations
    coco_path = "datasets/trashcan_test/annotations/instances_train2017.json"
    with open(coco_path, "r") as f:
        coco_data = json.load(f)

    # Find image ID
    image_id = None
    for img_info in coco_data["images"]:
        if img_info["file_name"] == "frame_0000.jpg":
            image_id = img_info["id"]
            break

    # Draw COCO format boxes
    img_coco = img.copy()
    for ann in coco_data["annotations"]:
        if ann["image_id"] == image_id:
            x, y, w, h = ann["bbox"]
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            class_id = ann["category_id"]

            cv2.rectangle(img_coco, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                img_coco,
                f"COCO cls:{class_id}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

    cv2.imwrite("ground_truth_coco.jpg", img_coco)

    print("Saved ground truth visualizations:")
    print("- ground_truth_yolo.jpg (red boxes)")
    print("- ground_truth_coco.jpg (blue boxes)")


if __name__ == "__main__":
    visualize_ground_truth()
