#!/usr/bin/env python3
"""
Extended training for YOLOX overfitting test
Trains for more iterations to actually learn the detections
"""

from loguru import logger

import torch
import torch.nn as nn

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from pycocotools.coco import COCO

from yolox.data.data_augment import TrainTransform, preproc
from yolox.data.datasets import COCODataset
from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead

# Configuration
num_classes = 81
batch_size = 1
img_size = (416, 416)
max_epochs = 1000  # Much more training
learning_rate = 0.001
device = torch.device("cpu")

logger.info("Running extended YOLOX training on CPU")


# Create model
def create_model():
    depth = 0.33
    width = 0.375
    in_channels = [256, 512, 1024]

    backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act="silu")
    head = YOLOXHead(num_classes, width, in_channels=in_channels, act="silu")
    model = YOLOX(backbone, head)

    # Initialize
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03

    model.head.initialize_biases(1e-2)
    return model


# Load dataset properly
def load_data():
    """Load the entire training dataset"""
    # Load annotations
    with open("datasets/trashcan_test/annotations/instances_train2017.json", "r") as f:
        coco_data = json.load(f)

    all_images = []
    all_targets = []

    for img_info in coco_data["images"]:
        img_path = os.path.join(
            "datasets/trashcan_test/train2017", img_info["file_name"]
        )

        # Load and preprocess image
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Could not load image: {img_path}")
            continue

        h, w = img.shape[:2]

        # Resize image
        img_resized, _ = preproc(img, img_size)
        img_tensor = torch.from_numpy(img_resized).float()

        # Get annotations for this image
        anns = [
            ann for ann in coco_data["annotations"] if ann["image_id"] == img_info["id"]
        ]

        # Convert to YOLOX format: [class_id, x_center, y_center, width, height]
        targets = []
        for ann in anns:
            x, y, w_box, h_box = ann["bbox"]
            # Convert to relative coordinates
            x_center = (x + w_box / 2) / w
            y_center = (y + h_box / 2) / h
            w_rel = w_box / w
            h_rel = h_box / h

            # Scale to resized image
            x_center *= img_size[0]
            y_center *= img_size[1]
            w_rel *= img_size[0]
            h_rel *= img_size[1]

            targets.append([ann["category_id"], x_center, y_center, w_rel, h_rel])

        if targets:  # Only add images with annotations
            targets_tensor = torch.tensor(targets)
            all_images.append(img_tensor)
            all_targets.append(targets_tensor)

    return all_images, all_targets


# Simple training loop focusing on one image
def train():
    logger.info("Creating model...")
    model = create_model()
    model.to(device)
    model.train()

    # For overfitting test, just use first image repeatedly
    logger.info("Loading first training image...")
    import json

    with open("datasets/trashcan_test/annotations/instances_train2017.json", "r") as f:
        coco_data = json.load(f)

    # Get first image
    img_info = coco_data["images"][0]
    img_path = os.path.join("datasets/trashcan_test/train2017", img_info["file_name"])

    # Load and preprocess image
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # Resize image
    img_resized, _ = preproc(img, img_size)
    img_tensor = torch.from_numpy(img_resized).unsqueeze(0).float()

    # Get annotations for this image
    anns = [
        ann for ann in coco_data["annotations"] if ann["image_id"] == img_info["id"]
    ]

    # Convert to YOLOX format: [class_id, x_center, y_center, width, height]
    targets = []
    for ann in anns:
        x, y, w_box, h_box = ann["bbox"]
        # Convert to relative coordinates
        x_center = (x + w_box / 2) / w
        y_center = (y + h_box / 2) / h
        w_rel = w_box / w
        h_rel = h_box / h

        # Scale to resized image
        x_center *= img_size[0]
        y_center *= img_size[1]
        w_rel *= img_size[0]
        h_rel *= img_size[1]

        targets.append([ann["category_id"], x_center, y_center, w_rel, h_rel])

    targets = torch.tensor(targets).unsqueeze(0) if targets else torch.zeros((1, 0, 5))

    img_tensor = img_tensor.to(device)
    targets = targets.to(device)

    logger.info(f"Loaded image: {img_path}")
    logger.info(f"Image shape: {img_tensor.shape}")
    logger.info(f"Number of targets: {targets.shape[1]}")
    logger.info(f"Target classes: {[int(t[0]) for t in targets[0]]}")

    # Optimizer with lower learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    logger.info("Starting extended training...")
    initial_loss = None
    best_loss = float("inf")

    for epoch in range(max_epochs):
        optimizer.zero_grad()

        # Forward pass - YOLOX expects targets during training
        outputs = model(img_tensor, targets)

        # YOLOX returns a dict with losses during training
        if isinstance(outputs, dict):
            loss = outputs["total_loss"]
            iou_loss = outputs.get("iou_loss", 0)
            conf_loss = outputs.get("conf_loss", 0)
            cls_loss = outputs.get("cls_loss", 0)
        else:
            loss = torch.tensor(0.0)

        if initial_loss is None:
            initial_loss = loss.item()

        # Backward
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)

        optimizer.step()

        # Log progress
        if epoch % 50 == 0:
            logger.info(
                f"Epoch {epoch}/{max_epochs}, Loss: {loss.item():.4f} "
                f"(iou: {iou_loss:.3f}, conf: {conf_loss:.3f}, cls: {cls_loss:.3f})"
            )

        # Save best model
        if loss.item() < best_loss:
            best_loss = loss.item()

    final_loss = loss.item()
    logger.info(f"\nTraining complete!")
    logger.info(f"Initial loss: {initial_loss:.4f}")
    logger.info(f"Final loss: {final_loss:.4f}")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Loss reduction: {(1 - final_loss/initial_loss)*100:.1f}%")

    # Save model
    logger.info("\nSaving model checkpoint...")
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": max_epochs,
        "loss": final_loss,
        "best_loss": best_loss,
    }
    checkpoint_path = "extended_overfit_model.pth"
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Model saved to: {checkpoint_path}")

    # Test inference on training image
    logger.info("\nTesting inference on training image...")
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        # During inference, YOLOX returns raw predictions
        from yolox.utils import postprocess

        predictions = postprocess(outputs, num_classes, 0.1, 0.3)

        if predictions[0] is not None:
            logger.info(f"Number of detections: {len(predictions[0])}")
            # Show detected classes
            if len(predictions[0]) > 0:
                detected_classes = predictions[0][:, 6].unique().int().tolist()
                logger.info(f"Detected classes: {detected_classes}")
        else:
            logger.warning("No detections found")


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback

        traceback.print_exc()
