#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Fixed YOLOX training script for finetuning on single image
Correctly handles coordinate transformation during preprocessing
"""

import argparse
import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from loguru import logger

from yolox.data.data_augment import preproc
from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

# Configuration
img_size = (416, 416)
max_epochs = 1000
learning_rate = 0.001
device = "cpu"  # Force CPU for Mac compatibility


def create_model(num_classes=81):
    """Create YOLOX-Tiny model with custom number of classes"""
    depth = 0.33
    width = 0.375
    in_channels = [256, 512, 1024]
    
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act="silu")
    head = YOLOXHead(num_classes, width, in_channels=in_channels, act="silu")
    model = YOLOX(backbone, head)
    
    # Initialize batch norm
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03
    
    return model


def train():
    logger.info("Creating model...")
    model = create_model()
    model.to(device)
    model.train()

    # For overfitting test, just use first image repeatedly
    logger.info("Loading first training image...")
    import json

    with open("../datasets/trashcan_test/annotations/instances_train2017.json", "r") as f:
        coco_data = json.load(f)

    # Get first image
    img_info = coco_data["images"][0]
    img_path = os.path.join("../datasets/trashcan_test/train2017", img_info["file_name"])

    # Load and preprocess image
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # Resize image - IMPORTANT: get the ratio
    img_resized, ratio = preproc(img, img_size)
    img_tensor = torch.from_numpy(img_resized).unsqueeze(0).float()

    # Get annotations for this image
    anns = [
        ann for ann in coco_data["annotations"] if ann["image_id"] == img_info["id"]
    ]

    # Convert to YOLOX format: [class_id, x_center, y_center, width, height]
    targets = []
    for ann in anns:
        x, y, w_box, h_box = ann["bbox"]
        
        # FIXED: Apply the same ratio transformation as the image
        # The image is resized by ratio and placed at top-left
        x_scaled = x * ratio
        y_scaled = y * ratio
        w_scaled = w_box * ratio
        h_scaled = h_box * ratio
        
        # Convert to center format
        x_center = x_scaled + w_scaled / 2
        y_center = y_scaled + h_scaled / 2
        
        targets.append([ann["category_id"], x_center, y_center, w_scaled, h_scaled])

    targets = torch.tensor(targets).unsqueeze(0) if targets else torch.zeros((1, 0, 5))

    img_tensor = img_tensor.to(device)
    targets = targets.to(device)

    logger.info(f"Loaded image: {img_path}")
    logger.info(f"Image shape: {img_tensor.shape}")
    logger.info(f"Number of targets: {targets.shape[1]}")
    logger.info(f"Preprocessing ratio: {ratio}")
    logger.info(f"Original image size: {w}x{h}")
    logger.info(f"Scaled image size: {int(w*ratio)}x{int(h*ratio)}")
    
    # Log the corrected target coordinates
    logger.info("Target boxes in 416x416 space:")
    for i, target in enumerate(targets[0]):
        logger.info(f"  Box {i}: class={int(target[0])}, center=({target[1]:.1f}, {target[2]:.1f}), size=({target[3]:.1f}, {target[4]:.1f})")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    logger.info("Starting training with fixed coordinates...")
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

    # Save model
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": max_epochs,
        "loss": final_loss,
    }
    save_path = "fixed_overfit_model.pth"
    torch.save(checkpoint, save_path)
    logger.info(f"Model saved to {save_path}")

    # Results summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Initial loss: {initial_loss:.4f}")
    logger.info(f"Final loss: {final_loss:.4f}")
    logger.info(f"Loss reduction: {(1 - final_loss/initial_loss) * 100:.1f}%")
    logger.info(f"Best loss: {best_loss:.4f}")

    # Test inference
    logger.info("\nTesting inference on training image...")
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        # In eval mode, model returns raw predictions
        from yolox.utils import postprocess
        outputs = postprocess(outputs, 81, 0.05, 0.3)
        
        if outputs[0] is not None:
            logger.info(f"Detected {len(outputs[0])} objects")
            detections = outputs[0].cpu()
            
            # Log detections
            for i, det in enumerate(detections):
                x1, y1, x2, y2, conf, score, cls = det
                logger.info(f"  Detection {i}: class={int(cls)}, conf={conf:.2f}, "
                          f"box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        else:
            logger.info("No objects detected")


if __name__ == "__main__":
    train()