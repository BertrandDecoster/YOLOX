#!/usr/bin/env python3
"""Debug script to visualize preprocessing and understand the 70-pixel offset issue"""

import os
import cv2
import numpy as np
import torch
from yolox.data.data_augment import preproc
from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
from yolox.utils import postprocess, vis, fuse_model
from yolox.data.datasets import COCO_CLASSES

def visualize_preprocessing():
    # Load test image
    img_path = "../datasets/trashcan_test/train2017/frame_0000.jpg"
    original_img = cv2.imread(img_path)
    
    if original_img is None:
        print(f"Failed to load image: {img_path}")
        return
    
    print(f"Original image shape: {original_img.shape}")
    
    # Apply preprocessing
    input_size = (416, 416)
    preprocessed_img, ratio = preproc(original_img, input_size)
    
    print(f"Preprocessing ratio: {ratio}")
    print(f"Preprocessed shape: {preprocessed_img.shape}")
    
    # Convert back to HWC for visualization
    vis_preprocessed = preprocessed_img.transpose(1, 2, 0).astype(np.uint8)
    vis_preprocessed = np.ascontiguousarray(vis_preprocessed)
    
    # Draw grid on preprocessed image to show coordinates
    for i in range(0, 416, 50):
        cv2.line(vis_preprocessed, (i, 0), (i, 416), (255, 0, 0), 1)
        cv2.line(vis_preprocessed, (0, i), (416, i), (255, 0, 0), 1)
        cv2.putText(vis_preprocessed, str(i), (i+2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        cv2.putText(vis_preprocessed, str(i), (2, i+15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    
    # Calculate expected position after resize
    h, w = original_img.shape[:2]
    new_h, new_w = int(h * ratio), int(w * ratio)
    print(f"Resized dimensions: {new_w}x{new_h}")
    print(f"Padding: right={416-new_w}, bottom={416-new_h}")
    
    # Save visualization
    cv2.imwrite("debug_original.jpg", original_img)
    cv2.imwrite("debug_preprocessed.jpg", vis_preprocessed)
    
    # Now test with model
    print("\nTesting with model...")
    model = create_model()
    model.eval()
    
    # Load checkpoint
    ckpt_path = "extended_overfit_model.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        print("Model loaded")
        
        # Run inference
        img_tensor = torch.from_numpy(preprocessed_img).unsqueeze(0).float()
        with torch.no_grad():
            outputs = model(img_tensor)
            outputs = postprocess(outputs, 81, 0.05, 0.3)
            
        if outputs[0] is not None:
            output = outputs[0].cpu()
            bboxes = output[:, 0:4]
            print(f"\nRaw model output boxes (in 416x416 space):")
            for i, box in enumerate(bboxes):
                print(f"  Box {i}: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
            
            # Scale back
            bboxes_scaled = bboxes / ratio
            print(f"\nScaled boxes (original image space):")
            for i, box in enumerate(bboxes_scaled):
                print(f"  Box {i}: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
            
            # Draw both on original image
            img_with_boxes = original_img.copy()
            for box in bboxes_scaled:
                x1, y1, x2, y2 = box.numpy().astype(int)
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_with_boxes, "Model", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imwrite("debug_model_boxes.jpg", img_with_boxes)
            print("\nSaved visualizations: debug_original.jpg, debug_preprocessed.jpg, debug_model_boxes.jpg")

def create_model(num_classes=81):
    """Create the same YOLOX-Tiny model as in training"""
    depth = 0.33
    width = 0.375
    in_channels = [256, 512, 1024]
    
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act="silu")
    head = YOLOXHead(num_classes, width, in_channels=in_channels, act="silu")
    model = YOLOX(backbone, head)
    
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03
    
    return model

if __name__ == "__main__":
    import os
    visualize_preprocessing()