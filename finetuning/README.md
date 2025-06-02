# YOLOX Finetuning Tools

This folder contains tools and guides for finetuning YOLOX models on custom datasets.

## Files Overview

### Training Scripts
- **`train_longer.py`** - Extended training script for overfitting tests
  - Trains YOLOX-Tiny on a single image for 1000 iterations
  - Designed to verify the model can learn from your data
  - Saves checkpoint as `extended_overfit_model.pth`
  - Usage: `python train_longer.py`

### Visualization Tools
- **`demo_finetune_visual.py`** - Main demo script for visualizing model predictions
  - Loads finetuned model and runs inference on images
  - Draws bounding boxes and saves annotated images
  - Provides detailed detection statistics
  - Usage: `python demo_finetune_visual.py --path <image_or_directory> --ckpt <model.pth>`

### Model Checkpoints
- **`extended_overfit_model.pth`** - Trained model checkpoint
  - YOLOX-Tiny model trained for 1000 iterations
  - Achieves 97.1% loss reduction on training image
  - Detects bicycles, cars, and trash cans

### Documentation
- **`FINETUNING.md`** - Comprehensive finetuning guide
  - Complete workflow from data gathering to deployment
  - Best practices for data labeling and augmentation
  - Training configuration and hyperparameter tuning

- **`OVERFITTING_TEST_GUIDE.md`** - Overfitting test guide
  - Step-by-step instructions for verifying training pipeline
  - Expected results and troubleshooting tips

### Utilities
- **`coco_formatter.py`** - Convert custom annotations to COCO format
  - Handles conversion from other formats (YOLO, Pascal VOC, etc.)
  - Creates properly formatted JSON files for YOLOX

## Quick Start

1. **Prepare your dataset** following the structure in FINETUNING.md
2. **Run overfitting test** to verify everything works:
   ```bash
   python train_longer.py
   ```
3. **Visualize results** on your images:
   ```bash
   python demo_finetune_visual.py --path ../datasets/trashcan_test/train2017/ --conf 0.05
   ```

## Notes

- These scripts are designed for CPU usage (Mac compatible)
- For production training, use GPU and the full YOLOX training pipeline
- The overfitting test intentionally trains on a single image to verify the pipeline