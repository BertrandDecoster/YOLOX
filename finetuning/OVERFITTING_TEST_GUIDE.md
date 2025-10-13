# YOLOX Overfitting Test Guide

This guide helps you verify that your YOLOX training pipeline is working correctly by intentionally overfitting on a small dataset.

## Purpose

Before training on a large dataset, it's crucial to verify that:
1. Your data pipeline is correctly loading images and annotations
2. The model can learn from your data
3. Loss functions are working properly
4. The training loop is functioning correctly

## Setup

### 1. Prepare Your Small Dataset

Add 5-10 images to test overfitting:
```bash
# Add images
cp your_images/*.jpg datasets/custom_dataset/train2017/

# Use the same images for validation (to verify overfitting)
cp your_images/*.jpg datasets/custom_dataset/val2017/
```

### 2. Create Annotations

Create minimal COCO format annotations in `datasets/custom_dataset/annotations/`:

**instances_train2017.json** and **instances_val2017.json** (same content for overfitting test):
```json
{
    "images": [
        {
            "id": 1,
            "file_name": "image1.jpg",
            "height": 480,
            "width": 640
        }
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [100, 100, 200, 200],
            "area": 40000,
            "iscrowd": 0
        }
    ],
    "categories": [
        {
            "id": 1,
            "name": "your_object",
            "supercategory": "object"
        }
    ]
}
```

### 3. Update Configuration

Edit `exps/example/custom/yolox_tiny_overfit.py`:
```python
self.num_classes = 1  # Change to match your number of classes
```

## Running the Test

### Method 1: Using the Shell Script
```bash
./scripts/overfit_test.sh
```

### Method 2: Direct Python Command
```bash
python tools/train_overfit_test.py \
    -f exps/example/custom/yolox_tiny_overfit.py \
    -d 1 \
    -b 2 \
    --fp16 \
    -o \
    --cache
```

### Method 3: Using YOLOX Module
```bash
python -m yolox.tools.train \
    -f exps/example/custom/yolox_tiny_overfit.py \
    -d 1 \
    -b 2 \
    --fp16 \
    -o \
    --cache
```

## Expected Results

### During Training
- **Loss Decrease**: Total loss should decrease from ~10-50 to <1.0
- **Individual Losses**: cls_loss, iou_loss, obj_loss should all decrease
- **No Augmentation**: Images should not be augmented (flipped, color-shifted, etc.)

### After Training (Evaluation)
```bash
# Evaluate on training set (should show overfitting)
python -m yolox.tools.eval \
    -f exps/example/custom/yolox_tiny_overfit.py \
    -c YOLOX_outputs/yolox_tiny_overfit/best_ckpt.pth \
    -b 2 \
    -d 1 \
    --conf 0.001
```

Expected metrics for successful overfitting:
- **AP@0.5**: >0.90 (90%+)
- **AP@0.5:0.95**: >0.70 (70%+)
- Very high precision and recall

## Troubleshooting

### High Loss / No Learning
1. **Check annotations**: Verify bbox coordinates are correct and within image bounds
2. **Check image loading**: Verify images are found and loaded properly
3. **Check class IDs**: Ensure category_id starts from 1 (not 0)

### CUDA Out of Memory
1. Reduce batch size in the config file
2. Reduce input size: change `self.input_size = (320, 320)`
3. Disable fp16 if issues persist

### Dataset Not Found
1. Check paths in the experiment file
2. Ensure COCO JSON files exist in annotations folder
3. Verify image filenames match those in JSON

## Next Steps

Once overfitting is confirmed:
1. Add your full dataset
2. Create a proper train/val split (don't use same images)
3. Enable augmentations by creating a new experiment file
4. Train with normal hyperparameters

## Configuration Explained

The overfitting configuration disables:
- **All augmentations**: mosaic, mixup, flipping, rotation, etc.
- **Learning rate decay**: Keeps LR constant
- **Regularization**: Allows model to memorize data

This is intentional for testing purposes only!