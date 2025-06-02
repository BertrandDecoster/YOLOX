# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

YOLOX is an anchor-free object detection model - a PyTorch implementation of an improved YOLO (You Only Look Once) architecture. It achieves state-of-the-art performance (51.5% mAP on COCO with YOLOX-X) while maintaining a simpler design than traditional anchor-based detectors.

## Common Commands

### Installation
```bash
# Install YOLOX in development mode
pip3 install -v -e .
# or
python3 setup.py develop
```

### Training
```bash
# Train YOLOX-S on 8 GPUs with mixed precision
python -m yolox.tools.train -n yolox-s -d 8 -b 64 --fp16 -o

# Train with custom experiment config
python -m yolox.tools.train -f exps/default/yolox_s.py -d 8 -b 64 --fp16 -o

# Multi-machine training (on master node)
python tools/train.py -n yolox-s -b 128 --dist-url tcp://123.123.123.123:12312 --num_machines 2 --machine_rank 0
```

### Evaluation
```bash
# Evaluate model
python -m yolox.tools.eval -n yolox-s -c yolox_s.pth -b 64 -d 8 --conf 0.001 --fp16 --fuse

# Speed test (single batch)
python -m yolox.tools.eval -n yolox-s -c yolox_s.pth -b 1 -d 1 --conf 0.001 --fp16 --fuse
```

### Inference
```bash
# Run inference on image
python tools/demo.py image -n yolox-s -c /path/to/yolox_s.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]

# Run inference on video
python tools/demo.py video -n yolox-s -c /path/to/yolox_s.pth --path /path/to/video --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]
```

### Model Export
```bash
# Export to ONNX
python tools/export_onnx.py -n yolox-s -c yolox_s.pth --output yolox_s.onnx

# Export to TorchScript
python tools/export_torchscript.py -n yolox-s -c yolox_s.pth --output yolox_s.pt
```

### Code Quality
```bash
# Run linting (based on setup.cfg)
flake8 . --max-line-length=100 --max-complexity=18 --exclude=__init__.py

# Sort imports
isort . --line-length=100 --multi-line-output=3
```

### Testing
```bash
# Run tests with pytest
pytest tests/

# Run specific test
python -m unittest tests.utils.test_model_utils
```

## Architecture Overview

The YOLOX architecture consists of three main components:

1. **Backbone (CSPDarknet)**: Feature extraction network that processes input images
   - Located in `/yolox/models/darknet.py`
   - Uses Cross Stage Partial connections for efficiency

2. **Neck (YOLOPAFPN)**: Path Aggregation Feature Pyramid Network for multi-scale features
   - Located in `/yolox/models/yolo_pafpn.py`
   - Combines features from different scales

3. **Head (YOLOXHead)**: Decoupled head for classification and regression
   - Located in `/yolox/models/yolo_head.py`
   - Separate branches for classification, regression, and IoU prediction
   - Anchor-free design using center point detection

Key architectural innovations:
- **Decoupled Head**: Separates classification and localization tasks for better performance
- **Anchor-free**: Eliminates anchor boxes, simplifying the design
- **SimOTA**: Dynamic label assignment strategy for training
- **Strong augmentation**: Mosaic and MixUp augmentation for robust training

## Project Structure

- `/yolox/`: Core package
  - `core/`: Training loop and distributed training utilities
  - `data/`: Data loading, augmentation (Mosaic, MixUp), and caching
  - `models/`: Model architectures (backbone, neck, head)
  - `evaluators/`: COCO and VOC evaluation metrics
  - `exp/`: Base experiment class and builders
  - `utils/`: Logging, checkpointing, visualization

- `/exps/`: Pre-configured experiment files for different model variants
  - `default/`: Standard models (nano, tiny, s, m, l, x)
  - `example/`: Examples for custom datasets

- `/tools/`: Entry points for training, evaluation, and deployment
  - `train.py`: Main training script
  - `eval.py`: Model evaluation
  - `demo.py`: Inference demonstration
  - `export_*.py`: Model conversion tools

- `/demo/`: Deployment examples for various frameworks
  - ONNXRuntime, TensorRT, OpenVINO, MegEngine, ncnn

## Key Implementation Details

- **Mixed Precision Training**: Uses PyTorch AMP for faster training with `--fp16` flag
- **Distributed Training**: Supports multi-GPU and multi-machine training via PyTorch DDP
- **Data Pipeline**: Efficient data loading with prefetching and optional RAM caching
- **Augmentation**: Strong augmentation pipeline including Mosaic, MixUp, and HSV adjustments
- **Dynamic Input Size**: Supports training with varying input sizes for better generalization
- **EMA**: Exponential Moving Average for stable model weights during training

## Finetuning on Custom Data - Key Learnings

### CPU Training on Mac
YOLOX assumes CUDA availability by default. For CPU training:
- The trainer calls `torch.cuda.set_device()` which fails on Mac
- Workaround: Create custom training scripts that bypass the standard trainer
- Set device to "cpu" explicitly and disable CUDA-specific features

### COCO Dataset Format
- **Class IDs start from 0** in the categories list, but annotations use 1-based indexing
- **Class 80 (trashcan)** is at index 79 in COCO_CLASSES array - causes IndexError in visualization
- Fix: Map class IDs for visualization: `cls_fixed[cls == 80] = 79`

### Model Checkpoint Format
Standard YOLOX checkpoint structure:
```python
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(), 
    "epoch": epoch_num,
    "loss": final_loss,
}
```

### Overfitting Test Strategy
For verifying the training pipeline:
1. Train on single image for many iterations (1000+)
2. Disable all augmentations (mosaic, mixup, flip, etc.)
3. Use high learning rate (0.001) with no decay
4. Expected: >90% loss reduction, detections on training image

### Common Issues
1. **Import errors**: Need to install dependencies: `loguru`, `pycocotools`, `opencv-python`
2. **Path issues**: Use absolute paths or proper relative paths from YOLOX root
3. **Memory**: Batch size 1-2 for CPU training to avoid OOM
4. **Visualization**: `cv2.imshow()` blocks execution - use headless version for batch processing

### Useful File Organization
Keep finetuning tools separate in `/finetuning/` folder:
- Training scripts (train_longer.py)
- Visualization tools (demo_finetune_visual.py)
- Model checkpoints (.pth files)
- Documentation (FINETUNING.md, guides)

### CRITICAL: Coordinate Transformation for Correct Bounding Boxes
**IMPORTANT**: When training YOLOX with custom data, you MUST correctly handle coordinate transformations to avoid bounding box offset issues.

The `preproc` function maintains aspect ratio when resizing images:
- Images are resized by a ratio `r = min(target_size[0]/h, target_size[1]/w)`
- The resized image is placed at top-left of the target canvas with padding

**Incorrect approach (causes ~70 pixel offset):**
```python
# DON'T DO THIS - assumes image is stretched to target size
x_center *= img_size[0]
y_center *= img_size[1]
```

**Correct approach:**
```python
# Get preprocessing ratio
img_resized, ratio = preproc(img, img_size)

# Transform annotations using the same ratio
x_scaled = x * ratio
y_scaled = y * ratio
w_scaled = w * ratio
h_scaled = h * ratio

# Convert to center format
x_center = x_scaled + w_scaled / 2
y_center = y_scaled + h_scaled / 2
```

This ensures that regardless of input size (416x416, 640x640, etc.), the model learns correct object locations and bounding boxes appear in the right place during inference.