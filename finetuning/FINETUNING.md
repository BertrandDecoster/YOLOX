# YOLOX Finetuning Guide

Complete guide for finetuning YOLOX models on custom datasets, from quick start to production deployment.

## Table of Contents
- [Quick Start](#quick-start)
- [Overfitting Test](#overfitting-test)
- [Available Tools](#available-tools)
- [Complete Finetuning Workflow](#complete-finetuning-workflow)

---

## Quick Start

### Prerequisites
```bash
# Install YOLOX in development mode
pip3 install -v -e .
```

### Verify Training Pipeline (Recommended First Step)
Before training on a full dataset, verify your setup works by overfitting on a single image:

```bash
# 1. Run overfitting test (200 epochs on single image)
cd finetuning
python overfit_single_picture.py

# 2. Visualize predictions
python viz_model_predictions.py --path ../datasets/trashcan_test/train2017/ --conf 0.05
```

Expected result: Model should achieve >90% loss reduction and detect objects in the training image.

### Training on Custom Dataset
Once overfitting test passes, proceed with full dataset training using the main YOLOX training pipeline (see [Complete Finetuning Workflow](#complete-finetuning-workflow) below).

---

## Overfitting Test

### Purpose
Verify that your training pipeline is working correctly before investing time in full dataset preparation:
1. Data pipeline correctly loads images and annotations
2. Model can learn from your data format
3. Loss functions are working properly
4. Training loop functions correctly

### Setup

#### 1. Prepare Small Test Dataset
Use 1-5 images for the overfitting test:

```bash
# Add test images
mkdir -p datasets/test_dataset/train2017
mkdir -p datasets/test_dataset/annotations
cp your_test_images/*.jpg datasets/test_dataset/train2017/
```

#### 2. Create COCO Format Annotations
Create `datasets/test_dataset/annotations/instances_train2017.json`:

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

**Note:** If you have YOLO format annotations, use `coco_formatter.py` to convert them.

#### 3. Update Training Script
Edit `overfit_single_picture.py` to point to your dataset:

```python
# Line 58-59: Update dataset paths
with open("datasets/test_dataset/annotations/instances_train2017.json", "r") as f:
    coco_data = json.load(f)

# Line 63: Update image directory
img_path = os.path.join("datasets/test_dataset/train2017", img_info["file_name"])
```

### Running the Test

```bash
cd finetuning
python overfit_single_picture.py
```

### Expected Results

**During Training (console output):**
- Initial loss: ~10-50
- Final loss: <1.0
- Loss reduction: >90%
- Individual losses (iou_loss, conf_loss, cls_loss) all decrease

**After Training:**
Model checkpoint saved to `weights/overfit_single_picture.pth`

**Verification:**
```bash
# Visualize predictions on training image
python viz_model_predictions.py \
    --path datasets/test_dataset/train2017/ \
    --ckpt weights/overfit_single_picture.pth \
    --conf 0.05
```

Check `viz/` directory for output images with bounding boxes. The model should detect all labeled objects with high confidence.

### Troubleshooting

**High Loss / No Learning:**
- Verify bbox coordinates are within image bounds
- Check that images load correctly
- Ensure category_id matches your class definitions
- Verify annotations use correct COCO format (x, y, width, height from top-left)

**Memory Issues:**
- Script uses CPU by default for Mac compatibility
- For GPU training, modify `device = "cpu"` to `device = "cuda"`
- Reduce image size in config: `img_size = (320, 320)`

**No Detections:**
- Lower confidence threshold: `--conf 0.01`
- Verify model loaded checkpoint correctly
- Check that bbox coordinates were transformed correctly during preprocessing

---

## Available Tools

### 1. overfit_single_picture.py
**Purpose:** Train YOLOX-Tiny on a single image to verify training pipeline

**Features:**
- Trains for 200 epochs on first image in dataset
- CPU-compatible (Mac development)
- Correctly handles coordinate transformations during preprocessing
- Saves checkpoint to `weights/overfit_single_picture.pth`

**Usage:**
```bash
python overfit_single_picture.py
```

**Key Parameters (edit in script):**
- `img_size`: Input size (default: 416x416)
- `max_epochs`: Training epochs (default: 200)
- `learning_rate`: Learning rate (default: 0.001)
- `device`: "cpu" or "cuda"

### 2. viz_model_predictions.py
**Purpose:** Visualize model predictions with bounding boxes on images

**Features:**
- Loads trained model checkpoint
- Runs inference on images or directories
- Draws bounding boxes with class labels
- Saves annotated images
- Provides detection statistics by class

**Usage:**
```bash
python viz_model_predictions.py --path <image_or_dir> --ckpt <checkpoint.pth> --conf 0.05

# Examples:
python viz_model_predictions.py --path ../datasets/test/train2017/ --conf 0.05
python viz_model_predictions.py --path image.jpg --ckpt weights/custom.pth --conf 0.25
```

**Arguments:**
- `--path`: Path to image file or directory
- `--ckpt`: Model checkpoint file (default: weights/overfit_single_picture.pth)
- `--conf`: Confidence threshold (default: 0.05)
- `--nms`: NMS threshold (default: 0.3)
- `--tsize`: Test image size (default: 416)
- `--output_dir`: Output directory for results (default: ./viz)
- `--display`: Show images with cv2.imshow (requires display)

**Output:**
- Annotated images saved to `viz/` directory
- Console logs with detection counts by class
- Summary statistics across all processed images

### 3. coco_formatter.py
**Purpose:** Convert YOLO format annotations to COCO format

**Features:**
- Reads YOLO .txt annotations (normalized coordinates)
- Converts to COCO JSON format
- Handles both train and validation splits
- Automatically calculates bbox areas

**Usage:**
1. Edit paths in script:
```python
class_file = "datasets/your_dataset/classes.txt"
yolo_dir = "datasets/your_dataset/train2017"
image_dir = "datasets/your_dataset/train2017"
output_file = "datasets/your_dataset/annotations/instances_train2017.json"
```

2. Run:
```bash
python coco_formatter.py
```

**YOLO Format (input):**
```
# classes.txt
bicycle
car
trash_can

# frame_0000.txt (class_id x_center y_center width height, all normalized 0-1)
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.15
```

**COCO Format (output):**
```json
{
  "images": [{"id": 0, "file_name": "frame_0000.jpg", "width": 640, "height": 480}],
  "annotations": [{"id": 1, "image_id": 0, "category_id": 0, "bbox": [x, y, w, h], "area": 0, "iscrowd": 0}],
  "categories": [{"id": 0, "name": "bicycle"}]
}
```

### 4. viz_compare_coco_yolo.py
**Purpose:** Compare YOLO and COCO ground truth annotations visually

**Features:**
- Loads both YOLO .txt and COCO .json annotations
- Visualizes bounding boxes from each format
- Helps verify format conversion accuracy
- Saves separate images for comparison

**Usage:**
1. Edit paths in script to point to your dataset
2. Run:
```bash
python viz_compare_coco_yolo.py
```

**Output:**
- `ground_truth_yolo.jpg`: Red bounding boxes from YOLO format
- `ground_truth_coco.jpg`: Blue bounding boxes from COCO format

Use this to verify that `coco_formatter.py` converted annotations correctly.

---

## Complete Finetuning Workflow

### 1. Data Gathering Strategy

**Quality over quantity principle**: Start with 500-1000 high-quality images per new class minimum. More diverse data leads to better generalization.

**Data collection approaches**:
- **Real-world capture**: Use multiple cameras, angles, lighting conditions, backgrounds, and distances. Capture objects in their natural context.
- **Web scraping**: Use Google Images, Flickr API, or specialized datasets. Ensure licensing compliance.
- **Synthetic generation**: Use 3D rendering, GANs, or augmentation tools for rare objects.
- **Video extraction**: Extract frames from videos at different intervals to get temporal variety.

**Critical considerations**:
- **Diversity**: Vary object sizes (small, medium, large in frame), orientations, occlusions, and environmental conditions.
- **Edge cases**: Include partially visible objects, crowded scenes, unusual angles, and challenging lighting.
- **Negative samples**: Include images without your target object to reduce false positives.
- **Class balance**: If adding to existing classes, maintain reasonable balance (no more than 10:1 ratio).

### 2. Data Labeling Process

**Annotation tools selection**:
- **LabelImg**: Simple, fast for small datasets, exports YOLO format directly.
- **CVAT**: Web-based, supports team collaboration, interpolation for videos.
- **Labelbox/V7/Roboflow**: Commercial options with AI-assisted labeling, quality control features.
- **Label Studio**: Open-source, flexible, supports pre-annotation with existing models.

**Labeling best practices**:
- **Tight bounding boxes**: Include all visible parts, but minimize background. Consistency is key.
- **Occlusion handling**: Label partially visible objects if >20% visible. Mark heavily occluded objects as "difficult".
- **Small objects**: Don't skip small instances - YOLOX handles multi-scale well.
- **Ambiguous cases**: Create labeling guidelines document. When uncertain, include rather than exclude.
- **Quality control**: Have 10% overlap between annotators, measure inter-annotator agreement.

**COCO format structure** (YOLOX native):
```json
{
  "images": [{"id": 1, "file_name": "img.jpg", "height": 480, "width": 640}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [x, y, w, h], "area": 0, "iscrowd": 0}],
  "categories": [{"id": 1, "name": "object_name", "supercategory": "category"}]
}
```

**Conversion from other formats**:
- **YOLO format**: Use `coco_formatter.py` utility (bbox center coordinates → top-left coordinates)
- **Pascal VOC**: XML parsing → JSON structure
- **Custom formats**: Write converters maintaining spatial accuracy

### 3. Dataset Preparation and Structure

**Directory organization**:
```
datasets/
├── your_dataset/
│   ├── train2017/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── val2017/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── annotations/
│   │   ├── instances_train2017.json
│   │   └── instances_val2017.json
```

**Data splitting strategy**:
- **Training set**: 70-80% of data. Should represent full diversity.
- **Validation set**: 10-15%. Used for hyperparameter tuning, early stopping.
- **Test set**: 10-15%. Never seen during training, final performance metric.
- **Stratified splitting**: Ensure each split has proportional class representation.

**Preprocessing considerations**:
- **Image standardization**: Convert to RGB, common formats (JPEG/PNG).
- **Resolution handling**: YOLOX handles multiple scales, but extreme variations need attention.
- **Corrupt file detection**: Validate all images load correctly before training.
- **Annotation validation**: Check bbox coordinates are within image bounds.

**Data augmentation planning**:
- **Built-in YOLOX augmentations**: Mosaic, MixUp, HSV shifts, affine transforms.
- **Additional augmentations**: Consider domain-specific augmentations (weather, blur, noise).
- **Augmentation probability tuning**: Start with defaults, adjust based on validation performance.

### 4. Model Configuration for New Classes

**Choosing base model architecture**:
- **YOLOX-Nano**: Mobile/edge deployment, <1M parameters.
- **YOLOX-Tiny/S**: Good balance, general purposes.
- **YOLOX-M/L**: Higher accuracy, more compute resources.
- **YOLOX-X**: Maximum accuracy, significant resources required.

**Transfer learning strategies**:
1. **Full finetuning**: Update all weights. Best for sufficient data (>5K images/class).
2. **Head-only finetuning**: Freeze backbone, update detection head. For limited data.
3. **Progressive unfreezing**: Start with head, gradually unfreeze deeper layers.

**Class configuration approaches**:
- **Adding to COCO classes**: Keep existing 80 classes, add new ones (81, 82, ...).
- **Replacing classes**: Modify specific class slots if similar objects.
- **Starting fresh**: New dataset with only your classes for specialized applications.

**Architecture modifications**:
- **Detection head channels**: Adjust for number of classes (num_classes parameter).
- **Anchor-free design**: No anchor configuration needed (YOLOX advantage).
- **FPN levels**: Consider modifying for specific object scales.
- **Loss weight balancing**: Adjust cls_loss, reg_loss, obj_loss ratios for your data.

**Pretrained weight handling**:
- **COCO pretrained**: Best starting point for most cases.
- **Partial loading**: Load compatible layers when class numbers differ.
- **Domain-specific pretrained**: Use if available (e.g., medical, aerial imagery).

### 5. Training Configuration and Hyperparameters

**Key hyperparameters for finetuning**:

**Learning rate scheduling**:
- **Initial LR**: Start lower than training from scratch (1e-4 to 1e-3).
- **Warmup**: 500-1000 iterations to stabilize training.
- **Schedule**: Cosine annealing or step decay. Reduce LR when validation plateaus.
- **Different LR for layers**: Higher LR for head, lower for backbone layers.

**Batch size optimization**:
- **Memory constraints**: Larger batches generally better, but fit within GPU memory.
- **Gradient accumulation**: Simulate larger batches on limited hardware.
- **Batch size-LR scaling**: Linear scaling rule - double batch size, double LR.
- **Minimum effective batch**: At least 16-32 for stable training with batch norm.

**Training duration**:
- **Epochs**: 50-100 for finetuning (vs 300 from scratch).
- **Early stopping**: Monitor validation mAP, patience of 10-20 epochs.
- **Overfitting indicators**: Training loss decreases but validation loss increases.

**Data augmentation tuning**:
- **Mosaic probability**: 0.5-1.0 for diverse scenes, lower for consistent backgrounds.
- **MixUp alpha**: 0.2-0.5, higher values for more regularization.
- **HSV augmentation**: Adjust based on lighting variation in your data.
- **Copy-paste**: Enable for small object detection improvement.

**Advanced configurations**:
- **Multi-scale training**: Enable for better scale invariance.
- **EMA decay**: 0.9998-0.9999 for stable predictions.
- **Label smoothing**: 0.0-0.1 to prevent overconfidence.
- **Gradient clipping**: Prevent explosion in early training.

### 6. Training Process and Monitoring

**Pre-training checklist**:
- Verify dataset paths and annotation loading.
- Test data pipeline with single batch visualization.
- Confirm GPU availability and memory requirements.
- Set up experiment tracking (tensorboard/wandb/mlflow).
- Create backup of original model weights.

**Training command** (using main YOLOX pipeline):
```bash
# Train with pretrained weights
python -m yolox.tools.train \
    -f exps/example/custom/yolox_s_custom.py \
    -d 8 \
    -b 64 \
    --fp16 \
    -o \
    --cache \
    -c /path/to/yolox_s.pth

# Common flags:
# -f: path to experiment file
# -d: number of GPUs
# -b: total batch size (recommended: num_gpu * 8)
# --fp16: enable mixed precision training
# -o, --occupy: occupy GPU memory first
# --cache: cache images to RAM for faster training
# -c: checkpoint file for fine-tuning
```

**Training monitoring metrics**:
- **Loss components**: Track cls_loss, reg_loss, obj_loss separately.
- **Learning rate**: Ensure proper scheduling and warmup.
- **Gradient norms**: Monitor for explosion/vanishing.
- **GPU utilization**: Optimize batch size for efficiency.
- **Training speed**: iterations/second, ETA tracking.

**Validation monitoring**:
- **mAP@0.5**: Primary metric for most applications.
- **mAP@0.5:0.95**: COCO-style metric for comprehensive evaluation.
- **Per-class AP**: Identify which classes need more data/tuning.
- **Inference speed**: FPS on validation images.
- **Confusion matrix**: Understand misclassification patterns.

**Common issues and solutions**:
- **Loss explosion**: Reduce learning rate, enable gradient clipping.
- **Slow convergence**: Increase learning rate, check data loading bottleneck.
- **Overfitting**: Add dropout, reduce model size, increase augmentation.
- **Class imbalance**: Use focal loss, adjust sample weights, oversample minority classes.
- **GPU OOM**: Reduce batch size, use gradient accumulation, enable mixed precision.

**Checkpointing strategy**:
- Save best model based on validation mAP.
- Regular checkpoints every N epochs.
- Keep last K checkpoints for rollback.
- Save optimizer state for resuming.

### 7. Evaluation and Validation Strategies

**Evaluation command**:
```bash
# Evaluate model
python -m yolox.tools.eval \
    -n yolox-s \
    -c /path/to/best_ckpt.pth \
    -b 64 \
    -d 8 \
    --conf 0.001 \
    --fp16 \
    --fuse

# Speed test (single batch, single GPU)
python -m yolox.tools.eval \
    -n yolox-s \
    -c /path/to/best_ckpt.pth \
    -b 1 \
    -d 1 \
    --conf 0.001 \
    --fp16 \
    --fuse
```

**Comprehensive evaluation metrics**:
- **Detection metrics**: Precision, Recall, F1-score at various IoU thresholds.
- **Speed metrics**: FPS, latency, throughput for deployment planning.
- **Error analysis**: False positives vs false negatives, failure case categorization.
- **Robustness testing**: Performance on edge cases, different lighting, occlusions.

**Test set best practices**:
- **True holdout**: Never use for hyperparameter tuning.
- **Realistic distribution**: Should match deployment conditions.
- **Temporal splitting**: For time-series data, use future dates for testing.
- **Cross-validation**: K-fold for small datasets to maximize data usage.

**Production readiness validation**:
- **Inference optimization**: Test with TensorRT/ONNX conversion.
- **Batch inference**: Verify performance at different batch sizes.
- **Memory profiling**: Peak memory usage during inference.
- **Edge case handling**: Empty images, extreme sizes, corrupted inputs.

**Demo/Inference**:
```bash
# Image inference
python tools/demo.py image \
    -n yolox-s \
    -c /path/to/yolox_s.pth \
    --path assets/dog.jpg \
    --conf 0.25 \
    --nms 0.45 \
    --tsize 640 \
    --save_result \
    --device gpu

# Video inference
python tools/demo.py video \
    -n yolox-s \
    -c /path/to/yolox_s.pth \
    --path /path/to/video.mp4 \
    --conf 0.25 \
    --nms 0.45 \
    --tsize 640 \
    --save_result \
    --device gpu
```

**A/B testing preparation**:
- **Baseline comparison**: Original model vs finetuned on same test set.
- **Statistical significance**: Ensure improvements are not random variation.
- **Real-world testing**: Deploy to small user group before full rollout.
- **Performance monitoring**: Set up alerts for accuracy degradation.

**Continuous improvement**:
- **Error mining**: Collect failure cases from production.
- **Active learning**: Identify uncertain predictions for labeling.
- **Data drift detection**: Monitor input distribution changes.
- **Retraining triggers**: Define when to update model (quarterly, performance-based).

### 8. Model Export and Deployment

**Export to ONNX**:
```bash
python tools/export_onnx.py \
    -n yolox-s \
    -c /path/to/yolox_s.pth \
    --output-name yolox_s.onnx
```

**Export to TorchScript**:
```bash
python tools/export_torchscript.py \
    -n yolox-s \
    -c /path/to/yolox_s.pth
```

**Final deployment checklist**:
- Model passes all accuracy thresholds.
- Inference speed meets requirements.
- Model size fits deployment constraints.
- Robustness testing completed.
- Documentation and versioning in place.

---

## Summary and Key Success Factors

**Critical success factors for YOLOX finetuning**:
1. **Data quality** trumps quantity - diverse, well-labeled data is essential.
2. **Start simple** - Use pretrained weights and proven architectures before customizing.
3. **Monitor everything** - Track metrics obsessively to catch issues early.
4. **Iterate quickly** - Small experiments to find optimal hyperparameters.
5. **Validate thoroughly** - Test edge cases before production deployment.

**Common pitfalls to avoid**:
- Insufficient data diversity leading to poor generalization.
- Overfitting due to aggressive learning rates or insufficient regularization.
- Ignoring class imbalance in datasets.
- Not validating on realistic test conditions.
- Skipping robustness testing for production deployment.

This comprehensive approach ensures successful addition of new object classes to YOLOX while maintaining high performance and reliability.
