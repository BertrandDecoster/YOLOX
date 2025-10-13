# YOLOX Dataset Preparation Guide

## Overview

This guide explains how to prepare your datasets for YOLOX training. The pipeline handles:
- Merging multiple COCO datasets
- Anti-leakage splitting (train/val/test)
- Image resizing with letterbox (maintains aspect ratio)
- Annotation adjustment (bbox scaling, ID remapping)
- Output in YOLOX-compatible COCO format

## Quick Start

### Basic Usage (Auto-Inferred Image Directories)

```bash
# Images are automatically found in same directory as JSON
python tools/prepare_dataset.py \
  --input-datasets \
    datasets_raw/Dataset1/annotations.json \
    datasets_raw/Dataset2/annotations.json \
  --output-dir datasets/MyDroneDataset \
  --target-size 640 640
```

### With Custom Names

```bash
python tools/prepare_dataset.py \
  --input-datasets \
    DroneData1:datasets_raw/DroneData1/annotations.json \
    DroneData2:datasets_raw/DroneData2/annotations.json \
  --output-dir datasets/CombinedDrones \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15
```

### With Custom Image Directories (Override)

Only needed if images are not in the same directory as annotations:

```bash
python tools/prepare_dataset.py \
  --input-datasets \
    Dataset1:datasets_raw/Dataset1/coco.json:datasets_raw/Dataset1/images \
    Dataset2:datasets_raw/Dataset2/coco.json:/mnt/other/images \
  --output-dir datasets/Combined
```

## Pipeline Steps

### 1. Merge Datasets (if multiple)

The merger:
- Combines multiple COCO JSON files
- Remaps all IDs to avoid conflicts
- Merges categories by name (default) or keeps separate
- Validates and clips bounding boxes
- Filters invalid annotations

### 2. Split Dataset

Uses weighted group-based splitting:
- Groups images by filename prefix (anti-leakage)
- Weighted algorithm ensures accurate train/val/test ratios
- Respects group constraints (related images stay together)

Example: All frames from `video18_*` stay in same split.

### 3. Process Images

For each image:
- Validates (checks for corruption)
- Applies letterbox resize (maintains aspect ratio)
- Saves to appropriate split directory (train2017/, val2017/, test2017/)

### 4. Adjust Annotations

For each annotation:
- Scales bbox according to resize ratio
- Adds padding offset from letterbox
- Clips to image boundaries
- Filters invalid/too-small bboxes
- Remaps image IDs

## Output Structure

```
datasets/YourDataset/
├── train2017/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── val2017/
│   └── ...
├── test2017/
│   └── ...
└── annotations/
    ├── instances_train2017.json
    ├── instances_val2017.json
    ├── instances_test2017.json
    ├── dataset_stats.json
    └── merged_annotations.json (intermediate file)
```

## Command-Line Options

### Input/Output

- `--input-datasets`: List of input datasets. Formats:
  - `path.json` (name and image dir auto-inferred)
  - `name:path.json` (image dir auto-inferred from JSON location)
  - `name:path.json:img_dir` (custom image directory override)
- `--output-dir`: Output directory (created in `datasets/`)

### Merging

- `--merge-categories`: Merge categories with same name (default: True)
- `--no-merge-categories`: Keep categories separate by dataset

### Splitting

- `--train-ratio`: Training set ratio (default: 0.7)
- `--val-ratio`: Validation set ratio (default: 0.15)
- `--test-ratio`: Test set ratio (default: 0.15)
- `--separator`: Separator for grouping (default: "_jpg.rf")
- `--seed`: Random seed (default: 42)
- `--use-uniform-split`: Use simpler uniform splitting (less accurate)

### Processing

- `--target-size`: Target image size in pixels (default: 640 640)
- `--skip-resize`: Skip resizing (copy original images)
- `--symlink-images`: Create symlinks instead of copying (faster)
- `--skip-validation`: Skip image validation (faster, risky)

### Other

- `--dry-run`: Preview what would be done without executing

## Examples

### Single Dataset

```bash
# Process single dataset (images auto-found in same directory)
python tools/prepare_dataset.py \
  --input-datasets datasets_raw/DroneOrig/annotations.json \
  --output-dir datasets/DroneProcessed
```

### Multiple Datasets with Different Categories

```bash
# Keep categories separate (prefixed by dataset name)
python tools/prepare_dataset.py \
  --input-datasets \
    drones:datasets_raw/drones/coco.json \
    birds:datasets_raw/birds/coco.json \
  --output-dir datasets/DronesAndBirds \
  --no-merge-categories
```

### Custom Split Ratios

```bash
# 80% train, 10% val, 10% test
python tools/prepare_dataset.py \
  --input-datasets datasets_raw/MyData/annotations.json \
  --output-dir datasets/MyDataProcessed \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1
```

### Different Image Size

```bash
# Resize to 1280x1280 for larger models
python tools/prepare_dataset.py \
  --input-datasets datasets_raw/HighRes/annotations.json \
  --output-dir datasets/HighResProcessed \
  --target-size 1280 1280
```

### Fast Processing (Symlinks)

```bash
# Use symlinks instead of copying (faster, but not portable)
python tools/prepare_dataset.py \
  --input-datasets datasets_raw/Large/annotations.json \
  --output-dir datasets/LargeProcessed \
  --symlink-images \
  --skip-validation
```

## Understanding Anti-Leakage Grouping

The `--separator` option controls how images are grouped to prevent data leakage.

**Example filenames:**
```
video18_001_jpg.rf.hash1.jpg
video18_002_jpg.rf.hash2.jpg
video18_003_jpg.rf.hash3.jpg
scene05_010_jpg.rf.hash4.jpg
```

With separator `_jpg.rf`, these are grouped as:
- Group "video18": 3 images (frames 1-3 stay together)
- Group "scene05": 1 image

This prevents frames from the same video appearing in both train and validation sets.

## Training After Preparation

Once your dataset is prepared:

### 1. Create Symlink (Optional)

```bash
cd <YOLOX_ROOT>
ln -s datasets/YourDataset datasets/YOLOX_CUSTOM
```

### 2. Create Experiment File

Create `exps/example/yolox_custom.py`:

```python
import os
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 1  # Your number of classes
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Dataset settings
        self.data_dir = "datasets/YourDataset"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        self.max_epoch = 300
        self.data_num_workers = 4
        self.eval_interval = 10
```

### 3. Train

```bash
python -m yolox.tools.train \
  -f exps/example/yolox_custom.py \
  -d 1 \
  -b 8 \
  --fp16 \
  -o \
  -c yolox_s.pth  # Pretrained weights
```

## Troubleshooting

### "All images went to train split"

**Cause**: All images have the same group prefix (or separator not found).

**Solution**:
- Check your filenames match the separator pattern
- Adjust `--separator` to match your naming convention
- Use `--use-uniform-split` if no grouping needed

### "Bboxes too small after resize"

**Cause**: Small objects become tiny at target resolution.

**Solution**:
- Use larger `--target-size` (e.g., 1280x1280)
- Filter small objects from source data
- Adjust `min_area` in validation functions

### "Images are distorted"

**Note**: This shouldn't happen - letterbox maintains aspect ratio!

If you see this:
- Check source images aren't already corrupted
- Verify output images manually
- Report bug with example images

### "Out of memory during processing"

**Solution**:
- Process dataset in batches (split source data)
- Use `--symlink-images` to avoid copying
- Process at smaller `--target-size` first

## Advanced Usage

### Programmatic API

```python
from tools.dataset_merger import COCOMerger
from tools.dataset_processor import DatasetProcessor

# Merge
merger = COCOMerger(merge_categories=True)
merged = merger.merge_datasets([
    ("dataset1", "path/to/dataset1.json"),
    ("dataset2", "path/to/dataset2.json"),
], output_json="merged.json")

# Process
processor = DatasetProcessor(
    input_json="merged.json",
    input_img_dir="images/",
    output_dir="output/",
    target_size=(640, 640),
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
)
stats = processor.process()
```

### Custom Validation

Edit `tools/dataset_utils.py` to add custom validation logic:

```python
def validate_bbox(bbox, img_width, img_height, min_area=1.0, custom_check=None):
    # Add your custom validation
    if custom_check and not custom_check(bbox):
        return False, "Custom validation failed"
    # ... rest of validation
```

## Best Practices

1. **Keep datasets_raw/ untouched**: Never modify original data
2. **Use descriptive names**: Name datasets clearly (e.g., `DronesUrban_v2`)
3. **Document splits**: Note seed and ratios used for reproducibility
4. **Version datasets**: Use git or timestamps to track dataset versions
5. **Validate outputs**: Check a few images manually after processing
6. **Save statistics**: Keep `dataset_stats.json` for analysis

## Performance Tips

- Use `--symlink-images` for local development (faster)
- Use `--skip-validation` if you trust your source data
- Process at multiple resolutions for multi-scale training
- Run processing on machine with fast storage (SSD)

## References

- Dataset splitter: `tools/dataset_splitter.py`
- Weighted partition: `tools/weighted_partition.py`
- Merger: `tools/dataset_merger.py`
- Processor: `tools/dataset_processor.py`
- Utilities: `tools/dataset_utils.py`
- Tests: `tests/test_dataset_preparation.py`
