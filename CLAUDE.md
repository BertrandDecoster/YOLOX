# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YOLOX is an anchor-free version of YOLO object detection with better performance. This is a PyTorch implementation that bridges research and industrial communities for computer vision object detection tasks.

## Essential Commands

### Installation
```bash
pip3 install -v -e .  # Install in development mode
```

### Training
```bash
# Train with model name (recommended approach)
python -m yolox.tools.train -n yolox-s -d 8 -b 64 --fp16 -o [--cache]

# Train with experiment file
python -m yolox.tools.train -f exps/default/yolox_s.py -d 8 -b 64 --fp16 -o [--cache]

# Common flags:
# -n: model name (yolox-s, yolox-m, yolox-l, yolox-x, yolox-nano, yolox-tiny)
# -f: path to experiment file (alternative to -n)
# -d: number of GPUs
# -b: total batch size (recommended: num_gpu * 8)
# --fp16: enable mixed precision training
# -o, --occupy: occupy GPU memory first
# --cache: cache images to RAM ('ram') or disk ('disk') for faster training
# -c, --ckpt: checkpoint file for fine-tuning or resuming
# --resume: resume training from checkpoint
# --logger: choose logger (tensorboard/wandb/mlflow, default: tensorboard)

# Training on custom data
python tools/train.py -f /path/to/your/Exp/file -d 8 -b 64 --fp16 -o -c /path/to/pretrained/weights
```

### Evaluation
```bash
# Evaluate model
python -m yolox.tools.eval -n yolox-s -c yolox_s.pth -b 64 -d 8 --conf 0.001 [--fp16] [--fuse]

# Speed test (single batch, single GPU)
python -m yolox.tools.eval -n yolox-s -c yolox_s.pth -b 1 -d 1 --conf 0.001 --fp16 --fuse

# Common flags:
# --fuse: fuse conv and bn layers for faster inference
# --conf: confidence threshold (default: 0.001)
```

### Demo/Inference
```bash
# Image inference
python tools/demo.py image -n yolox-s -c /path/to/yolox_s.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]

# Video inference
python tools/demo.py video -n yolox-s -c /path/to/yolox_s.pth --path /path/to/video --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]
```

### Export Models
```bash
# Export to ONNX
python tools/export_onnx.py -n yolox-s -c /path/to/yolox_s.pth --output-name yolox_s.onnx

# Export to TorchScript
python tools/export_torchscript.py -n yolox-s -c /path/to/yolox_s.pth
```

### Testing
```bash
# Run tests (limited test suite available)
pytest tests/
```

## Architecture Overview

### Core Components

**Experiment System (`yolox/exp/`):**
- The entire model configuration is managed through Exp files, which is the central design pattern
- `yolox_base.py` (Exp class): Base experiment class containing ALL model, training, and testing configurations
- All experiment files inherit from this base and override specific parameters
- Key configs: `num_classes`, `depth`, `width`, `input_size`, learning rates, augmentation settings, etc.
- Standard model configs are in `exps/default/` (yolox_s.py, yolox_m.py, etc.)
- Custom dataset examples in `exps/example/`

**Model Architecture (`yolox/models/`):**
- `yolox.py`: Main YOLOX model class with backbone + head architecture
- `yolo_pafpn.py`: YOLOPAFPN backbone (Feature Pyramid Network with Path Aggregation)
- `yolo_head.py`: YOLOXHead detection head (anchor-free design)
- `darknet.py`: CSPDarknet backbone implementation
- `network_blocks.py`: Common building blocks (Focus, SPP, CSPLayer, etc.)
- `losses.py`: Loss functions (IoU loss, BCE loss, L1 loss)

**Training Pipeline (`yolox/core/`):**
- `trainer.py`: Main Trainer class orchestrating the entire training loop
  - Manages distributed training, mixed precision (fp16), EMA
  - Handles data loading, optimization, checkpointing, logging
  - Key methods: `before_train()`, `train_in_epoch()`, `train_in_iter()`, `after_epoch()`
- `launch.py`: Distributed training launcher

**Data Pipeline (`yolox/data/`):**
- `datasets/`: COCO and VOC dataset implementations
- `data_augment.py`: Strong augmentations (Mosaic, MixUp, HSV, etc.)
- `mosaicdetection.py`: Mosaic augmentation wrapper
- `dataloading.py`: Custom dataloader implementation
- `data_prefetcher.py`: CUDA data prefetcher for performance
- `samplers.py`: Custom samplers for distributed training

**Evaluation (`yolox/evaluators/`):**
- `coco_evaluator.py`: COCO mAP evaluation
- `voc_evaluator.py`: VOC evaluation

**Tools (`tools/` and `yolox/tools/`):**
- Entry points for train, eval, demo, export operations
- `visualize_assign.py`: Assignment visualization tool

### Model Variants

Models are differentiated by `depth` and `width` multipliers:
- YOLOX-Nano: depth=0.33, width=0.25 (mobile)
- YOLOX-Tiny: depth=0.33, width=0.375 (mobile)
- YOLOX-S: depth=0.33, width=0.50 (small)
- YOLOX-M: depth=0.67, width=0.75 (medium)
- YOLOX-L: depth=1.00, width=1.00 (large)
- YOLOX-X: depth=1.33, width=1.25 (extra large)
- YOLOX-Darknet53: depth=1.00, width=1.00 with Darknet53 backbone

### Training Flow

1. Create/select an Exp file (inherits from `yolox/exp/yolox_base.py`)
2. Exp file defines:
   - Model architecture (via `get_model()`)
   - Dataset and dataloaders (via `get_dataset()`, `get_data_loader()`)
   - Optimizer and scheduler (via `get_optimizer()`, `get_lr_scheduler()`)
   - Evaluator (via `get_evaluator()`)
3. Trainer initializes from Exp and handles the complete training lifecycle
4. Training uses strong augmentations (Mosaic + MixUp) which are disabled in the last 15 epochs
5. Model checkpoints are saved with EMA weights if enabled

### Key Design Patterns

**Experiment-driven configuration:** Everything (model, data, training, eval) is configured in one Exp file rather than scattered config files.

**Inheritance-based customization:** Create custom experiments by inheriting `Exp` class and overriding only what changes (see `exps/example/` for examples).

**Modular components:** Backbone, head, dataset, evaluator are all swappable through the Exp file's getter methods.

**Strong augmentation with gradual annealing:** Mosaic/MixUp augmentations are disabled in the last `no_aug_epochs` (default: 15) for better convergence.

## Working with Custom Data

1. Prepare dataset in COCO or VOC format (or implement custom Dataset class in `yolox/data/datasets/`)
2. Create custom Exp file inheriting from `yolox.exp.Exp`:
   - Override `num_classes`, `depth`, `width` in `__init__()`
   - Override `get_dataset()`, `get_data_loader()`, `get_eval_loader()`, `get_evaluator()` to point to your data
3. Symlink dataset to `datasets/` directory
4. Train with: `python tools/train.py -f path/to/your_exp.py -d 8 -b 64 --fp16 -o -c /path/to/pretrained_weights`

Always use COCO pretrained weights for initialization (model head shape differences are handled automatically).

## Dataset Setup

Expected dataset location: `datasets/COCO/` (symlink your COCO dataset here)
```bash
cd <YOLOX_HOME>
ln -s /path/to/your/COCO ./datasets/COCO
```

## Important Notes

- The codebase uses module-based execution: `python -m yolox.tools.train` (not `python tools/train.py` in most cases, though both work)
- Mixed precision (--fp16) is highly recommended for speed
- Recommended batch size: num_gpus × 8
- The trainer automatically handles mosaic augmentation disabling and L1 loss addition in the final epochs
- EMA (Exponential Moving Average) is enabled by default and used for evaluation
- Distributed training requires proper NCCL configuration (handled by `configure_nccl()`)
- Model checkpoints include: latest, best, last_epoch, last_mosaic_epoch, and optionally epoch_N if `save_history_ckpt=True`
