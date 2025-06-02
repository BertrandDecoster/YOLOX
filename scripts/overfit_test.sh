#!/bin/bash
# Script to run overfitting test on custom dataset

echo "Starting YOLOX overfitting test..."
echo "This test will train on a small dataset without augmentation to verify the training pipeline."
echo ""

# Activate virtual environment if needed (uncomment and modify path)
# source /path/to/your/venv/bin/activate

# Run training
python tools/train_overfit_test.py \
    -f exps/example/custom/yolox_tiny_overfit.py \
    -d 1 \
    -b 2 \
    --fp16 \
    -o \
    --cache

echo ""
echo "Training complete! Check the results in YOLOX_outputs/yolox_tiny_overfit/"
echo ""
echo "To evaluate on the training set (should show overfitting with high mAP):"
echo "python -m yolox.tools.eval -f exps/example/custom/yolox_tiny_overfit.py -c YOLOX_outputs/yolox_tiny_overfit/best_ckpt.pth -b 2 -d 1 --conf 0.001"