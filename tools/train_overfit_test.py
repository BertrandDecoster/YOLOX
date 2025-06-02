#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Training script for overfitting test on small custom dataset.
This script is designed to verify that the model can learn/memorize a small dataset.

Usage:
    python tools/train_overfit_test.py

Expected behavior:
    - Training loss should decrease significantly (close to 0)
    - Model should achieve very high mAP on training data
    - This confirms the training pipeline is working correctly
"""

import argparse
import os
import sys
import torch

# Add YOLOX to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import configure_nccl, configure_omp, get_num_devices


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Overfit Test Training")
    
    # Use our custom experiment file
    parser.add_argument(
        "-f",
        "--exp_file",
        default="exps/example/custom/yolox_tiny_overfit.py",
        type=str,
        help="experiment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    
    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=2, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=1, type=int, help="device for training"
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "-c", "--ckpt", default=None, type=str, help="checkpoint file"
    )
    parser.add_argument(
    ` "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument("-e", "--start_epoch", default=None, type=int, help="resume training start epoch")
    
    # Other options
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "--cache",
        dest="cache",
        default=False,
        action="store_true",
        help="Caching imgs to RAM for fast training.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    
    return parser


def main(exp, args):
    if exp.seed is not None:
        import random
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        torch.cuda.manual_seed_all(exp.seed)
        
    # Configure environment
    configure_nccl()
    configure_omp()
    
    # Print configuration for verification
    print("\n" + "="*50)
    print("OVERFITTING TEST CONFIGURATION")
    print("="*50)
    print(f"Model: YOLOX-Tiny (width={exp.width}, depth={exp.depth})")
    print(f"Classes: {exp.num_classes}")
    print(f"Dataset: {exp.data_dir}")
    print(f"Batch size: {exp.batch_size}")
    print(f"Max epochs: {exp.max_epoch}")
    print(f"Learning rate: {exp.basic_lr_per_img * exp.batch_size}")
    print(f"Input size: {exp.input_size}")
    print("\nAugmentations: ALL DISABLED (for overfitting test)")
    print("="*50 + "\n")
    
    # Launch training
    trainer = exp.get_trainer(args)
    trainer.train()
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("Check the following to verify overfitting:")
    print("1. Training loss should be very low (close to 0)")
    print("2. Check logs in YOLOX_outputs/yolox_tiny_overfit/")
    print("3. Run evaluation on training set - should get very high mAP")
    print("="*50 + "\n")


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)
    
    if not args.experiment_name:
        args.experiment_name = exp.exp_name
    
    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()
    
    if args.batch_size is not None:
        exp.batch_size = args.batch_size
    
    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args),
    )