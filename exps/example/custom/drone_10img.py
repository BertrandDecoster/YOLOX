#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch

from yolox.exp import Exp as MyExp

import os


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()

        # Model configuration - using YOLOX-S architecture
        self.depth = 0.33
        self.width = 0.50  # YOLOX-S width
        self.num_classes = 1  # Drone dataset has 1 class (only used category)

        # Data configuration - 10 image dataset
        self.data_dir = "datasets/DroneOverfit10"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        # Training configuration for overfitting test on CPU
        self.max_epoch = 10  # Stop after X epochs
        self.data_num_workers = (
            0  # IMPORTANT: Set to 0 for CPU/Mac to avoid multiprocessing issues
        )
        self.eval_interval = 1  # Evaluate every epoch to track overfitting

        # Small batch size for CPU overfitting test
        self.batch_size = 2  # Very small batch size for CPU

        # Input size - standard YOLOX-S resolution
        self.input_size = (640, 640)  # Standard YOLOX-S input size
        self.test_size = (640, 640)

        # Training hyperparameters optimized for overfitting
        self.basic_lr_per_img = 0.01 / 64.0  # Higher learning rate
        self.scheduler = "yoloxwarmcos"  # YOLOX scheduler
        self.no_aug_epochs = 100  # Disable augmentation for entire training (set high so it's always disabled)
        self.warmup_epochs = 1  # Minimal warmup
        self.min_lr_ratio = 1.0  # Keep LR constant (no decay)

        # Disable augmentations for overfitting test
        self.mosaic_prob = 0.0  # No mosaic
        self.mixup_prob = 0.0  # No mixup
        self.hsv_prob = 0.0  # No HSV augmentation
        self.flip_prob = 0.0  # No flipping
        self.degrees = 0.0  # No rotation
        self.translate = 0.0  # No translation
        self.scale = (1.0, 1.0)  # No scaling
        self.shear = 0.0  # No shearing
        self.enable_mixup = False

        # Lower confidence threshold to detect evaluation issues early
        self.test_conf = 0.001  # Very low threshold (default is 0.01)

        # Loss weights - standard
        self.obj_loss_weight = 1.0
        self.cls_loss_weight = 1.0
        self.iou_loss_weight = 5.0

        # Other settings
        self.print_interval = 1  # Print every iteration
        self.save_history_ckpt = False  # Don't save intermediate checkpoints

        # Experiment name
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_model(self):
        """
        Override to ensure model initialization is compatible with CPU
        """
        from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(
                self.depth, self.width, in_channels=in_channels, act=self.act
            )
            head = YOLOXHead(
                self.num_classes, self.width, in_channels=in_channels, act=self.act
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model

    def get_data_loader(
        self, batch_size, is_distributed, no_aug=False, cache_img=False
    ):
        """
        Override to ensure no augmentation and CPU-friendly settings
        """
        from yolox.data import (
            COCODataset,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            TrainTransform,
            YoloBatchSampler,
            worker_init_reset_seed,
        )
        from yolox.utils import wait_for_the_master

        with wait_for_the_master():
            dataset = COCODataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob
                ),
                cache=cache_img,
            )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob
            ),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        if is_distributed:
            from yolox.utils import get_world_size

            batch_size = batch_size // get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        # IMPORTANT: pin_memory=False for CPU training
        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": False}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    # Removed get_eval_loader and get_evaluator overrides - using base class versions which support trainset parameter and have pin_memory=False

    # Removed get_eval_loader and get_evaluator overrides - using base class versions which support trainset parameter and have pin_memory=False
