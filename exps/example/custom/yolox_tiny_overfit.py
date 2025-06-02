#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        
        # Model configuration - using tiny model for quick testing
        self.depth = 0.33
        self.width = 0.375
        self.num_classes = 1  # CHANGE THIS to match your number of classes
        
        # Data configuration
        self.data_dir = "datasets/custom_dataset"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        
        # Training configuration for overfitting test
        self.max_epoch = 100  # Enough epochs to see overfitting
        self.data_num_workers = 2  # Reduce for small dataset
        self.eval_interval = 5  # Evaluate every 5 epochs
        
        # Small batch size for overfitting test
        self.batch_size = 2  # Very small batch size
        
        # Input size
        self.input_size = (416, 416)  # Smaller size for faster training
        self.test_size = (416, 416)
        
        # Training hyperparameters optimized for overfitting
        self.basic_lr_per_img = 0.01 / 64.0  # Higher learning rate
        self.scheduler = "constant"  # No LR decay for overfitting test
        self.no_aug_epochs = 100  # Disable augmentation for entire training
        self.warmup_epochs = 1  # Minimal warmup
        self.min_lr_ratio = 1.0  # Keep LR constant
        
        # Disable augmentations for overfitting test
        self.mosaic_prob = 0.0  # No mosaic
        self.mixup_prob = 0.0   # No mixup
        self.hsv_prob = 0.0     # No HSV augmentation
        self.flip_prob = 0.0    # No flipping
        self.degrees = 0.0      # No rotation
        self.translate = 0.0    # No translation
        self.scale = (1.0, 1.0) # No scaling
        self.shear = 0.0        # No shearing
        self.enable_mixup = False
        
        # Loss weights - standard
        self.obj_loss_weight = 1.0
        self.cls_loss_weight = 1.0
        self.iou_loss_weight = 5.0
        
        # Other settings
        self.print_interval = 1  # Print every iteration
        self.save_history_ckpt = False  # Don't save intermediate checkpoints
        
        # Experiment name
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        
    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        """
        Override to ensure no augmentation for overfitting test
        """
        from yolox.data import (
            COCODataset,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import wait_for_the_master
        
        with wait_for_the_master():
            dataset = COCODataset(
                data_dir=self.data_dir,
                json_file=self.train_ann if not no_aug else self.val_ann,
                img_size=self.input_size,
                preproc=None,
                cache=cache_img,
            )
        
        # Always use no augmentation for overfitting test
        self.dataset = dataset
        
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
        
        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)
        
        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["sampler"] = sampler
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed
        
        train_loader = DataLoader(self.dataset, batch_size=batch_size, **dataloader_kwargs)
        
        return train_loader
    
    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        """
        Override to use custom validation dataset
        """
        from yolox.data import COCODataset, ValTransform
        
        valdataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else self.test_ann,
            name="val2017",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )
        
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)
        
        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        
        val_loader = torch.utils.data.DataLoader(
            valdataset, batch_size=batch_size, **dataloader_kwargs
        )
        
        return val_loader