#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Demo script that actually visualizes the detections from the finetuned model
Shows the images with bounding boxes drawn
"""

from loguru import logger

import cv2
import numpy as np

import torch

from yolox.data.data_augment import preproc
from yolox.data.datasets import COCO_CLASSES
from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead
from yolox.utils import fuse_model, postprocess, vis

import argparse
import os
import time

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Finetuned Demo with Visualization!")
    parser.add_argument(
        "--path",
        default="./datasets/trashcan_test/train2017/",
        help="path to images or directory",
    )
    parser.add_argument(
        "--output_dir", default="./viz", help="directory to save output images"
    )

    # Model checkpoint
    parser.add_argument(
        "-c",
        "--ckpt",
        # default="weights/overfit_single_picture.pth",
        default="weights/yolox_tiny.pth",
        type=str,
        help="checkpoint file",
    )

    # Detection parameters
    parser.add_argument("--conf", default=0.05, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=416, type=int, help="test img size")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can be cpu/gpu",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="display images with cv2.imshow (requires display)",
    )

    return parser


def get_image_list(path):
    image_names = []
    if os.path.isdir(path):
        for maindir, subdir, file_name_list in os.walk(path):
            for filename in file_name_list:
                apath = os.path.join(maindir, filename)
                ext = os.path.splitext(apath)[1]
                if ext in IMAGE_EXT:
                    image_names.append(apath)
    else:
        image_names = [path]
    return sorted(image_names)


def create_model(num_classes=81):
    """Create the same YOLOX-Tiny model as in training"""
    depth = 0.33
    width = 0.375
    in_channels = [256, 512, 1024]

    backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act="silu")
    head = YOLOXHead(num_classes, width, in_channels=in_channels, act="silu")
    model = YOLOX(backbone, head)

    # Initialize
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03

    return model


class Predictor(object):
    def __init__(
        self,
        model,
        device="cpu",
        conf_thre=0.3,
        nms_thre=0.3,
        test_size=(416, 416),
        num_classes=81,
    ):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.confthre = conf_thre
        self.nmsthre = nms_thre
        self.test_size = test_size

    def inference(self, img_path):
        img_info = {"id": 0}
        img_info["file_name"] = os.path.basename(img_path)
        img = cv2.imread(img_path)

        if img is None:
            logger.error(f"Failed to load image: {img_path}")
            return None, None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float()
        img = img.to(self.device)

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            logger.info("Inference time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img, []

        output = output.cpu()

        bboxes = output[:, 0:4]
        # Preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        # Get detection info
        detections = []
        for i in range(len(cls)):
            class_id = int(cls[i])
            class_name = (
                COCO_CLASSES[class_id]
                if class_id < len(COCO_CLASSES)
                else f"class_{class_id}"
            )
            score = float(scores[i])
            bbox = bboxes[i].tolist()
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "score": score,
                    "bbox": bbox,
                }
            )

        # Draw bounding boxes
        # Fix class indices that are out of bounds for visualization
        cls_fixed = cls.clone()
        cls_fixed[cls >= len(COCO_CLASSES)] = len(COCO_CLASSES) - 1
        vis_res = vis(img, bboxes, scores, cls_fixed, cls_conf, COCO_CLASSES)

        return vis_res, detections


def main():
    args = make_parser().parse_args()

    logger.info("Creating model...")
    num_classes = 80 if "yolox" in args.ckpt else 81
    model = create_model(num_classes=num_classes)
    model.eval()

    # Load checkpoint
    if not os.path.exists(args.ckpt):
        logger.error(f"Checkpoint not found: {args.ckpt}")
        logger.info("Please run train_longer.py first to generate the checkpoint")
        return

    logger.info(f"Loading checkpoint from {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    logger.info("Checkpoint loaded successfully")
    if "loss" in ckpt:
        logger.info(f"Model was trained to loss: {ckpt['loss']:.4f}")

    model = model.to(args.device)

    # Fuse model for faster inference
    logger.info("Fusing model...")
    model = fuse_model(model)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create predictor
    predictor = Predictor(
        model,
        args.device,
        args.conf,
        args.nms,
        (args.tsize, args.tsize),
        num_classes=81,
    )

    # Get list of images
    image_paths = get_image_list(args.path)
    logger.info(f"Found {len(image_paths)} images to process")

    # Process each image
    all_results = {}

    for idx, image_path in enumerate(image_paths):
        logger.info(f"\nProcessing image {idx+1}/{len(image_paths)}: {image_path}")

        outputs, img_info = predictor.inference(image_path)
        if outputs is None:
            continue

        result_image, detections = predictor.visual(
            outputs[0] if outputs[0] is not None else None, img_info
        )

        # Count detections by class
        class_counts = {}
        for det in detections:
            class_name = det["class_name"]
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1

        # Log results
        logger.info(f"Total detections: {len(detections)}")
        if class_counts:
            logger.info(f"Detections by class: {class_counts}")
        else:
            logger.info("No detections found")

        # Save results
        all_results[os.path.basename(image_path)] = {
            "total": len(detections),
            "by_class": class_counts,
            "detections": detections,
        }

        # Save image
        save_path = os.path.join(args.output_dir, os.path.basename(image_path))
        cv2.imwrite(save_path, result_image)
        logger.info(f"Saved result to: {save_path}")

        # Display if requested
        if args.display:
            cv2.imshow("YOLOX Detections", result_image)
            key = cv2.waitKey(0)
            if key == 27 or key == ord("q"):  # ESC or 'q' to quit
                break

    if args.display:
        cv2.destroyAllWindows()

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("DETECTION SUMMARY")
    logger.info("=" * 60)

    total_detections = 0
    total_by_class = {}

    for img_name, results in all_results.items():
        logger.info(f"\n{img_name}:")
        logger.info(f"  Total detections: {results['total']}")
        if results["by_class"]:
            for class_name, count in results["by_class"].items():
                logger.info(f"    - {class_name}: {count}")
                total_by_class[class_name] = total_by_class.get(class_name, 0) + count
        total_detections += results["total"]

    logger.info("\n" + "-" * 60)
    logger.info(
        f"OVERALL: {total_detections} detections across {len(all_results)} images"
    )
    if total_by_class:
        logger.info("Total by class:")
        for class_name, count in sorted(total_by_class.items()):
            logger.info(f"  - {class_name}: {count}")

    logger.info(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
