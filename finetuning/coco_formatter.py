from PIL import Image

import json
import os


def yolo_to_coco(yolo_dir, image_dir, class_file, output_file):
    """Convert YOLO format annotations to COCO format"""

    coco_format = {"images": [], "annotations": [], "categories": []}

    with open(class_file, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    categories = [{"id": i, "name": c} for i, c in enumerate(classes)]

    coco_format["categories"] = categories

    annotation_id = 1

    for idx, image_file in enumerate(os.listdir(image_dir)):
        if not image_file.endswith((".jpg", ".png")):
            continue

        img = Image.open(os.path.join(image_dir, image_file))
        width, height = img.size

        # Add image info
        image_info = {
            "id": idx,
            "file_name": image_file,
            "width": width,
            "height": height,
        }
        coco_format["images"].append(image_info)

        # Read corresponding YOLO annotation
        txt_file = os.path.splitext(image_file)[0] + ".txt"
        txt_path = os.path.join(yolo_dir, txt_file)

        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    class_id = int(parts[0])
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    bbox_width = float(parts[3]) * width
                    bbox_height = float(parts[4]) * height

                    # Convert to COCO format (x, y, width, height)
                    x = x_center - bbox_width / 2
                    y = y_center - bbox_height / 2

                    annotation = {
                        "id": annotation_id,
                        "image_id": idx,
                        "category_id": class_id,
                        "bbox": [x, y, bbox_width, bbox_height],
                        "area": bbox_width * bbox_height,
                        "segmentation": [],
                        "iscrowd": 0,
                    }
                    coco_format["annotations"].append(annotation)
                    annotation_id += 1

    # Save COCO format annotation
    with open(output_file, "w") as f:
        json.dump(coco_format, f, indent=4)


if __name__ == "__main__":
    class_file = (
        "datasets/trashcan_test/classes.txt"  # Path to your class file if needed
    )

    yolo_dir = "datasets/trashcan_test/train2017"
    image_dir = "datasets/trashcan_test/train2017"
    output_file = "datasets/trashcan_test/annotations/instances_train2017.json"
    yolo_to_coco(yolo_dir, image_dir, class_file, output_file)
    print(f"COCO format annotations saved to {output_file}")
    yolo_dir = "datasets/trashcan_test/val2017"
    image_dir = "datasets/trashcan_test/val2017"
    output_file = "datasets/trashcan_test/annotations/instances_val2017.json"
    yolo_to_coco(yolo_dir, image_dir, class_file, output_file)
    print(f"COCO format annotations saved to {output_file}")
