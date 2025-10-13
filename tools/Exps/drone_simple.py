# plates.py
from yolox.exp import Exp as MyExp

import os


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "datasets/FirstTest"
        self.train_ann = "train_annotations.coco.json"
        self.val_ann = "val_annotations.coco.json"
        self.test_ann = "test_annotations.coco.json"

        self.num_classes = 1

        self.max_epoch = 30
        self.data_num_workers = 8
        self.eval_interval = 1
