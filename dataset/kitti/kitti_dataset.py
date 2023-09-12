import os
import cv2
import numpy as np
from dataset.kitti.utils.calib import Calib
from dataset.kitti.utils.object3d import Object3d

class KITTIDataset:
    def __init__(self,dataset_path, split="training") -> None:
        super(KITTIDataset, self).__init__()
        self.dataset_path = dataset_path
        self.images = os.path.join(dataset_path, "object", split, "image_2")
        self.points = os.path.join(dataset_path, "object", split, "velodyne")
        self.calib = os.path.join(dataset_path, "object", split, "calib")
        self.labels = os.path.join(dataset_path, "object", split, "label_2")
        self.preds_baseline = os.path.join(dataset_path, "preds", "baseline")
        self.preds_sgfnet = os.path.join(dataset_path, "preds", "sgfnet")
        # self.preds = "/home/wyl/ws/vis3d/data/kitti/preds/sgfnet/car/epoch_70/val/final_result/data"

        self.valsets = self.get_imgsets(split="val")


    def get_imgsets(self, split):
        with open(os.path.join(self.dataset_path, "ImageSets", split + ".txt"), 'r') as f:
            lines = f.readlines()
        imgsets = []
        for line in lines:
            imgsets.append(line.strip())
        return imgsets
        


    def get_calib(self, index):
        calib_path = os.path.join(self.calib, "{:06d}.txt".format(index))
        with open(calib_path) as f:
            lines = f.readlines()

        lines = list(filter(lambda x: len(x) and x != '\n', lines))
        dict_calib = {}
        for line in lines:
            key, value = line.split(":")
            dict_calib[key] = np.array([float(x) for x in value.split()])
        return Calib(dict_calib)

    def get_image(self, index):
        img_path = os.path.join(self.images, "{:06d}.png".format(index))
        return cv2.imread(img_path)

    def get_points(self, index):
        pcd_path = os.path.join(self.points, "{:06d}.bin".format(index))
        points = np.fromfile(pcd_path, dtype=np.float32, count=-1).reshape([-1, 4])
        return points[:, :3]
    
    def get_labels(self, index):
        calib = self.get_calib(index)
        labels_path = os.path.join(self.labels, "{:06d}.txt".format(index))
        with open(labels_path) as f:
            lines = f.readlines()
        lines = list(filter(lambda x: len(x) > 0 and x != '\n', lines))

        return [Object3d(x, calib) for x in lines]
    
    def get_preds_baseline(self, index):
        calib = self.get_calib(index)
        labels_path = os.path.join(self.preds_baseline, "{:06d}.txt".format(index))
        with open(labels_path) as f:
            lines = f.readlines()
        lines = list(filter(lambda x: len(x) > 0 and x != '\n', lines))

        return [Object3d(x, calib) for x in lines]
    
    def get_preds_sgfnet(self, index):
        calib = self.get_calib(index)
        labels_path = os.path.join(self.preds_sgfnet, "{:06d}.txt".format(index))
        with open(labels_path) as f:
            lines = f.readlines()
        lines = list(filter(lambda x: len(x) > 0 and x != '\n', lines))

        return [Object3d(x, calib) for x in lines]

    def get_objs(self, index):
        return self.get_labels(index), self.get_preds_baseline(index), self.get_preds_sgfnet(index), self.get_points(index), self.get_calib(index), self.get_image(index)