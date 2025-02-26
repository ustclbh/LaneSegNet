import glob
import os

import cv2
import numpy as np
from mmdet.utils.general_utils import mkdir, getPathList, path_join

from .custom import CustomDataset
from .pipelines import Compose
from .builder import DATASETS


@DATASETS.register_module
class CulaneDataset(CustomDataset):

    def __init__(self,
                 data_root,
                 data_list,
                 pipeline,
                 test_mode=False,
                 test_suffix='png',
                 seg_label_list= '/list/train_gt.txt'):
        self.img_prefix = data_root
        self.test_suffix = test_suffix
        self.test_mode = test_mode
        self.seg_label_list =  seg_label_list
        self.lane_seg_labels = []
        # read image list
        self.img_infos, self.annotations, self.lane_seg_labels = self.parser_datalist(data_list, self.seg_label_list)

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        # build data pipeline
        self.pipeline = Compose(pipeline)

    def parser_datalist(self, data_list, seg_label_list):
        img_infos, annotations, lane_seg_labels = [], [], []
        if os.path.isfile(data_list):
            with open(data_list) as f:
                lines = f.readlines()
                for line in lines:
                    items = line.strip().split()
                    img_dir = items[0]
                    lane_seg_label = items[1] if len(items) > 1 else None
                    
                    img_infos.append(img_dir)
                    if lane_seg_label:
                        lane_seg_labels.append(lane_seg_label)
                    if not self.test_mode:
                        anno_dir = img_dir.replace('.jpg', '.lines.txt')
                        annotations.append(anno_dir)
        else:
            self.img_prefix = ""
            img_infos = getPathList(data_list, self.test_suffix)
        if os.path.isfile(seg_label_list):
            with open(seg_label_list) as f:
                lines = f.readlines()
                for line in lines:
                    items = line.strip().split()
                    lane_seg_labels.append(items[1])

        return img_infos, annotations, lane_seg_labels

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            self.flag[i] = 1

    def __len__(self):
        return len(self.img_infos)

    def prepare_train_img(self, idx):
        imgname = path_join(self.img_prefix, self.img_infos[idx])
        sub_img_name = self.img_infos[idx]
        img = cv2.imread(imgname)
        offset_x = 0
        offset_y = 0
        ori_shape = img.shape
        kps, id_classes, id_instances = self.load_labels(
            idx, offset_x, offset_y)
            
        # Load lane segmentation label if available
        lane_seg_label = None
        if len(self.lane_seg_labels) > idx:
            lane_seg_path = path_join(self.img_prefix, self.lane_seg_labels[idx])
            lane_seg_label = cv2.imread(lane_seg_path, cv2.IMREAD_GRAYSCALE)
            
        results = dict(
            filename=imgname,
            sub_img_name=sub_img_name,
            img=img,
            gt_points=kps,
            id_classes=id_classes,
            id_instances=id_instances,
            img_shape=ori_shape,
            ori_shape=ori_shape,
            lane_seg_label=lane_seg_label
        )
        results = self.pipeline(results)
        return results

    def prepare_test_img(self, idx):
        imgname = path_join(self.img_prefix, self.img_infos[idx])
        sub_img_name = self.img_infos[idx]
        img = cv2.imread(imgname)
        ori_shape = img.shape
        results = dict(
            filename=imgname,
            sub_img_name=sub_img_name,
            img=img,
            gt_points=[],
            id_classes=[],
            id_instances=[],
            img_shape=ori_shape,
            ori_shape=ori_shape,
            lane_seg_label=None
        )
        return self.pipeline(results)

    def load_labels(self, idx, offset_x, offset_y):
        anno_dir = path_join(self.img_prefix, self.annotations[idx])
        shapes = []
        with open(anno_dir, 'r') as anno_f:
            lines = anno_f.readlines()
            for line in lines:
                coords = []
                coords_str = line.strip().split(' ')
                for i in range(len(coords_str) // 2):
                    coord_x = float(coords_str[2 * i]) + offset_x
                    coord_y = float(coords_str[2 * i + 1]) + offset_y
                    coords.append(coord_x)
                    coords.append(coord_y)
                if len(coords) > 3:
                    shapes.append(coords)
        id_classes = [1 for i in range(len(shapes))]
        id_instances = [i + 1 for i in range(len(shapes))]
        return shapes, id_classes, id_instances
