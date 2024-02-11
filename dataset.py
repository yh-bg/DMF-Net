# 数据处理
import os
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms
from os.path import join
# from transforms import Stretch
import cv2


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])

])

transform_pan = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[ .5], std=[.5])
])


class FlameSet():
    def __init__(self,image_dir):

        input_pan_dir = join(image_dir, "pan")
        pan=os.listdir(input_pan_dir)
        self.pan=[os.path.join(input_pan_dir,k) for k in pan]
        input_ms_dir = join(image_dir, "ms")
        ms = os.listdir(input_ms_dir)
        self.ms = [os.path.join(input_ms_dir, k) for k in ms]


        gt_dir = join(image_dir, "label")
        gt = os.listdir(gt_dir)
        self.gt = [os.path.join(gt_dir, k) for k in gt]
        self.filename = [os.path.join(gt_dir, k).split('/')[-1] for k in gt]

        self.ms_transform = transform
        self.pan_transform = transform_pan
        self.target_transform = transform

    def __getitem__(self, index):
        pan_path = self.pan[index]
        pan_img = cv2.imread(pan_path,flags=0)
        pan_img = np.reshape(pan_img,(256,256,1))

        ms_path = self.ms[index]
        ms_img = cv2.imread(ms_path,flags = 1)
        ms_img = np.reshape(ms_img, (64, 64, 3))

        gt_path = self.gt[index]
        gt_img = cv2.imread(gt_path,flags=1)
        gt_img = np.reshape(gt_img, (256, 256, 3))

        gt_filename = self.filename[index]

        if self.pan_transform:
            input_pan = self.pan_transform(pan_img)

        if self.ms_transform:
            input_ms = self.ms_transform(ms_img)

        if self.target_transform:
            target = self.target_transform(gt_img)

        return input_pan,input_ms,target,gt_filename

    def __len__(self):
        return len(self.pan)


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch
