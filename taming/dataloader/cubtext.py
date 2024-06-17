# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import PIL
from typing import Any, Tuple, Union, Optional, Callable

import torch
from torchvision import transforms as T
from torchvision.datasets import ImageNet
from torch.utils.data import Dataset
from torchvision.datasets.folder import ImageFolder
import os
import numpy as np
import random

def read_images(path):
    f = open(path)
    lines = f.readlines()
    img_id2path = {}
    for line in lines:
        id, img_path = line.strip('\n').split(' ')
        img_id2path[id] = img_path
    f.close()
    return img_id2path

class CUBTrain(Dataset):
    def __init__(self, root='/workplace/dataset/cub', resolution: Union[Tuple[int, int], int] = 256, resize_ratio: float = 0.75, clip_text_truncate_len: int=30):
        images_path = os.path.join(root, 'images.txt')
        split_path = os.path.join(root, 'train_test_split.txt')
        self.clip_text_truncate_len = clip_text_truncate_len
        self.img_id2path = read_images(images_path)
        self.img_id2split = read_images(split_path)
        self.data = []
        self.labels = []
        self.name2label = {}
        for id, img_path in self.img_id2path.items():
            class_name = img_path.split('/')[0]
            if class_name not in self.name2label.keys():
                self.name2label[class_name] = len(self.name2label)
            if self.img_id2split[id] == '1':
                self.data.append(os.path.join(root, 'images', img_path))
                self.labels.append(root+'/birds/text/'+img_path[:-3]+'txt')

        self.transform = T.Compose([
            T.Resize(resolution),
            T.RandomCrop(resolution),
            T.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
        ])


    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = PIL.Image.open(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        img = (img/127.5 - 1.0).astype(np.float32)

        with open(label,'r') as f:
            lines = f.readlines()
            lines_filter = [i for i in lines if len(i.split(" ")) + 2 < self.clip_text_truncate_len]
            random_label = random.choice(lines_filter).replace('\n', '').lower()

        return {'image': img, 'text': random_label}

    def __len__(self):
        return len(self.data)






class CUBValidation(Dataset):
    def __init__(self, root='/dataset/cub', resolution: Union[Tuple[int, int], int] = 256, resize_ratio: float = 0.75, clip_text_truncate_len: int=30):
        images_path = os.path.join(root, 'images.txt')
        split_path = os.path.join(root, 'train_test_split.txt')
        self.clip_text_truncate_len = clip_text_truncate_len

        self.img_id2path = read_images(images_path)
        self.img_id2split = read_images(split_path)

        self.data = []
        self.labels = []
        self.name2label = {}
        for id, img_path in self.img_id2path.items():
            class_name = img_path.split('/')[0]
            if class_name not in self.name2label.keys():
                self.name2label[class_name] = len(self.name2label)
            if self.img_id2split[id] == '0':
                self.data.append(os.path.join(root, 'images', img_path))
                self.labels.append(root+'/birds/text/'+img_path[:-3]+'txt')

        self.transform = T.Compose([
            T.Resize(resolution),
            T.CenterCrop(resolution),
            lambda x: np.asarray(x),
        ])
            
    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = PIL.Image.open(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        img = (img/127.5 - 1.0).astype(np.float32)
        # with open(label,'r') as f :
        #     lines = f.readlines()
        #     random_label = random.choice(lines).replace('\n', '').lower()

        with open(label,'r') as f:
            lines = f.readlines()
            lines_filter = [i for i in lines if len(i.split(" ")) + 2 < self.clip_text_truncate_len]
            random_label = random.choice(lines_filter).replace('\n', '').lower()


        return {'image': img, 'text': random_label}

    def __len__(self):
        return len(self.data)

