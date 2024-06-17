# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import PIL
from typing import Any, Tuple, Union, Optional, Callable
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torchvision import transforms as T
from torchvision.datasets import ImageNet
from torch.utils.data import Dataset
from torchvision.datasets.folder import ImageFolder
import os
import numpy as np
import pickle
import random


def is_truncated(filepath):
    try:
        PIL.Image.open(filepath)
    except:
        return True
    return False


def read_images(path):
    f = open(path)
    lines = f.readlines()
    img_id2path = {}
    for line in lines:
        id, img_path = line.strip('\n').split(' ')
        img_id2path[id] = img_path
    f.close()
    return img_id2path


class CelehqTrain(Dataset):
    def __init__(self, root='celehq', resolution: Union[Tuple[int, int], int] = 256, resize_ratio: float = 0.75,
                 clip_text_truncate_len: int = 30):
        with open(os.path.join(root, 'train', 'filenames.pickle'), 'rb') as handle:
            images = pickle.load(handle)
        self.root = root
        self.data = []
        self.text = []
        self.n_labels = 19

        self.clip_text_truncate_len = clip_text_truncate_len

        for image in images:
            self.data.append(os.path.join(self.root, 'images', image + '.jpg'))  # 加载图片
            self.text.append(os.path.join(self.root, 'celeba-caption', image + '.txt'))

            # self.segment.append(os.path.join(self.root, 'CelebAMaskHQ-mask', image+'.png'))  # 加载分割图像
        self.transform = T.Compose([
            T.Resize(resolution),
            # T.CenterCrop(resolution),
            # T.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
        ])

    def __getitem__(self, index):
        img, text = self.data[index], self.text[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = PIL.Image.open(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        img = (img / 127.5 - 1.0).astype(np.float32)

        with open(text, 'r') as f:
            lines = f.readlines()
            lines_filter = [i for i in lines if len(i.split(" ")) + 2 < self.clip_text_truncate_len]
            random_label = random.choice(lines_filter).replace('\n', '').lower()

        #
        # seg = PIL.Image.open(seg)
        # if self.transform is not None:
        #     seg = self.transform(seg)
        # seg = seg.astype(np.uint8)[:,:]
        # seg = np.eye(self.n_labels)[seg]

        return {'image': img, 'text': random_label}
        # return {'image': img, 'class': label}

    def __len__(self):
        return len(self.data)


class CelehqValidation(Dataset):
    def __init__(self, root='celehq', resolution: Union[Tuple[int, int], int] = 256, resize_ratio: float = 0.75,
                 clip_text_truncate_len: int = 30):
        with open(os.path.join(root, 'test', 'filenames.pickle'), 'rb') as handle:
            images = pickle.load(handle)
        self.root = root
        self.data = []
        self.text = []
        self.n_labels = 19

        self.clip_text_truncate_len = clip_text_truncate_len

        for image in images:
            self.data.append(os.path.join(self.root, 'images', image + '.jpg'))
            self.text.append(os.path.join(self.root, 'celeba-caption', image + '.txt'))

        self.transform = T.Compose([
            T.Resize(resolution),
            # T.CenterCrop(resolution),
            # T.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
        ])
        # print(len(self.labels))

    def __getitem__(self, index):
        img, text = self.data[index], self.text[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = PIL.Image.open(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        img = (img / 127.5 - 1.0).astype(np.float32)

        with open(text, 'r') as f:
            lines = f.readlines()
            lines_filter = [i for i in lines if len(i.split(" ")) + 2 < self.clip_text_truncate_len]
            random_label = random.choice(lines_filter).replace('\n', '').lower()

        # seg = PIL.Image.open(seg)
        # if self.transform is not None:
        #     seg = self.transform(seg)
        # seg = seg.astype(np.uint8)
        # seg = np.eye(self.n_labels)[seg]

        return {'image': img, 'text': random_label}
        # return {'image': img, 'class': label}

    def __len__(self):
        return len(self.data)
