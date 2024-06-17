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
import json


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


class COCOTrain(Dataset):
    def __init__(self, root: str, resolution: Union[Tuple[int, int], int] = 256, resize_ratio: float = 0.75,):
        # self.images = os.listdir(os.path.join(root,'train2014'))
        self.root = os.path.join(root, 'train2014')
        self.data = []

        self.text = []
        annotation_path = os.path.join(root, 'annotations', 'captions_train2014.json')
        imags_caption = json.load(open(annotation_path))


        # 需要保持文本信息和图片信息都有
        for info in imags_caption['annotations']:
            image_path = os.path.join(self.root, 'COCO_train2014_000000' + str(info['image_id']) + '.jpg')
            if (is_truncated(image_path)):
                continue
            else:
                self.data.append(image_path)
                self.text.append(info['caption'].replace('\n', '').lower())

        self.transform = T.Compose([
            T.Resize(resolution),
            T.CenterCrop(resolution),
            T.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
        ])
        # print(len(self.labels))

    def __getitem__(self, index):
        img = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = PIL.Image.open(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        img = (img / 127.5 - 1.0).astype(np.float32)

        text = self.text[index]

        return {'image': img, 'text': text}

    def __len__(self):
        return len(self.data)


class COCOValidation(Dataset):
    def __init__(self, root: str, resolution: Union[Tuple[int, int], int] = 256, resize_ratio: float = 0.75):
        self.images = os.listdir(os.path.join(root, 'val2014'))
        self.root = os.path.join(root, 'val2014')
        self.data = []
        for image in self.images:
            if (is_truncated(os.path.join(self.root, image))):
                continue
            else:
                self.data.append(os.path.join(self.root, image))
        self.transform = T.Compose([
            T.Resize(resolution),
            T.CenterCrop(resolution),
            T.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
        ])

    def __getitem__(self, index):
        img = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = PIL.Image.open(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        img = (img / 127.5 - 1.0).astype(np.float32)


        return {'image': img}

    def __len__(self):
        return len(self.data)


class COCOTest(Dataset):
    def __init__(self, root='./dataset/COCO/coco2014/test2014', resolution: Union[Tuple[int, int], int] = 256,
                 resize_ratio: float = 0.75):
        self.images = os.listdir(root)
        self.data = []
        for image in self.images:
            if (is_truncated(os.path.join(root, image))):
                continue
            else:
                self.data.append(os.path.join(root, image))
        self.transform = T.Compose([
            T.Resize(resolution),
            T.CenterCrop(resolution),
            T.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
        ])
        # print(len(self.labels))

    def __getitem__(self, index):
        img = self.data[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = PIL.Image.open(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        img = (img / 127.5 - 1.0).astype(np.float32)
        return {'image': img}
        # return {'image': img, 'class': label}

    def __len__(self):
        return len(self.data)