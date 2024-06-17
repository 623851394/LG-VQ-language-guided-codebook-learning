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


# class ImageNetBase(ImageNet):
#     def __init__(self, root: str, split: str,
#                  transform: Optional[Callable] = None) -> None:
#         super().__init__(root=root, split=split, transform=transform)
        
#     def __getitem__(self, index: int) -> Tuple[Any, Any]:
#         sample, target = super().__getitem__(index)

#         return {'image': sample, 'class': torch.tensor([target])}

class ImageNetTrain(Dataset):
    def __init__(self, config=None):
        root = '/workplace/dataset/UCML2/'
        resolution = config.size
        root = os.path.join(root, 'train')
        self.data = []
        self.labels = []
        self.name2label = {}
        for class_name in os.listdir(root):
            if class_name not in self.name2label.keys():
                self.name2label[class_name] = len(self.name2label)
            for file in os.listdir(os.path.join(root, class_name)):
                self.data.append(os.path.join(root, class_name, file))
                self.labels.append(self.name2label[class_name])

        self.transform = T.Compose([
            T.Resize(resolution),
            T.RandomCrop(resolution),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])
            
    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = PIL.Image.open(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return {'image': img, 'class': torch.tensor([label])}

    def __len__(self):
        return len(self.data)

class ImageNetValidation(Dataset):
    def __init__(self, config=None):
        root = '/workplace/dataset/UCML2/'
        resolution = config.size
        root = os.path.join(root, 'val')
        self.data = []
        self.labels = []
        self.name2label = {}
        for class_name in os.listdir(root):
            if class_name not in self.name2label.keys():
                self.name2label[class_name] = len(self.name2label)
            for file in os.listdir(os.path.join(root, class_name)):
                self.data.append(os.path.join(root, class_name, file))
                self.labels.append(self.name2label[class_name])

        self.transform = T.Compose([
            T.Resize(resolution),
            T.RandomCrop(resolution),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])
            
    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = PIL.Image.open(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return  {'image': img, 'class': torch.tensor([label])}

    def __len__(self):
        return len(self.data)

