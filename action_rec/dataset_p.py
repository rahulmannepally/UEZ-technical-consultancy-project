#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 05:10:20 2021

@author: rahulm
"""

import numpy as np
import os
import pickle
import torch

from torch.utils.data import Dataset, DataLoader

CATEGORY_INDEX = {
    "walking":0,
    "handwaving": 1
}

class RawDataset(Dataset):
    def __init__(self, directory, dataset="train"):
        self.instances, self.labels = self.read_dataset(directory, dataset)

        self.instances = torch.from_numpy(self.instances)
        self.labels = torch.from_numpy(self.labels)

    def __len__(self):
        return self.instances.shape[0]

    def __getitem__(self, idx):
        sample = { 
            "instance": self.instances[idx],
            "label": self.labels[idx] 
        }

        return sample

    def zero_center(self, mean):
        self.instances -= float(mean)

    def read_dataset(self, directory, dataset="train"):
        if dataset == "train":
            filepath = os.path.join(directory, "train.p")
        elif dataset == "dev":
            filepath = os.path.join(directory, "dev.p")
        else:
            filepath = os.path.join(directory, "test.p")

        videos = pickle.load(open(filepath, "rb"))

        instances = []
        labels = []
        for video in videos:
            for frame in video["frames"]:
                instances.append(frame.reshape((1, 60, 80)))
                labels.append(CATEGORY_INDEX[video["category"]])

        instances = np.array(instances, dtype=np.float32)
        labels = np.array(labels, dtype=np.uint8)

        self.mean = np.mean(instances)

        return instances, labels