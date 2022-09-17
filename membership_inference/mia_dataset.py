# -*- coding: utf-8 -*-
"""
Dataset architecture for attack dataset
"""

import torch.utils.data as data


class MiaDataset(data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        label = self.y[idx]
        inputs = self.x[idx]
        return inputs, label

    def __len__(self):
        return len(self.x)
