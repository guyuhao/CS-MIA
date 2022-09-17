# -*- coding: utf-8 -*-
"""
Define tabular dataset, used by Purchase100 and Texas100
"""

import torch.utils.data as data
import torch


class TextDataset(data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        label = torch.tensor(self.y[idx])
        inputs = torch.tensor(self.x[idx])
        return inputs, label

    def __len__(self):
        return len(self.x)
