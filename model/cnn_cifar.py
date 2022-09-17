# -*- coding: utf-8 -*-
"""
CNN model architecture for CIFAR dataset
"""

import torch.nn as nn
import torch.nn.functional as F


__all__ = ['cnn_cifar']


class CnnCifar(nn.Module):
    def __init__(self, num_classes=100):
        super(CnnCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.classifier = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.classifier(x)
        return x


def cnn_cifar(**kwargs):
    model = CnnCifar(**kwargs)
    return model

