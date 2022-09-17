# -*- coding: utf-8 -*-
"""
CNN model architecture for MNIST dataset
"""

import torch.nn as nn
import torch.nn.functional as F


__all__ = ['cnn_mnist']


class CnnMnist(nn.Module):
    def __init__(self, num_classes=100):
        super(CnnMnist, self).__init__()
        input_size = 1
        self.input_size = input_size
        self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=32, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc_features = 4 * 4 * 32
        self.fc1 = nn.Linear(in_features=self.fc_features, out_features=128)
        self.classifier = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, self.fc_features)  # reshape x
        x = F.tanh(self.fc1(x))
        x = self.classifier(x)
        return x


def cnn_mnist(**kwargs):
    model = CnnMnist(**kwargs)
    return model

