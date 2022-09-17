# -*- coding: utf-8 -*-
"""
FCN model architecture for Texas100 dataset
"""

import torch.nn as nn


__all__ = ['fcn_texas']


class FcnTexas(nn.Module):

    def __init__(self, num_classes=100):
        super(FcnTexas, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(6169, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh()
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def fcn_texas(**kwargs):
    model = FcnTexas(**kwargs)
    return model
