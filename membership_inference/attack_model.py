# -*- coding: utf-8 -*-
"""
FCN model architecture for CS-MIA attack model
"""

import torch.nn as nn

__all__ = ['attack_model']

from torch.nn import init


class AttackModel(nn.Module):
    def __init__(self, n_in, n_hidden, direct=False):
        super(AttackModel, self).__init__()
        if direct:
            self.classify = nn.Sequential(
                nn.Flatten(),
                nn.Linear(n_in, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, 1),
                nn.Sigmoid()
            )
        else:
            self.classify = nn.Sequential(
                nn.Linear(n_in, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, 1),
                nn.Sigmoid()
            )
        for layer in self.classify:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight, gain=1)

    def forward(self, x):
        x = self.classify(x)
        return x


def attack_model(**kwargs):
    model = AttackModel(**kwargs)
    return model
