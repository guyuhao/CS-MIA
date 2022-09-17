# -*- coding: utf-8 -*-
"""
Model architecture for ML-Leaks attack model
"""

import torch.nn as nn

__all__ = ['ml_leaks_model']

from torch.nn import init


class MLLeaksModel(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(MLLeaksModel, self).__init__()
        self.classify = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.Linear(n_hidden, 1),
            nn.Sigmoid()
        )
        for layer in self.classify:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight, gain=1)

    def forward(self, x):
        x = self.classify(x)
        return x


def ml_leaks_model(**kwargs):
    model = MLLeaksModel(**kwargs)
    return model
