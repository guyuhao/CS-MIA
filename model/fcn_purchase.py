# -*- coding: utf-8 -*-
"""
FCN model architecture for Purchase100 dataset
"""

import torch.nn as nn
import torch.nn.functional as F

__all__ = ['fcn_purchase']


class FcnPurchase(nn.Module):

    def __init__(self, num_classes=100, dropout=False, p=None, adv_reg=False):
        super(FcnPurchase, self).__init__()
        self.dropout = dropout
        self.dropout_p = p
        self.adv_reg = adv_reg
        if not dropout:
            self.features = nn.Sequential(
                nn.Linear(600, 1024),
                nn.Tanh(),
                nn.Linear(1024, 512),
                nn.Tanh(),
                nn.Linear(512, 256),
                nn.Tanh(),
                nn.Linear(256, 128),
                nn.Tanh()
            )
        else:
            self.features = nn.Sequential(
                nn.Linear(600, 1024),
                nn.Tanh(),
                nn.Dropout(self.dropout_p),
                nn.Linear(1024, 512),
                nn.Tanh(),
                nn.Dropout(self.dropout_p),
                nn.Linear(512, 256),
                nn.Tanh(),
                nn.Dropout(self.dropout_p),
                nn.Linear(256, 128),
                nn.Tanh(),
                nn.Dropout(self.dropout_p)
            )
        self.classifier = nn.Linear(128, num_classes)
        # self.classifier = nn.Sequential(
        #     nn.Linear(600, 128),
        #     nn.Tanh(),
        #     nn.Linear(128, num_classes)
        # )
        # for layer in self.classifier:
        #     if isinstance(layer, nn.Linear):  # 判断是否是线性层
        #         init.xavier_uniform_(layer.weight, gain=1)

    def forward(self, x):
        x = self.features(x)
        if self.adv_reg:
            # for adversarial regulation defense, return prediction vector and extracted features for membership inference
            return self.classifier(x), x
        else:
            x = self.classifier(x)
            return x


def fcn_purchase(**kwargs):
    model = FcnPurchase(**kwargs)
    return model
