# -*- coding: utf-8 -*-
"""
Model architecture of white-box attack model
"""

import torch
from torch import nn


class WhiteboxModel(nn.Module):
    def __init__(self, num_classes, num_models, target_classifier_grad_size):
        self.num_classes = num_classes
        self.num_models = num_models
        super(WhiteboxModel, self).__init__()
        self.grads_conv = nn.Sequential(
            nn.Conv2d(1, 1000, kernel_size=(1, num_classes), stride=1),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.grads_linear = nn.Sequential(
            nn.Linear(target_classifier_grad_size * 1000, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.labels = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.preds = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.correct = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.loss = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(64 * 4 * self.num_models, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1),
        )
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                nn.init.normal_(self.state_dict()[key], std=0.01)
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0
        self.output = nn.Sigmoid()


    def forward(self, gs, cs, os, los):
        for i in range(self.num_models):
            out_g = self.grads_conv(gs[i]).view([gs[i].size()[0], -1])
            out_g = self.grads_linear(out_g)  # gradient of target model
            out_c = self.correct(cs[i])  # label of target model
            out_o = self.preds(os[i])  # model output of target model
            out_lo = self.loss(los[i])  # loss of target model
            if i == 0:
                com_inp = torch.cat((out_g, out_c, out_o, out_lo), 1)
            else:
                com_inp = torch.cat((out_g, out_c, out_o, out_lo, com_inp), 1)
        is_member = self.combine(com_inp)
        return self.output(is_member)
