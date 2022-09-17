# -*- coding: utf-8 -*-
"""
Preprocess Texas100 dataset
randomly select 10000 records to build test dataset, and remaining as train dataset
"""

import numpy as np

DATA_PATH = '../../data/texas100/'
TEST_SIZE = 10000

with open(DATA_PATH+"feats", "r") as f:
    inputset = f.readlines()
with open(DATA_PATH+"labels", "r") as f:
    labelset = f.readlines()
inputset = np.array(inputset)
labelset = np.array(labelset)
all_indices = np.arange(len(inputset))
test_indices = np.random.choice(all_indices, TEST_SIZE, replace=False)
train_indices = np.setdiff1d(all_indices, test_indices)
test_input_set = inputset[test_indices].tolist()
train_input_set = inputset[train_indices].tolist()
test_label_set = labelset[test_indices].tolist()
train_label_set = labelset[train_indices].tolist()
with open(DATA_PATH+"feats_train", "w") as f:
    f.writelines(train_input_set)
with open(DATA_PATH+"labels_train", "w") as f:
    f.writelines(train_label_set)
with open(DATA_PATH+"feats_test", "w") as f:
    f.writelines(test_input_set)
with open(DATA_PATH+"labels_test", "w") as f:
    f.writelines(test_label_set)
