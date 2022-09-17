# -*- coding: utf-8 -*-
"""
Preprocess Purchase100 dataset
randomly select 50000 records to build test dataset, and remaining as train dataset
"""

from random import shuffle

DATA_PATH = '../../data/purchase/'
TEST_SIZE = 50000

with open(DATA_PATH+"dataset_purchase", "r") as f:
    dataset = f.readlines()
shuffle(dataset)
test_set = dataset[:TEST_SIZE]
train_set = dataset[TEST_SIZE:]
with open(DATA_PATH+"dataset_purchase_train", "w") as f:
    f.writelines(train_set)
with open(DATA_PATH+"dataset_purchase_test", "w") as f:
    f.writelines(test_set)
