# -*- coding: utf-8 -*-
"""
split dataset into train and shadow sub-datasets, and divide train datasets for all clients
"""
import logging
import os

import federated.federated_data_partition as fdp
import membership_inference.mia_data_partition as mdp
from common import get_args


def split_data(args):
    if args.non_iid == 0:
        target_train_indices = mdp.data_partition(args)
        fdp.data_partition(target_train_indices, args)
    else:
        target_train_indices = mdp.data_partition(args)
        fdp.non_iid_data_partition(target_train_indices, args)


if __name__ == '__main__':
    # parse configuration
    args = get_args()
    logging.debug(args)

    split_data(args)
