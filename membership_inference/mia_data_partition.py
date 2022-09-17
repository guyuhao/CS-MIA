# -*- coding: utf-8 -*-
"""
divide dataset into two sub-datasets for target and attack model
"""

import numpy as np

from common import get_dataset, DATA_PATH, TARGET_INDICES_FILE, SHADOW_INDICES_FILE


def get_data_indices(train_size, test_size,
                     target_train_size=int(1e4), target_test_size=int(1e4),
                     attack_train_size=int(1e4), attack_test_size=int(1e4)):
    """
    divide the training and testing dataset for the target and shadow model

    :param train_size: total size of train dataset
    :param test_size: total size of test dataset
    :param target_train_size: size of train dataset for clients
    :param target_test_size: target size of test dataset
    :param attack_train_size: size of train members data
    :param attack_test_size: size of train non-members data
    :return: tuple containing: target_train_indices, target_test_indices, shadow_train_indices, shadow_test_indices
    """
    # divide the training dataset for the target and shadow model
    train_indices = np.arange(train_size)
    target_train_indices = np.random.choice(train_indices, target_train_size, replace=False)
    shadow_train_indices = np.setdiff1d(train_indices, target_train_indices)
    if len(shadow_train_indices) < attack_train_size:
        temp_indices = np.random.choice(target_train_indices, attack_train_size-len(shadow_train_indices), replace=False)
        shadow_train_indices = np.concatenate([shadow_train_indices, temp_indices])
    else:
        shadow_train_indices = shadow_train_indices[:attack_train_size]
    # divide the testing dataset for the target and shadow model
    test_indices = np.arange(test_size)
    target_test_indices = np.random.choice(test_indices, target_test_size, replace=False)
    shadow_test_indices = np.setdiff1d(test_indices, target_test_indices)
    if len(shadow_test_indices) < attack_test_size:
        temp_indices = np.random.choice(target_test_indices, attack_test_size - len(shadow_test_indices),
                                        replace=False)
        shadow_test_indices = np.concatenate([shadow_test_indices, temp_indices])
    else:
        shadow_test_indices = shadow_test_indices[:attack_test_size]
    return target_train_indices, target_test_indices, shadow_train_indices, shadow_test_indices


def data_partition(args):
    """
    divide the training and testing for the target and shadow model

    :param args: configuration
    :return: indices of train dataset for clients
    """
    train_dataset, test_dataset = get_dataset(args.dataset)

    target_train_size = args.target_train_size

    target_train_indices, target_test_indices, shadow_train_indices, shadow_test_indices = get_data_indices(
        train_size=len(train_dataset),
        test_size=len(test_dataset),
        target_train_size=target_train_size,
        target_test_size=args.target_test_size,
        attack_train_size=args.attack_train_m_size,
        attack_test_size=args.attack_train_nm_size)

    non_iid_prefix = '_non_iid' if args.non_iid else ''
    # save indices of training and testing dataset for the shadow model
    np.savez(DATA_PATH + args.dataset + '/' + SHADOW_INDICES_FILE +
             '{}_{}_{}_{}.npz'.format(non_iid_prefix, args.dataset, args.attack_train_m_size, len(shadow_test_indices)),
             train_indices=shadow_train_indices, test_indices=shadow_test_indices)

    # save indices of training and testing dataset for the target model
    np.savez(DATA_PATH + args.dataset + '/' + TARGET_INDICES_FILE +
             '{}_{}_{}_{}.npz'.format(non_iid_prefix, args.dataset, args.target_train_size, len(target_test_indices)),
             train_indices=target_train_indices, test_indices=target_test_indices)

    return target_train_indices
