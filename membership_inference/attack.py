# -*- coding: utf-8 -*-
import numpy as np
import torch

from common import load_data, TARGET_INDICES_FILE


def get_attack_test_data(args):
    """
    load attack test dataset

    :param args: configuration
    :return: loader of test members data and non-members data
    """
    # load test members
    non_iid_prefix = '_non_iid' if args.non_iid else ''
    indices_file = 'client{}_remove_indices{}_{}_{}_{}.npz'.format(
        args.target_client, non_iid_prefix, args.dataset, args.client_train_size, args.attack_test_m_size)
    train_loader, _ = load_data(args.dataset,
                                indices_file,
                                args.whitebox_batch_size,  # whitebox攻击对测试集的batch有要求
                                'target')

    # load test no-members
    indices_file = TARGET_INDICES_FILE + '{}_{}_{}_{}.npz'.format(
        non_iid_prefix, args.dataset, args.target_train_size, args.target_test_size)
    _, test_loader = load_data(args.dataset,
                               indices_file,
                               args.whitebox_batch_size,  # whitebox攻击对测试集的batch有要求
                               'train-test')
    test_dataset = test_loader.dataset
    if len(test_dataset) > args.attack_test_nm_size:
        test_indices = np.random.choice(np.arange(len(test_dataset)), args.attack_test_nm_size, replace=False)
        sub_test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
        test_loader = torch.utils.data.DataLoader(sub_test_dataset, batch_size=args.whitebox_batch_size, shuffle=True)

    return train_loader, test_loader
