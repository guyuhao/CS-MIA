# -*- coding: utf-8 -*-
"""
evaluate the impact of the number of selected clients on active and passive global CS-MIA
"""
import logging
import warnings
import random

import numpy as np

warnings.filterwarnings('ignore')

import copy
import os
import sys

sys.path.append(os.path.abspath('%s/..' % sys.path[0]))

from common import get_args, init_model, CHECKPOINT_PATH
from federated.federated_learning import federated_train

from membership_inference.attack import get_attack_test_data
from membership_inference.global_cs import GlobalCS
from split_data import split_data

os.environ['FLAGS_eager_delete_tensor_gb'] = '0.0'


def del_target(args):
    dir_path = '{}/{}_{}'.format(CHECKPOINT_PATH, args.dataset, args.target_model)
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if 'target_select_{}_{}client'.format(args.target_model, args.n_client) in file:
                os.remove(os.path.join(root, file))


def experiment(args):
    init_global_model = init_model(args)
    train_loader, test_loader = get_attack_test_data(args)
    selected_radios = np.arange(0, 1.1, 0.1)
    for radio in selected_radios:
        selected_num = max(1, int(args.n_client*radio))
        args.n_selected_client = selected_num
        global_model = copy.deepcopy(init_global_model)
        logging.debug('-' * 10 + 'Select {} Clients active'.format(selected_num) + '-' * 10)
        _, temp_dict, server_models, target_client_models = federated_train(
            global_model=global_model,
            start_epoch=0,
            end_epoch=args.global_epochs,
            args=args,
            select_client=True,
            server_positive_attack=True)
        logging.debug(temp_dict)
        mia_compare(server_models, target_client_models, train_loader, test_loader, args, 'active')

        logging.debug('-' * 10 + 'Select {} Clients passive'.format(selected_num) + '-' * 10)
        global_model = copy.deepcopy(init_global_model)
        fake_global_model = copy.deepcopy(init_global_model)

        _, temp_dict, server_models, target_client_models = federated_train(
            global_model=global_model,
            start_epoch=0,
            end_epoch=args.global_epochs,
            args=args,
            select_client=True,
            server_passive_attack=True,
            fake_global_model=fake_global_model)
        logging.debug(temp_dict)
        mia_compare(server_models, target_client_models, train_loader, test_loader, args, 'passive')
    return


def mia_compare(server_models, target_client_models, train_loader, test_loader, args, type):
    logging.debug('-' * 10 + 'global CS-MIA' + '-' * 10 + '\n')
    if type == 'passive':
        indices = [i for i in range(0, len(target_client_models)) if target_client_models[i] is not None]
        server_models = [server_models[i] for i in indices]
        target_client_models = [target_client_models[i] for i in indices]
    if len(target_client_models) == 0:
        logging.debug(
            'CS-MIA {} attack {} client: acc 0.0, pre 0.0, recall 0.0, f1 0.0'.format(type, args.n_selected_client))
        return
    globalCS = GlobalCS()
    globalCS.get_train_dataset(server_models, args)
    test_dataset = globalCS.get_test_dataset(target_client_models, train_loader, test_loader, args)
    acc, pre, recall, f1 = globalCS.attack(test_dataset, args)
    logging.debug('CS-MIA {} attack {} client: acc {}, pre {}, recall {}, f1 {}'.format(type, args.n_selected_client, acc, pre, recall, f1))


if __name__ == '__main__':
    # parse configuration
    args = get_args()
    logging.debug(args)

    if args.split_data:
        split_data(args)
    experiment(args)
