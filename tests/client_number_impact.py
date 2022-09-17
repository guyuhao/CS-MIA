# -*- coding: utf-8 -*-
"""
evaluate the impact of the number of clients on local CS-MIA
"""
import logging
import warnings

warnings.filterwarnings('ignore')

import copy
import os
import sys

sys.path.append(os.path.abspath('%s/..' % sys.path[0]))

from common import get_args, init_model
from federated.federated_learning import federated_train
from membership_inference.local_cs import LocalCS

from membership_inference.attack import get_attack_test_data
from split_data import split_data

os.environ['FLAGS_eager_delete_tensor_gb'] = '0.0'

def experiment(args):
    compare_clients = list(range(0, args.n_client+1, 10))
    init_global_model = init_model(args)
    train_loader, test_loader = get_attack_test_data(args)
    for n_client in compare_clients[1:]:
        logging.debug('-'*10 + 'Select All {} Clients'.format(n_client) + '-'*10)
        args.n_client = n_client
        global_model = copy.deepcopy(init_global_model)
        global_models, _ = federated_train(
            global_model=global_model,
            start_epoch=0,
            end_epoch=args.global_epochs,
            args=args)
        mia_attack(global_models, train_loader, test_loader, args)
        del global_models
    return


def mia_attack(models, train_loader, test_loader, args):
    logging.debug('-' * 10 + 'CS-MIA' + '-' * 10 + '\n')
    localCS = LocalCS()
    localCS.get_train_dataset(models, args)
    test_dataset = localCS.get_test_dataset(models, train_loader, test_loader, args)
    acc, pre, recall, f1 = localCS.attack(test_dataset, args)
    logging.debug('CS-MIA client {}: acc {}, pre {}, recall {}, f1 {}'.format(args.n_client, acc, pre, recall, f1))


if __name__ == '__main__':
    # parse configuration
    args = get_args()
    logging.debug(args)

    if args.split_data:
        split_data(args)
    experiment(args)
