# -*- coding: utf-8 -*-
"""
evaluate local CS-MIA under DP defense with various epsilon
"""
import logging
import warnings

warnings.filterwarnings('ignore')

import copy
import os
import sys

sys.path.append(os.path.abspath('%s/..' % sys.path[0]))

from common import get_args, init_model, CHECKPOINT_PATH
from federated.federated_learning import federated_train
from membership_inference.local_cs import LocalCS

from membership_inference.attack import get_attack_test_data

os.environ['FLAGS_eager_delete_tensor_gb'] = '0.0'

from split_data import split_data


def del_model(args):
    dir_path = '{}/{}_{}'.format(CHECKPOINT_PATH, args.dataset, args.target_model)
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if 'target'.format(args.target_model, args.n_client) in file:
                os.remove(os.path.join(root, file))

def experiment(args):
    init_global_model = init_model(args)
    train_loader, test_loader = get_attack_test_data(args)
    ep_list = list(range(30, 95, 10))
    for ep in ep_list:
        args.epsilon = ep
        logging.debug('-'*10 + '{} Epsilon'.format(ep) + '-'*10)
        global_model = copy.deepcopy(init_global_model)
        global_models, _ = federated_train(
            global_model=global_model,
            start_epoch=0,
            end_epoch=args.global_epochs,
            args=args,
            dp=True)
        mia_compare(global_models, train_loader, test_loader, args)
    return


def mia_compare(models, train_loader, test_loader, args):
    localCS = LocalCS()
    localCS.get_train_dataset(models, args)
    test_dataset = localCS.get_test_dataset(models, train_loader, test_loader, args)
    acc, pre, recall, f1 = localCS.attack(test_dataset, args)
    logging.debug('CS-MIA epsilon {}: acc {}, pre {}, recall {}, f1 {}'.format(args.epsilon, acc, pre, recall, f1))


if __name__ == '__main__':
    # parse configuration
    args = get_args()
    logging.debug(args)

    if args.split_data:
        split_data(args)
        del_model(args)
    experiment(args)
