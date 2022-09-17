# -*- coding: utf-8 -*-
"""
evaluate the impact of confidence metric on passive global CS-MIA
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
from membership_inference.attack import get_attack_test_data
from split_data import split_data

from membership_inference.global_cs import GlobalCS

os.environ['FLAGS_eager_delete_tensor_gb'] = '0.0'


def experiment(args):
    init_global_model = init_model(args)
    train_loader, test_loader = get_attack_test_data(args)
    global_model = copy.deepcopy(init_global_model)
    fake_global_model = copy.deepcopy(init_global_model)
    _, _, server_models, target_client_models = federated_train(
        global_model=global_model,
        start_epoch=0,
        end_epoch=args.global_epochs,
        args=args,
        server_passive_attack=True,
        fake_global_model=fake_global_model)
    client_models = []
    for temp in target_client_models:
        if temp is not None:
            client_models.append(temp['model'])
        else:
            client_models.append(None)
    # evaluate metric: Mentr
    logging.debug('-' * 10 + 'Mentr Attack' + '-' * 10)
    mia_attack(server_models, client_models, train_loader, test_loader, args, 'mentr')
    # evaluate metric: Pr
    logging.debug('-' * 10 + 'Mentr Attack' + '-' * 10)
    mia_attack(server_models, client_models, train_loader, test_loader, args, 'conf')
    return


def mia_attack(server_models, target_client_models, train_loader, test_loader, args, assurance_type):
    indices = [i for i in range(0, len(target_client_models)) if target_client_models[i] is not None]
    server_models = [server_models[i] for i in indices]
    target_client_models = [target_client_models[i] for i in indices]
    globalCS = GlobalCS(assurance_type)
    globalCS.get_train_dataset(server_models, args)
    test_dataset = globalCS.get_test_dataset(target_client_models, train_loader, test_loader, args)
    acc, pre, recall, f1 = globalCS.attack(test_dataset, args)
    logging.debug('global CS-MIA metric {}: acc {}, pre {}, recall {}, f1 {}'.format(assurance_type, acc, pre, recall, f1))

if __name__ == '__main__':
    # parse configuration
    args = get_args()
    logging.debug(args)

    if args.split_data:
        split_data(args)
    experiment(args)
