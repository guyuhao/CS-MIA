# -*- coding: utf-8 -*-
"""
evaluate active global attack without defense
"""
import logging
import warnings

warnings.filterwarnings('ignore')

import copy
import os
import sys

sys.path.append(os.path.abspath('%s/..' % sys.path[0]))

import numpy as np

from common import get_args, init_model, DATA_PATH, CHECKPOINT_PATH
from federated.federated_learning import federated_train

import numpy

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


def attack_experiment(init_global_model, train_loader, test_loader, args, type, target_clients_dict=dict()):
    global_model = copy.deepcopy(init_global_model)
    select_client = False
    if type != 'all':
        select_client = True
        load = False
        if args.load_select:
            file = DATA_PATH + '{}/clients_indices_{}_{}_{}_{}.npz'.format(args.dataset, args.dataset, args.n_client,
                                                             args.n_selected_client, type)
            if os.path.isfile(file):
                data = np.load(file, allow_pickle=True)
                target_clients_dict = data['dict'][()]
                load = True
        if not load:
            for epoch in range(0, args.global_epochs, 1):
                target_clients = np.arange(args.n_client)
                if epoch != args.global_epochs - 1:
                    target_clients = np.random.choice(target_clients, args.n_selected_client, replace=False)
                else:
                    target_clients = np.delete(target_clients, [args.target_client])
                    target_clients = np.random.choice(target_clients, args.n_selected_client - 1, replace=False)
                    target_clients = numpy.append(target_clients, [args.target_client])
                target_clients_dict[epoch] = target_clients
    _, _, server_models, target_client_models = federated_train(
        global_model=global_model,
        start_epoch=0,
        end_epoch=args.global_epochs,
        args=args,
        target_clients_dict=target_clients_dict,
        select_client=select_client,
        server_positive_attack=True)
    mia_compare(server_models, target_client_models, train_loader, test_loader, args)

    if type != 'all':
        if args.save_select:
            print('save {}: {}'.format(type, target_clients_dict))
            np.savez(DATA_PATH + '{}/clients_indices_{}_{}_{}_{}.npz'.
                     format(args.dataset, args.dataset, args.n_client, args.n_selected_client, type), dict=target_clients_dict)
    return target_clients_dict


def experiment(args):
    if args.save_select:
        del_target(args)
    if args.save_select:
        file = DATA_PATH + '{}/clients_indices_{}_{}_{}_yes.npz'.format(args.dataset, args.dataset, args.n_client, args.n_selected_client)
        if os.path.isfile(file):
            os.remove(file)
        file = DATA_PATH + '{}/clients_indices_{}_{}_{}_no.npz'.format(args.dataset, args.dataset, args.n_client, args.n_selected_client)
        if os.path.isfile(file):
            os.remove(file)
    init_global_model = init_model(args)
    train_loader, test_loader = get_attack_test_data(args)
    # evaluate in complete aggregation scenario
    logging.debug('-'*10 + 'Select All Clients' + '-'*10)
    attack_experiment(init_global_model, train_loader, test_loader, args, 'all')
    # evaluate in fractional aggregation scenario
    logging.debug('-' * 10 + 'Select Random Clients, Yes' + '-' * 10)
    attack_experiment(init_global_model, train_loader, test_loader, args, 'yes')
    return


def mia_compare(server_models, target_client_models, train_loader, test_loader, args):
    # active global CS-MIA
    logging.debug('-' * 10 + 'active global CS-MIA' + '-' * 10 + '\n')
    globalCS = GlobalCS()
    globalCS.get_train_dataset(server_models, args)
    del server_models
    test_dataset = globalCS.get_test_dataset(target_client_models, train_loader, test_loader, args)
    acc, pre, recall, f1 = globalCS.attack(test_dataset, args)
    logging.debug('active global CS-MIA: acc {}, pre {}, recall {}, f1 {}'.format(acc, pre, recall, f1))


if __name__ == '__main__':
    # parse configuration
    args = get_args()
    logging.debug(args)

    if args.split_data:
        split_data(args)
    experiment(args)
