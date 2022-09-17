# -*- coding: utf-8 -*-
"""
evaluate the impact of non-IID data on local CS-MIA
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
from membership_inference.local_cs import LocalCS

import numpy

from membership_inference.ml_leaks import MLLeaks
from membership_inference.threshold_attack import ThresholdAttack
from membership_inference.attack import get_attack_test_data
from membership_inference.whitebox_attack import WhiteboxAttack

os.environ['FLAGS_eager_delete_tensor_gb'] = '0.0'


def train_target_model(global_model, start_epoch, end_epoch, args, select_client=False, target_clients_dict=dict()):
    global_model_list, _ = federated_train(
        global_model=global_model,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        args=args,
        target_clients_dict=target_clients_dict,
        select_client=select_client)
    model = global_model_list[-1]
    return model


def del_target(args):
    dir_path = '{}/{}_{}'.format(CHECKPOINT_PATH, args.dataset, args.target_model)
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if 'target_select_{}client'.format(args.n_client) in file:
                os.remove(os.path.join(root, file))


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
    global_models = list()
    # evaluate in complete aggregation scenario
    logging.debug('-'*10 + 'Select All Clients' + '-'*10)
    global_model = copy.deepcopy(init_global_model)
    for epoch in range(0, args.global_epochs, 1):
        global_model = train_target_model(
            global_model=global_model,
            start_epoch=epoch,
            end_epoch=epoch + 1,
            args=args,
            select_client=False)
        global_models.append(copy.deepcopy(global_model))
    mia_compare(global_models, train_loader, test_loader, args, 'all')
    # evaluate in fractional aggregation scenario: 3/5(Y)
    logging.debug('-' * 10 + 'Select Random Clients, Yes' + '-' * 10)
    global_model = copy.deepcopy(init_global_model)
    global_models.clear()
    target_clients_dict = dict()
    if args.load_select:
        file = DATA_PATH + '{}/clients_indices_{}_{}_{}_yes.npz'.format(args.dataset, args.dataset, args.n_client, args.n_selected_client)
        if os.path.isfile(file):
            data = np.load(file, allow_pickle=True)
            target_clients_dict = data['dict'][()]
            print(target_clients_dict)
    for epoch in range(0, args.global_epochs, 1):
        if epoch not in target_clients_dict.keys():
            target_clients = np.arange(args.n_client)
            if epoch != args.global_epochs-1:
                target_clients = np.random.choice(target_clients, args.n_selected_client, replace=False)
            else:
                target_clients = np.delete(target_clients, [args.target_client])
                target_clients = np.random.choice(target_clients, args.n_selected_client-1, replace=False)
                target_clients = numpy.append(target_clients, [args.target_client])
            target_clients_dict[epoch] = target_clients
        global_model = train_target_model(
            global_model=global_model,
            start_epoch=epoch,
            end_epoch=epoch + 1,
            target_clients_dict=target_clients_dict,
            args=args,
            select_client=True)
        global_models.append(copy.deepcopy(global_model))
    mia_compare(global_models, train_loader, test_loader, args, 'yes')
    if args.save_select:
        print('save yes {}'.format(target_clients_dict))
        np.savez(DATA_PATH + '{}/clients_indices_{}_{}_{}_yes.npz'.
                 format(args.dataset, args.dataset, args.n_client, args.n_selected_client), dict=target_clients_dict)
    # evaluate in fractional aggregation scenario: 3/5(N)
    logging.debug('-' * 10 + 'Select Random Clients, No' + '-' * 10)
    global_model = copy.deepcopy(init_global_model)
    global_models.clear()
    del target_clients_dict[args.global_epochs-1]
    if args.load_select:
        file = DATA_PATH + '{}/clients_indices_{}_{}_{}_no.npz'.format(args.dataset, args.dataset, args.n_client, args.n_selected_client)
        if os.path.isfile(file):
            data = np.load(file, allow_pickle=True)
            target_clients_dict = data['dict'][()]
            print(target_clients_dict)
    for epoch in range(0, args.global_epochs, 1):
        if epoch not in target_clients_dict.keys():
            target_clients = np.arange(args.n_client)
            if epoch != args.global_epochs - 1:
                target_clients = np.random.choice(target_clients, args.n_selected_client, replace=False)
            else:
                target_clients = np.delete(target_clients, [args.target_client])
                target_clients = np.random.choice(target_clients, args.n_selected_client, replace=False)
            target_clients_dict[epoch] = target_clients
        global_model = train_target_model(
            global_model=global_model,
            start_epoch=epoch,
            end_epoch=epoch + 1,
            target_clients_dict=target_clients_dict,
            args=args,
            select_client=True)
        global_models.append(copy.deepcopy(global_model))
    mia_compare(global_models, train_loader, test_loader, args, 'no')
    if args.save_select:
        print('save no {}'.format(target_clients_dict))
        np.savez(DATA_PATH + '{}/clients_indices_{}_{}_{}_no.npz'.
                 format(args.dataset, args.dataset, args.n_client, args.n_selected_client), dict=target_clients_dict)
    return


# 比较不同mia攻击的效果
def mia_compare(models, train_loader, test_loader, args, type):
    model = models[-1]
    # threshold
    logging.debug('-' * 10 + 'Threshold MIA' + '-' * 10 + '\n')
    threshold_attack = ThresholdAttack()
    train_dataset = threshold_attack.get_train_dataset(model, args)
    test_dataset = threshold_attack.get_test_dataset(model, train_loader, test_loader, args)
    acc, pre, recall, f1 = threshold_attack.attack(train_dataset, test_dataset, args)
    logging.debug('threshold mia: acc {}, pre {}, recall {}, f1 {}'.format(acc, pre, recall, f1))
    # ML-Leaks
    logging.debug('-' * 10 + 'ML Leaks' + '-' * 10 + '\n')
    mlLeaks = MLLeaks()
    mlLeaks.train_shadow_models(model, args)
    attack_test_x, attack_test_y = mlLeaks.get_test_data(
        target_model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        args=args)
    dataset = (attack_test_x, attack_test_y)
    acc, pre, recall, f1 = mlLeaks.attack(dataset, args)
    logging.debug('ml leaks: acc {}, pre {}, recall {}, f1 {}'.format(acc, pre, recall, f1))
    # whitebox
    logging.debug('-' * 10 + 'Whitebox' + '-' * 10 + '\n')
    whitebox_attack = WhiteboxAttack(args)
    acc, pre, recall, f1 = whitebox_attack.attack(models, train_loader, test_loader)
    logging.debug('whitebox: acc {}, pre {}, recall {}, f1 {}'.format(acc, pre, recall, f1))
    # CS-MIA
    logging.debug('-' * 10 + 'CS-MIA' + '-' * 10 + '\n')
    localCS = LocalCS()
    localCS.get_train_dataset(models, args)
    test_dataset = localCS.get_test_dataset(models, train_loader, test_loader, args)
    acc, pre, recall, f1 = localCS.attack(test_dataset, args)
    logging.debug('CS-MIA: acc {}, pre {}, recall {}, f1 {}'.format(acc, pre, recall, f1))


if __name__ == '__main__':
    # parse configuration
    args = get_args()
    logging.debug(args)

    experiment(args)
