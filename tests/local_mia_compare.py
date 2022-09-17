# -*- coding: utf-8 -*-
"""
compare different local attacks without defense
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
from split_data import split_data
from membership_inference.blindmi_diff_w import BlindMI_Diff_W


os.environ['FLAGS_eager_delete_tensor_gb'] = '0.0'


def del_target(args):
    dir_path = '{}/{}_{}'.format(CHECKPOINT_PATH, args.dataset, args.target_model)
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if 'target_select_{}_{}client'.format(args.target_model, args.n_client) in file:
                os.remove(os.path.join(root, file))


def attack_experiment(init_global_model, train_loader, test_loader, type, args, target_clients_dict=dict()):
    global_model = copy.deepcopy(init_global_model)
    select_client = False
    if type != 'all':
        select_client = True
        load = False
        if args.load_select:
            file = DATA_PATH + '{}/clients_indices_{}_{}_{}_{}.npz'.format(args.dataset, args.dataset, args.n_client, args.n_selected_client, type)
            if os.path.isfile(file):
                data = np.load(file, allow_pickle=True)
                target_clients_dict = data['dict'][()]
                load = True
        if not load:
            if type == 'yes':
                for epoch in range(0, args.global_epochs, 1):
                    target_clients = np.arange(args.n_client)
                    if epoch != args.global_epochs - 1:
                        target_clients = np.random.choice(target_clients, args.n_selected_client, replace=False)
                    else:
                        target_clients = np.delete(target_clients, [args.target_client])
                        target_clients = np.random.choice(target_clients, args.n_selected_client - 1, replace=False)
                        target_clients = numpy.append(target_clients, [args.target_client])
                    target_clients_dict[epoch] = target_clients
            else:
                del target_clients_dict[args.global_epochs - 1]
                target_clients = np.arange(args.n_client)
                target_clients = np.delete(target_clients, [args.target_client])
                target_clients = np.random.choice(target_clients, args.n_selected_client, replace=False)
                target_clients_dict[args.global_epochs - 1] = target_clients
    global_models, _ = federated_train(
        global_model=global_model,
        start_epoch=0,
        end_epoch=args.global_epochs,
        args=args,
        target_clients_dict=target_clients_dict,
        select_client=select_client)
    mia_compare(global_models, train_loader, test_loader, args)
    if type != 'all':
        if args.save_select:
            print('save {}: {}'.format(type, target_clients_dict))
            np.savez(DATA_PATH + '{}/clients_indices_{}_{}_{}_{}.npz'.
                     format(args.dataset, args.dataset, args.n_client, args.n_selected_client, type),
                     dict=target_clients_dict)
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
    attack_experiment(init_global_model, train_loader, test_loader, 'all', args)
    # evaluate in fractional aggregation scenario: 3/5(Y)
    logging.debug('-' * 10 + 'Select Random Clients, Yes' + '-' * 10)
    target_clients_dict = attack_experiment(init_global_model, train_loader, test_loader, 'yes', args)
    # evaluate in fractional aggregation scenario: 3/5(N)
    logging.debug('-' * 10 + 'Select Random Clients, No' + '-' * 10)
    attack_experiment(init_global_model, train_loader, test_loader, 'no', args, target_clients_dict)
    return


# 比较不同mia攻击的效果
def mia_compare(models, train_loader, test_loader, args):
    if args.non_iid == 0:
        model = models[-1]
        # blindmi
        # logging.debug('-' * 10 + 'BlindMI' + '-' * 10 + '\n')
        # blindMI_diff_w = BlindMI_Diff_W()
        # x, y_true, m_true = blindMI_diff_w.get_test_data(
        #     train_loader=train_loader,
        #     test_loader=test_loader,
        #     args=args)
        # acc, pre, recall, f1 = blindMI_diff_w.diff_Mem_attack(
        #     x_=x,
        #     y_true=y_true,
        #     m_true=m_true,
        #     target_model=model,
        #     args=args
        # )
        # logging.debug('blindMI_diff_w: acc {}, pre {}, recall {}, f1 {}'.format(acc, pre, recall, f1))
        #
        # # threshold
        # logging.debug('-' * 10 + 'Threshold MIA' + '-' * 10 + '\n')
        # threshold_attack = ThresholdAttack(args.metric)
        # train_dataset = threshold_attack.get_train_dataset(model, args)
        # test_dataset = threshold_attack.get_test_dataset(model, train_loader, test_loader, args)
        # acc, pre, recall, f1 = threshold_attack.attack(train_dataset, test_dataset, args)
        # logging.debug('threshold mia: acc {}, pre {}, recall {}, f1 {}'.format(acc, pre, recall, f1))
        #
        # # ML-Leaks
        # logging.debug('-' * 10 + 'ML Leaks' + '-' * 10 + '\n')
        # mlLeaks = MLLeaks()
        # mlLeaks.train_shadow_models(model, args)
        # attack_test_x, attack_test_y = mlLeaks.get_test_data(
        #     target_model=model,
        #     train_loader=train_loader,
        #     test_loader=test_loader,
        #     args=args)
        # dataset = (attack_test_x, attack_test_y)
        # acc, pre, recall, f1 = mlLeaks.attack(dataset, args)
        # logging.debug('ml leaks: acc {}, pre {}, recall {}, f1 {}'.format(acc, pre, recall, f1))
        #
        # # whitebox
        # logging.debug('-' * 10 + 'Whitebox' + '-' * 10 + '\n')
        # whitebox_attack = WhiteboxAttack(args)
        # acc, pre, recall, f1 = whitebox_attack.attack(models, train_loader, test_loader)
        # logging.debug('whitebox: acc {}, pre {}, recall {}, f1 {}'.format(acc, pre, recall, f1))

    logging.debug('-' * 10 + 'CS-MIA' + '-' * 10 + '\n')
    print(len(models))
    localCS = LocalCS(args.metric, direct=False)
    localCS.get_train_dataset(models, args)
    test_dataset = localCS.get_test_dataset(models, train_loader, test_loader, args)
    acc, pre, recall, f1 = localCS.attack(test_dataset, args)
    logging.debug('CS-MIA: acc {}, pre {}, recall {}, f1 {}'.format(acc, pre, recall, f1))


if __name__ == '__main__':
    # parse configuration
    args = get_args()
    logging.debug(args)

    if args.split_data:
        split_data(args)
    experiment(args)
