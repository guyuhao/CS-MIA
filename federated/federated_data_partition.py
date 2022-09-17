# -*- coding: utf-8 -*-
"""
Implementation of data partition for FL participants
"""
import logging
import random
from collections import defaultdict
from random import shuffle

import numpy as np

from common import DATA_PATH, get_dataset, get_dataset_labels


def data_partition(indices, args):
    """
    IID data partition for FL participants

    :param indices: indices of train dataset to divide
    :param args: configuration
    :return: list of train dataset indices for each FL participant
    """
    client_indices = []
    shuffle(indices)

    # generate auxiliary dataset for all clients if using adversarial regulation defense
    # only support Purchase100
    adv_reg = True if 'adv_reg' in args.keys() and args.adv_reg else False
    if adv_reg:
        ref_indices = []
        for i in range(args.n_client):
            ref_indices.append(indices[i * args.client_train_size: (i + 1) * args.client_train_size])
        for i in range(args.n_client):
            np.savez(DATA_PATH + '{}/adv_client{}_indices_{}_{}.npz'.format(
                args.dataset, i, args.dataset, args.client_train_size),
                     target_indices=ref_indices[i])
        indices = indices[: int(len(indices)/2)]

    # split train dataset for all clients with 'attack_test_m_size' records, owned alone by per client
    for i in range(args.n_client):
        client_indices.append(indices[i*args.attack_test_m_size: (i+1)*args.attack_test_m_size])

    # save train dataset (indices) of target client, as test members of attack model
    np.savez(DATA_PATH + '{}/client{}_remove_indices_{}_{}_{}.npz'.format(
        args.dataset, args.target_client, args.dataset, args.client_train_size, args.attack_test_m_size),
             target_indices=client_indices[args.target_client])

    # save partial train dataset (indices) of local attacker, as train members of attack model
    temp_indices = client_indices[args.cs_mia_client]
    shuffle(temp_indices)
    attack_indices = temp_indices[:args.attack_train_m_size]
    np.savez(DATA_PATH + '{}/client{}_cs_mia_indices_{}_{}_{}.npz'.format(
        args.dataset, args.cs_mia_client, args.dataset, args.client_train_size, args.attack_train_m_size),
             target_indices=attack_indices)

    indices = indices[args.n_client*args.attack_test_m_size:]
    # split remaining train dataset for all clients to satisfy 'client_train_size' records
    # train dataset of clients may overlap, such as on MNIST and CIFAR
    remain_size = args.client_train_size-args.attack_test_m_size
    if remain_size != 0:
        if len(indices) >= args.n_client*remain_size:
            for i in range(args.n_client):
                client_indices[i] = np.append(
                    client_indices[i],
                    indices[i*remain_size: (i+1)*remain_size])
        else:
            for i in range(args.n_client):
                client_indices[i] = np.append(
                    client_indices[i],
                    np.random.choice(indices, remain_size, replace=False))

    # save train dataset (indices) of all clients
    for i in range(args.n_client):
        np.savez(DATA_PATH + '{}/client{}_indices_{}_{}_{}.npz'.format(
            args.dataset, i, args.dataset, args.client_train_size, args.attack_test_m_size),
                 target_indices=client_indices[i])
    return client_indices


def non_iid_data_partition(indices, args):
    """
    non IID data partition for FL participants using Dirichlet distribution

    :param indices: indices of train dataset to divide
    :param args: configuration
    """
    # get labels of train dataset
    data_classes = {}
    train_dataset, _ = get_dataset(args.dataset)
    targets = get_dataset_labels(train_dataset, args.dataset)

    # save indices of train dataset for each class
    for ind in indices:
        label = int(targets[ind])
        if label in data_classes:
            data_classes[label].append(ind)
        else:
            data_classes[label] = [ind]
    class_size = len(data_classes[0])
    per_participant_list = defaultdict(list)
    no_classes = len(data_classes.keys())

    # for each class, use Dirichlet distribution to split dataset for all clients
    for n in range(no_classes):
        random.shuffle(data_classes[n])
        sampled_probabilities = class_size * np.random.dirichlet(
            np.array(args.n_client * [args.non_iid_alpha]))
        for user in range(args.n_client):
            data_size = int(round(sampled_probabilities[user]))
            sampled_list = data_classes[n][:min(len(data_classes[n]), data_size)]
            per_participant_list[user].extend(sampled_list)
            data_classes[n] = data_classes[n][min(len(data_classes[n]), data_size):]

    for client, client_indices in per_participant_list.items():
        logging.debug('origin client: {}, size: {}'.format(client, len(client_indices)))

        # save train dataset (indices) of all clients
        np.savez(DATA_PATH + '{}/client{}_indices_non_iid_{}_{}_{}.npz'.format(
            args.dataset, client, args.dataset, args.client_train_size, args.attack_test_m_size),
                 target_indices=client_indices)

        # save train dataset (indices) of target client, as test members of attack model
        if client == args.target_client:
            shuffle(client_indices)
            target_indices = client_indices[:args.attack_test_m_size]
            np.savez(DATA_PATH + '{}/client{}_remove_indices_non_iid_{}_{}_{}.npz'.format(
                args.dataset, args.target_client, args.dataset, args.client_train_size, args.attack_test_m_size),
                     target_indices=target_indices)
        # save partial train dataset (indices) of local attacker, as train members of attack model
        elif client == args.cs_mia_client:
            shuffle(client_indices)
            target_indices = client_indices[:args.attack_train_m_size]
            np.savez(DATA_PATH + '{}/client{}_cs_mia_indices_non_iid_{}_{}_{}.npz'.format(
                args.dataset, args.cs_mia_client, args.dataset, args.client_train_size, args.attack_train_m_size),
                     target_indices=target_indices)
    return
