# -*- coding: utf-8 -*-
"""
Implementation of secure aggregation, only support FedAvg
refers to "Practical Secure Aggregation for Privacy-Preserving Machine Learning"
"""
import copy
from copy import deepcopy

import numpy as np
import torch


def generate_weights(seed, dim):
    """
    generate random mask based on seed

    :param seed: random seed
    :param dim: dimension of model parameters
    :return: random mask
    """
    np.random.seed(seed)
    return np.float32(np.random.random(size=dim))


def prepare_weights(shared_keys, myid, weights, secret_key, mod, snd_key, dim):
    """
    add mask to client's local model

    :param shared_keys: pairwise public key of each client shared with other clients
    :param myid: client id
    :param weights: client's local model parameters
    :param secret_key: client's private key
    :param mod: mod
    :param snd_key: client's random seed
    :param dim: dimension of model parameters
    :return: masked model parameters
    """
    wghts = deepcopy(weights)
    for i, k in enumerate(wghts):
        d = dim[i]
        # add shared mask to model
        for sid in shared_keys:
            if sid > myid:
                wghts[k] += generate_weights((shared_keys[sid] ** secret_key) % mod, d)
            elif sid < myid:
                wghts[k] -= generate_weights((shared_keys[sid] ** secret_key) % mod, d)

        # add random mask to model
        wghts[k] += generate_weights(snd_key, d)
    return wghts


def model_dimension(weights):
    """
    get dimension of model parameters

    :param weights: model parameters
    :return: dimension
    """
    dimensions = []
    for k in weights.keys():
        tmp = weights[k].cpu().detach().numpy()
        cur_shape = tmp.shape
        dimensions.append(cur_shape)
    return dimensions


def reveal(weight, drop_clients, dim, id, keys, secret_key, mod):
    """
    recover drop clients' shared mask with each selected client

    :param weight: sum of selected clients' model parameters, containing mask
    :param drop_clients: drop clients
    :param dim: dimension of model parameters
    :param id: index of active client
    :param keys: pairwise public key of each client shared with other clients
    :param secret_key: private key of active client
    :param mod: mod
    :return: drop clients' shared mask with each selected client
    """
    wghts = deepcopy(weight)
    for i, k in enumerate(wghts):
        wghts[k] = torch.zeros_like(wghts[k])
        d = dim[i]
        for each in drop_clients:
            if each < id:
                wghts[k] -= generate_weights((keys[each] ** secret_key) % mod, d)
            elif each > id:
                wghts[k] += generate_weights((keys[each] ** secret_key) % mod, d)
    for i, k in enumerate(wghts):
        wghts[k] = -1*wghts[k]
    return wghts


def private_secret(weight, snd_key_dict, dim, index):
    """
    recover client's random mask

    :param weight: model parameters
    :param snd_key_dict: random seed of each client
    :param dim: dimension of model parameters
    :param index: client id
    :return: client's random mask
    """
    wghts = deepcopy(weight)
    for i, k in enumerate(wghts):
        wghts[k] = torch.zeros_like(wghts[k])
        d = dim[i]
        wghts[k] = -1*generate_weights(snd_key_dict[index], d)
    return wghts


def aggregate_model_reconstruction(weights, active_clients, dim, pub_key_dict, secret_key_dict, snd_key_dict, mod, args):
    """
    remove mask to aggregate global model

    :param weights: clients' local model parameters
    :param active_clients: selected clients in current round
    :param dim: dimension of model parameters
    :param pub_key_dict: pairwise public key of each client shared with other clients
    :param secret_key_dict: private key of each client
    :param snd_key_dict: random seed of each client
    :param mod: mod
    :param args: configuration
    :return: sum of global model parameters after aggregation
    """
    aggregate = None
    for temp in weights:
        aggregate = copy.deepcopy(weights[temp])
        break
    for j, k in enumerate(aggregate):
        for i, index in enumerate(weights):
            if i == 0:
                aggregate[k] = weights[index][k].clone()
            else:
                aggregate[k] += weights[index][k]

    # get drop clients in current round
    drop_clients = np.setdiff1d(np.arange(args.n_client), active_clients)

    # remove clients' shared mask
    if len(drop_clients) > 0:
        reveal_list = []
        for index in active_clients:
            reveal_result = reveal(weight=aggregate,
                                   drop_clients=drop_clients,
                                   dim=dim,
                                   id=index,
                                   keys=pub_key_dict,
                                   secret_key=secret_key_dict[index],
                                   mod=mod)
            reveal_list.append(reveal_result)

        for j, k in enumerate(aggregate):
            for weight in reveal_list:
                aggregate[k] += weight[k]

    # remove clients' random mask
    secret_list = []
    for index in active_clients:
        secret = private_secret(weight=aggregate,
                                snd_key_dict=snd_key_dict,
                                dim=dim,
                                index=index)
        secret_list.append(secret)
    for j, k in enumerate(aggregate):
        for weight in secret_list:
            aggregate[k] += weight[k]

    return aggregate
