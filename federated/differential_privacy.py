# -*- coding: utf-8 -*-
"""
Implementation of DP defense, refers to "Federated Learning With Differential Privacy: Algorithms and Performance Analysis"
"""
import copy
import math

import numpy as np
import torch


def get_total_norm(parameters, norm_type=2):
    """
    calculate norm of model parameters

    :param parameters: model parameters
    :param int norm_type: 2 means L2 norm
    :return: norm of model parameters
    """
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].device
    total_norm = torch.norm(torch.stack([torch.norm(p.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def clip_norm(parameters, max_norm, norm_type=2):
    """
    clip norm of model parameters

    :param parameters: model parameters
    :param max_norm: clip threshold
    :param norm_type: 2 means L2 norm
    """
    max_norm = float(max_norm)
    total_norm = get_total_norm(parameters, norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    # print(total_norm)
    if clip_coef < 1:
        for p in parameters:
            p.detach().mul_(clip_coef.to(p.device))
    # print(get_total_norm(parameters, norm_type))


def get_C(parameters_list):
    """
    get clip threshold of model parameters, constant

    :param parameters_list: list of model parameters
    :return: clip threshold
    """
    # pick median of all model parameters as clip threshold
    # required in paper, but resulting in high train loss, so use constant instead
    # norm_list = list()
    # for parameters in parameters_list:
    #     if isinstance(parameters, list):
    #         total_norm = get_total_norm(parameters)
    #     else:
    #         total_norm = get_total_norm([p for p in parameters.parameters()])
    #     norm_list.append(total_norm.item())
    # return np.median(norm_list)

    return 95.0


def clip(model, parameters_list):
    """
    clip model parameters

    :param model: model
    :param parameters_list: model parameters during training used to compute clip threshold, not used
    :return: clip threshold
    """
    C = get_C(parameters_list)
    clip_norm([p for p in model.parameters()], C)
    return C


def client_add_noise(model, C, epoch, args):
    """
    add noise to local model updates

    :param model: model
    :param C: clip threshold
    :param epoch: current round, not used
    :param args: configuration
    """
    c = math.sqrt(2 * math.log(1.25 / args.delta))
    # L = epoch + 1
    L = args.global_epochs
    m = args.client_train_size
    epsilon = args.epsilon
    sigma = (2 * c * L * C) / (m * epsilon)
    # print(sigma)
    for p in model.parameters():
        grad_noise = torch.normal(0, sigma, size=p.shape, device=p.device)
        p.data += grad_noise


def client_differential_privacy(model, parameters_list, epoch, args):
    """
    perform LDP introduced in paper

    :param model: client model
    :param parameters_list: client model parameters during training
    :param epoch: current round
    :param args: configuration
    """
    C = clip(model, parameters_list)
    client_add_noise(model, C, epoch, args)
    # print(get_total_norm([p for p in model.parameters()]))


def server_add_noise(model, epoch, n_selected_client, client_models, args):
    """
    add noise to global model

    :param model: global model
    :param epoch: current round, not used
    :param n_selected_client: number of selected clients in current round
    :param client_models: clients' model updates
    :param args: configuration
    """
    T = args.global_epochs
    L = T
    # L = 1
    N = args.n_client
    if n_selected_client == N:
        if T > (L*math.sqrt(N)):
            C = get_C(client_models)
            # print (C)
            c = math.sqrt(2 * math.log(1.25 / args.delta))
            m = args.client_train_size
            epsilon = args.epsilon
            sigma = (2*c*C*math.sqrt(T*T-L*L*N))/(m*N*epsilon)
        else:
            sigma = 0
    else:
        epsilon = args.epsilon
        K = n_selected_client
        r = -math.log(1-K/N+K/N*math.pow(math.e, -epsilon/(L*math.sqrt(K))))
        if T > epsilon/r:
            b = -T/epsilon*math.log(1-N/K+N/K*math.pow(math.e, -epsilon/T))
            C = get_C(client_models)
            c = math.sqrt(2 * math.log(1.25 / args.delta))
            m = args.client_train_size
            sigma = 2*c*C*math.sqrt((T*T)/(b*b)-L*L*K)/(m*K*epsilon)
        else:
            sigma = 0
    if sigma != 0:
        for p in model.parameters():
            grad_noise = torch.normal(0, sigma, size=p.shape, device=p.device)
            p.data += grad_noise


def server_differential_privacy(model, epoch, n_selected_client, client_models, args):
    """
    perform CDP introduced in paper

    :param model: global model
    :param epoch: current round, not used
    :param n_selected_client: number of selected clients in current round
    :param client_models: clients' model updates
    :param args: configuration
    """
    server_add_noise(model, epoch, n_selected_client, client_models, args)
