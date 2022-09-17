# -*- coding: utf-8 -*-
"""
Implementation of standalone federated learning
"""

import copy
import logging
import time
from random import shuffle, randrange

import numpy as np
import torch

from common import train_model, load_data, test_model, TARGET_INDICES_FILE, SHADOW_INDICES_FILE, load_model, \
    train_for_gradient, get_optimizer, init_model, active_train_model
from federated.differential_privacy import client_differential_privacy, server_differential_privacy
from federated.knowledge_distillation import knowledge_distillation
from federated.secure_aggregation import model_dimension, aggregate_model_reconstruction, prepare_weights
from membership_inference.attack import get_attack_test_data

learning_rates = None
sigma = None

# whether local attacker perform active CS-MIA, not support True
client_active = False

# variables for secure aggregation defense
base = 2
mod = 100103
secret_key_dict = dict()  # private key of each client, the key is client id
pub_key_dict = dict()  # pairwise public key of each client shared with other clients, used with private key, the key is client id
snd_key_dict = dict()  # random seed of each client, the key is client id
dimensions = None

# variables for knowledge distillation defense
K = 10
T = 3

# variables for adversarial regularization defense
client_attack_models = list()


def federated_train(global_model, start_epoch, end_epoch, args,
                    target_clients_dict=dict(), select_client=False,
                    server_positive_attack=False, server_active_select=False, server_active_part=False,
                    server_passive_attack=False, fake_global_model=None,
                    dp=False):
    """
    federated training during rounds from 'start_epoch' to 'end_epoch'

    :param global_model: global model
    :param start_epoch: start round of FL
    :param end_epoch: end round of FL
    :param args: configuration
    :param target_clients_dict: specify selected clients in each round if valid, the key is round (start from 0)
    :param select_client: whether to aggregate fractional clients in each round, False means complete aggregation
    :param server_positive_attack: whether to conduct active global attack, including active participation and active selection
    :param server_active_select: whether to perform active selection for active global attacker, only valid if 'server_positive_attack' is False
    :param server_active_part: whether to perform active participation for active global attacker, only valid if 'server_positive_attack' is False
    :param server_passive_attack: whether to conduct passive global attack
    :param fake_global_model: shadow global model used in passive global attack
    :param dp: whether to use DP defense
    :return:
        tuple containing with global attack:
            (1) global_model_list: global models over rounds
            (2) target_clients_dict: selected clients in each round, the key is round (start from 0)
            (3) server_model_list:
            (4) target_client_model_list:
        tuple containing without attack:
            (1) global_model_list: global models over rounds
            (2) target_clients_dict: selected clients in each round, the key is round (start from 0)
    """
    global learning_rates, sigma
    global secret_key_dict, pub_key_dict, snd_key_dict
    global client_attack_models
    sigma = None
    non_iid_prefix = '_non_iid' if args.non_iid else ''
    if learning_rates is None:
        learning_rates = [args.target_learning_rate] * args.n_client

    # load test dataset
    indices_file = TARGET_INDICES_FILE + '{}_{}_{}_{}.npz'.\
        format(non_iid_prefix, args.dataset, args.target_train_size, args.target_test_size)
    _, test_loader = load_data(args.dataset, indices_file, args.target_batch_size, 'train-test')

    global_model_list = list()
    server_model_list = list()
    target_client_model_list = list()

    # active global attack includes active selection and active partition
    if server_positive_attack:
        server_active_select = True
        server_active_part = True

    server_attack = True if server_passive_attack or server_active_select or server_active_part else False
    save_step = 1
    if args.cuda:
        global_model = global_model.cuda()
        if fake_global_model is not None:
            fake_global_model = fake_global_model.cuda()
    global_optimizer = None

    aggregation = 'FedSGD' \
        if 'aggregation_algorithm' in args.keys() and args.aggregation_algorithm == 'FedSGD' \
        else 'FedAvg'

    # adjust learning rate by MultiStepLR for FedSGD
    global_scheduler = None
    if aggregation == 'FedSGD':
        global_optimizer = get_optimizer(global_model, args.target_learning_rate, args)
        global_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            global_optimizer,
            milestones=args.target_schedule,
            gamma=args.target_gamma)
        save_step = args.cs_mia_attack_step

    # initialize keys for each client with secure aggregation defense
    if 'sec_agg' in args.keys() and args.sec_agg:
        if len(pub_key_dict) == 0:
            for i in range(args.n_client):
                secret_key = randrange(mod)
                secret_key_dict[i] = secret_key
                pub_key_dict[i] = (base**secret_key) % mod
                snd_key_dict[i] = randrange(mod)

    for epoch in range(start_epoch, end_epoch):
        # federated training in one round
        # logging.debug("Global Epoch Training: {}".format(epoch))
        start_time = time.time()
        target_clients = np.arange(args.n_client)

        # select clients to aggregate in current round for fractional aggregation
        if select_client:
            if args.load_select == 1 and epoch in target_clients_dict.keys():
                target_clients = target_clients_dict[epoch]
            else:  # randomly select clients if not specified
                target_clients = np.random.choice(target_clients, args.n_selected_client, replace=False)
                target_clients_dict[epoch] = target_clients

            # select target client in each round in active global attack
            if server_active_select:  # 强制选中目标参与方
                if args.target_client not in target_clients:
                    shuffle(target_clients)
                    target_clients = np.delete(target_clients, 0)
                    target_clients = np.append(target_clients, args.target_client)
                    target_clients_dict[epoch] = target_clients

        client_models = list()
        target_select = False

        # local training of all clients
        for index in target_clients:
            # logging.debug("Global Epoch {} Training Client {}".format(epoch, index))
            # local training of each client
            client_model, client_data_size = client_train(
                global_model=global_model,
                client_index=index,
                global_epoch=epoch,
                select_client=select_client,
                args=args,
                differential_privacy=dp,
                aggregation=aggregation
            )

            # local attacker perform gradient ascent to conduct active attack, ineffective
            if client_active and index == 1:
                client_active_train(model=client_model, args=args)

            # save model parameters for FedAvg, or gradients for FedSGD
            if aggregation == 'FedAvg':
                client_models.append({'index': index, 'model': client_model, 'data_size': client_data_size})
            else:
                client_models.append({'index': index, 'grad': client_model, 'data_size': client_data_size})

            # save target client's model if it's selected in current round for global attacker
            if server_attack:
                if 'sec_agg' not in args.keys() or args.sec_agg == 0:
                    if epoch % save_step == 0:
                        if index == args.target_client:
                            target_client_model_list.append(get_target_client_model(
                                global_model=global_model,
                                global_optimizer=global_optimizer,
                                client_model=client_models[-1],
                                aggregation=aggregation,
                                args=args
                            ))
                            target_select = True
        # save None if target client is not selected in current round for global attacker, which is removed in further attack
        if server_attack:
            if 'sec_agg' not in args.keys() or args.sec_agg == 0:
                if epoch % save_step == 0:
                    if not target_select:
                        target_client_model_list.append(None)

        # malicious server trains local model in global attack
        if server_attack:
            # attacker trains local model based on pre-round global model in active global CS-MIA
            if server_active_part:
                logging.debug("Global Epoch {} Training Server".format(epoch))
                server_model, server_data_size = server_train(
                    global_model=global_model,
                    global_epoch=epoch,
                    select_client=select_client,
                    args=args,
                    aggregation=aggregation)
            # attacker trains local model based on pre-round shadow global model in passive global CS-MIA
            if server_passive_attack:
                logging.debug("Global Epoch {} Training Server".format(epoch))
                server_model, server_data_size = server_train(
                    global_model=fake_global_model,
                    global_epoch=epoch,
                    select_client=select_client,
                    args=args,
                    aggregation=aggregation)

            # save shadow model in current round for malicious server
            if epoch % save_step == 0:
                # save shadow/true global model to conduct local CS-MIA for passive/active global attack with secure aggregation defense
                if 'sec_agg' in args.keys() and args.sec_agg:
                    # only support FedAvg
                    tmp = {'model': fake_global_model if server_passive_attack else global_model}
                    server_model_list.append(tmp)
                # save server's local model for global attack
                else:
                    server_model_list.append(get_target_client_model(
                                global_model=fake_global_model if server_passive_attack else global_model,
                                global_optimizer=global_optimizer,
                                client_model=server_model,
                                aggregation=aggregation,
                                args=args
                            ))

        # save global model in current round for malicious server to conduct local CS-MIA, only support FedAvg
        if server_attack:
            if 'sec_agg' in args.keys() and args.sec_agg:
                tmp = {'model': global_model}
                target_client_model_list.append(tmp)

        # add malicious server's local model to aggregate for active global CS-MIA
        if server_active_part:
            if aggregation == 'FedAvg':
                client_models.append({'index': -1, 'model': server_model, 'data_size': server_data_size})
            else:
                client_models.append({'index': -1, 'grad': server_model, 'data_size': server_data_size})

        # aggregate to get shadow global model for passive global CS-MIA
        if server_passive_attack:
            fake_client_models = list()
            fake_client_models += client_models
            if aggregation == 'FedAvg':
                fake_client_models.append({'index': -1, 'model': server_model, 'data_size': server_data_size})
            else:
                fake_client_models.append({'index': -1, 'grad': server_model, 'data_size': server_data_size})
            fake_global_model = fed_aggregation(
                global_model=fake_global_model,
                global_optimizer=global_optimizer,
                global_scheduler=global_scheduler,
                client_models=fake_client_models,
                aggregation=aggregation,
                fake=True,
                args=args,
                dp=False,
                epoch=epoch
            )
            if args.cuda:
                fake_global_model = fake_global_model.cuda()

        # aggregate to get global model for normal federated training or active global CS-MIA
        global_model = fed_aggregation(
            global_model=global_model,
            global_optimizer=global_optimizer,
            global_scheduler=global_scheduler if not server_passive_attack else None,
            client_models=client_models,
            aggregation=aggregation,
            args=args,
            dp=dp,
            epoch=epoch
        )

        if args.cuda:
            global_model = global_model.cuda()

        # remove malicious server's local model for active global CS-MIA
        if server_active_part:
            client_models.pop()

        end_time = time.time()
        logging.debug('Global Epoch {}, training time: {}s'.format(epoch, end_time - start_time))

        # evaluate global model performance on test dataset
        test_losses, acc, pre, recall, f1 = test_model(global_model, test_loader, args, args.num_classes)
        logging.debug('Global Epoch {}, Global Model testing loss: {}, accuracy: {}'.format(epoch, test_losses, acc))

        # evaluate shadow global model performance on test dataset
        if fake_global_model is not None:
            losses, acc, pre, recall, f1 = test_model(fake_global_model, test_loader, args, args.num_classes)
            logging.debug('Global Epoch {}, Fake Global Model training loss: {}, accuracy: {}'.format(epoch, losses, acc))

        # save global model in current round for local attacker
        if not server_attack:
            if epoch % save_step == 0:
                global_model_list.append(copy.deepcopy(global_model))

    if server_attack:
        # translate state dict to model in FedSGD
        if aggregation == 'FedSGD':
            for i, temp in enumerate(server_model_list):
                temp_model = init_model(args)
                temp_model.load_state_dict(temp)
                server_model_list[i] = temp_model
            for i, temp in enumerate(target_client_model_list):
                if temp is not None:
                    temp_model = init_model(args)
                    temp_model.load_state_dict(temp['model'])
                    target_client_model_list[i]['model'] = temp_model

        return global_model_list, target_clients_dict, server_model_list, target_client_model_list
    else:
        return global_model_list, target_clients_dict


def client_active_train(model, args):
    """
    local attacker perform active attack, ineffective

    :param model: attacker's local model after training in the current round
    :param args: configuration
    """
    train_loader, test_loader = get_attack_test_data(args)
    active_train_model(model=model, train_loader=train_loader, test_loader=test_loader, args=args)
    return


def get_target_client_model(global_model, global_optimizer, client_model, aggregation, args):
    """
    get model parameters of local model

    :param global_model: model architecture
    :param global_optimizer: optimizer of local model
    :param client_model: model parameters for FedAvg, or gradients for FedSGD
    :param aggregation: aggregation algorithm, 'FedAvg' or 'FedSGD'
    :param args: configuration
    :return: model parameters of local model
    """
    # for FedAvg, return copy of client_model
    if aggregation == 'FedAvg':
        return copy.deepcopy(client_model)
    # for FedSGD, generate model parameters based on gradients
    else:
        # get gradients
        if 'grad' in client_model.keys():
            grad = client_model['grad']
        else:
            grad = client_model
        # perform gradient descent on global model to get model parameters
        temp_global_model = copy.deepcopy(global_model)
        temp_global_optimizer = get_optimizer(temp_global_model, args.target_learning_rate, args)
        temp_global_optimizer.load_state_dict(global_optimizer.state_dict())
        temp_global_optimizer.zero_grad()
        for name, param in temp_global_model.named_parameters():
            param.grad = grad[name].detach().clone()
        temp_global_optimizer.step()

        # if client_model is dict, add model parameters to dict
        # else return model parameters directly
        if 'grad' in client_model.keys():
            result = dict()
            for name, value in client_model.items():
                if name != 'grad':
                    result[name] = value
            result['model'] = temp_global_model.cpu().state_dict()
        else:
            result = temp_global_model.cpu().state_dict()
        return result


def client_load(global_model, client_index, global_epoch, select_client, args):
    """
    load client's local model from local file, not used

    :param global_model: model architecture
    :param client_index: index of client
    :param global_epoch: current round
    :param select_client: whether is in fractional aggregation
    :param args: configuration
    :return: loaded client's local model
    """
    model = copy.deepcopy(global_model)
    if select_client:
        model_type = 'target_select_{}_{}client_{}_{}'.format(args.target_model, args.n_client, client_index, global_epoch)
    else:
        model_type = 'target_all_{}_{}client_{}_{}'.format(args.target_model, args.n_client, client_index, global_epoch)
    model = load_model(model, model_type, args)
    return model


def client_train(global_model, client_index, global_epoch, select_client, args, aggregation, differential_privacy=False):
    """
    local training of each client in one round

    :param global_model: pre-round global model
    :param client_index: index of client
    :param global_epoch: current round
    :param select_client: whether is in fractional aggregation
    :param args: configuration
    :param aggregation: aggregation algorithm, 'FedAvg' or 'FedSGD'
    :param differential_privacy: whether to use DP defense
    :return: tuple containing:
        (1) model: trained local model for FedAvg, or gradients after training for FedSGD
        (2) size: size of client's local train dataset
    """
    global sigma
    global dimensions

    model = copy.deepcopy(global_model)

    # load local train dataset of client
    non_iid_prefix = '_non_iid' if args.non_iid else ''
    indices_file = 'client{}_indices{}_{}_{}_{}.npz'.format(
        client_index, non_iid_prefix, args.dataset, args.client_train_size, args.attack_test_m_size)
    train_loader, _ = load_data(dataset=args.dataset,
                                indices_file=indices_file,
                                batch_size=args.target_batch_size,
                                type='target')

    # train local model and return it for FedAvg
    if aggregation == 'FedAvg':
        if select_client:
            model_type = 'target_select_{}_{}client_{}_{}'.format(args.target_model, args.n_client, client_index,
                                                                  global_epoch)
        else:
            model_type = 'target_all_{}_{}client_{}_{}'.format(args.target_model, args.n_client, client_index,
                                                               global_epoch)

        # train local model with knowledge distillation defense
        if 'know_dist' in args.keys() and args.know_dist:
            knowledge_distillation(train_loader, model, args)
            return model, len(train_loader.dataset)

        # train local model without defense
        result = train_model(model=model,
                             model_type=model_type,
                             train_loader=train_loader,
                             test_loader=None,
                             args=args,
                             learning_rate=None,
                             dp=differential_privacy,
                             load=args.load_target)

        # add noise to trained local model with DP defense
        if differential_privacy:
            client_differential_privacy(model=model,
                                        parameters_list=result,
                                        epoch=global_epoch,
                                        args=args)

        # add mask to trained local model with secure aggregation defense
        if 'sec_agg' in args.keys() and args.sec_agg:
            if dimensions is None:
                dimensions = model_dimension(global_model.state_dict())
            weights = model.cpu().state_dict()
            masked_weights = prepare_weights(shared_keys=pub_key_dict,
                                             myid=client_index,
                                             weights=weights,
                                             secret_key=secret_key_dict[client_index],
                                             mod=mod,
                                             snd_key=snd_key_dict[client_index],
                                             dim=dimensions)
            return masked_weights, len(train_loader.dataset)

        return model, len(train_loader.dataset)
    # train local model and return gradients for FedSGD
    else:
        grad_dict = train_for_gradient(
            model=model,
            train_loader=train_loader,
            args=args,
            dp=differential_privacy)
        return grad_dict, len(train_loader.dataset)


def server_train(global_model, global_epoch, select_client, args, aggregation, differential_privacy=False):
    """
    global attacker perform local training

    :param global_model: pre-round global model for active global CS-MIA, or pre-round shadow global model for passive global CS-MIA
    :param global_epoch: current round
    :param select_client: whether is in fractional aggregation
    :param args: configuration
    :param aggregation: aggregation algorithm, 'FedAvg' or 'FedSGD'
    :param differential_privacy: whether to use DP defense
    :return: tuple containing:
        (1) model: trained local model for FedAvg, or gradients after training for FedSGD
        (2) size: size of server's local train dataset
    """
    model = copy.deepcopy(global_model)

    # load auxiliary dataset of global attacker
    non_iid_prefix = '_non_iid' if args.non_iid else ''
    indices_file = SHADOW_INDICES_FILE + '{}_{}_{}_{}.npz'.format(
        non_iid_prefix, args.dataset, args.attack_train_m_size, args.attack_train_nm_size)
    train_loader, test_loader = load_data(
        dataset=args.dataset,
        indices_file=indices_file,
        batch_size=args.target_batch_size,
        type='train-test')

    # train local model and return it for FedAvg
    if aggregation == 'FedAvg':
        if select_client:
            model_type = 'target_select_{}_server_{}_{}'.format(args.target_model, args.n_client, global_epoch)
        else:
            model_type = 'target_all_{}_server_{}_{}'.format(args.target_model, args.n_client, global_epoch)

        # train local model without defense
        result = train_model(model=model,
                             model_type=model_type,
                             train_loader=train_loader,
                             test_loader=None,
                             args=args,
                             learning_rate=None,
                             load=False)

        # add noise to trained local model with DP defense
        if differential_privacy:
            client_differential_privacy(model=model,
                                        parameters_list=result,
                                        epoch=global_epoch,
                                        args=args)

        if 'sec_agg' in args.keys() and args.sec_agg:
            return model.cpu().state_dict(), len(train_loader.dataset)
        else:
            return model, len(train_loader.dataset)
    # train local model and return gradients for FedSGD
    else:
        grad_dict = train_for_gradient(
            model=model,
            train_loader=train_loader,
            args=args)
        return grad_dict, len(train_loader.dataset)


def fed_aggregation(global_model, global_optimizer, global_scheduler, client_models, aggregation, epoch, args,
                    fake=False, dp=False):
    """
    server performs aggregation

    :param global_model: pre-round global model
    :param global_optimizer: optimizer of global model
    :param global_scheduler: scheduler of global model
    :param client_models: model parameters updated by clients in FedAvg, or gradients in FedSGD
    :param aggregation: aggregation algorithm, 'FedAvg' or 'FedSGD'
    :param epoch: current round
    :param args: configuration
    :param fake: whether to aggregate shadow global model
    :param dp: whether to use DP defense
    :return: aggregated global model in current round
    """
    # aggregation in FedSGD
    if aggregation == 'FedSGD':
        return fed_sgd(
            global_model=global_model,
            global_optimizer=global_optimizer,
            global_scheduler=global_scheduler,
            client_models=client_models,
            fake=fake,
            args=args
        )
    # aggregation in FedAvg
    else:
        # judge whether to aggregate global attacker's local model
        contain_server = False
        if client_models[-1]['index'] == -1:
            contain_server = True

        # perform FedAvg aggregation
        result = fed_avg(client_models, epoch, args, contain_server, dp)

        if 'sec_agg' in args.keys() and args.sec_agg:
            model = copy.deepcopy(global_model)
            model.load_state_dict(result)
            return model
        else:
            return result


def fed_sgd(global_model, global_optimizer, global_scheduler, client_models, fake, args):
    """
    perform FedSGD aggregation

    :param global_model: pre-round global model
    :param global_optimizer: optimizer of global model
    :param global_scheduler: scheduler of global model
    :param client_models: gradients updated by clients
    :param fake: whether to aggregate shadow global model
    :param args: configuration
    :return: aggregated global model
    """
    # copy global model parameters and optimizer for aggregating shadow global model
    if fake:
        model = copy.deepcopy(global_model)
        optimizer = get_optimizer(model, args.target_learning_rate, args)
        optimizer.load_state_dict(global_optimizer.state_dict())
    else:
        model = global_model
        optimizer = global_optimizer

    assert len(client_models) > 0

    # compute average of clients' updated gradients
    global_grad = copy.deepcopy(client_models[0]['grad'])
    avg_grad_dict = global_grad
    local_grads = list()
    data_size_list = list()
    for temp_model in client_models:
        local_grads.append(temp_model['grad'])
        data_size_list.append(int(temp_model['data_size']))
    all_data_size = np.sum(data_size_list)
    client_num = len(client_models)
    for layer in avg_grad_dict.keys():
        avg_grad_dict[layer] *= 0
        for client_index in range(client_num):
            avg_grad_dict[layer] += (data_size_list[client_index] / all_data_size) * local_grads[client_index][
                layer]

    # perform gradient descent to global model based on average gradients
    optimizer.zero_grad()
    for name, param in model.named_parameters():
        temp_grad = avg_grad_dict[name].to(param.device)
        param.grad = temp_grad.detach().clone()
    optimizer.step()

    # adjust learning rate
    if global_scheduler is not None:
        global_scheduler.step()

    return model


def fed_avg(client_models, epoch, args, contain_server=False, dp=False):
    """
    perform FedAvg aggregation

    :param client_models: model parameters updated by clients
    :param epoch: current round
    :param args: configuration
    :param contain_server: whether to aggregate global attacker's local model, True in active global attack
    :param dp: whether to use DP defense
    :return: aggregated global model without secure aggregation defense, and model parameters otherwise
    """
    assert len(client_models) > 0
    # perform FedAvg aggregation with secure aggregation defense
    if 'sec_agg' in args.keys() and args.sec_agg:
        final_index = len(client_models)
        if contain_server:
            final_index = final_index - 1
        active_clients = []
        model_dict = dict()
        for i in range(0, final_index):
            temp = client_models[i]
            active_clients.append(temp['index'])
            model_dict[temp['index']] = temp['model']

        # aggregate clients' local models with secure aggregation defense
        global_model_params = aggregate_model_reconstruction(
            weights=model_dict,
            active_clients=active_clients,
            dim=dimensions,
            pub_key_dict=pub_key_dict,
            secret_key_dict=secret_key_dict,
            snd_key_dict=snd_key_dict,
            mod=mod,
            args=args)
        for layer in global_model_params.keys():
            global_model_params[layer] = global_model_params[layer] / len(active_clients)

        # for active global attack, aggregate malicious server's local model
        if contain_server:
            server_model = client_models[-1]['model']
            client_data_size = int(client_models[0]['data_size'])
            server_data_size = int(client_models[-1]['data_size'])
            all_data_size = client_data_size + server_data_size
            for layer in global_model_params.keys():
                global_model_params[layer] = (client_data_size / all_data_size) * global_model_params[layer] + \
                                             (server_data_size / all_data_size) * server_model[layer]

        return global_model_params
    # perform FedAvg aggregation without secure aggregation defense
    else:
        global_model = copy.deepcopy(client_models[0]['model'])
        avg_state_dict = global_model.state_dict()

        local_state_dicts = list()
        data_size_list = list()
        for model in client_models:
            local_state_dicts.append(model['model'].state_dict())
            data_size_list.append(int(model['data_size']))

        all_data_size = np.sum(data_size_list)
        client_num = len(client_models)
        for layer in avg_state_dict.keys():
            avg_state_dict[layer] *= 0
            for client_index in range(client_num):
                avg_state_dict[layer] += (data_size_list[client_index]/all_data_size)*local_state_dicts[client_index][layer]

        global_model.load_state_dict(avg_state_dict)

        # add noise to aggregated global model with DP defense
        if dp:
            temp_models = client_models if not contain_server else client_models[:-1]
            server_differential_privacy(model=global_model,
                                        epoch=epoch,
                                        n_selected_client=len(temp_models),
                                        client_models=[model['model'] for model in temp_models],
                                        args=args)
        return global_model

