# -*- coding: utf-8 -*-
"""
Implementation of knowledge distillation defense, only support FedAvg
refers to "Resisting membership inference attacks through knowledge distillation"
"""
import copy

import torch
from torch import nn, optim

from common import train_model
from utils import AverageMeter

K = 10
T = 3


def private_learning(train_loader, model, args):
    """
    train k teacher models

    :param train_loader: loader of client's local dataset
    :param model: pre-round global model
    :param args: configuration
    :return: sub-datasets, teacher models
    """
    # split train dataset into K sub-datasets
    data_size = len(train_loader.dataset)
    size_list = [data_size//K] * K
    train_dataset = train_loader.dataset
    dataset_list = torch.utils.data.random_split(train_dataset, size_list)

    # train a teacher model on each sub-dataset
    model_list = []
    for i, dataset in enumerate(dataset_list):
        temp_model = copy.deepcopy(model)
        loader = torch.utils.data.DataLoader(dataset, batch_size=train_loader.batch_size, shuffle=True)
        train_model(model=temp_model,
                    model_type='target_teacher_{}'.format(i),
                    train_loader=loader,
                    test_loader=None,
                    args=args,
                    learning_rate=None,
                    load=False)
        model_list.append(temp_model)
    return dataset_list, model_list


def pre_distillation(partition_sets, teacher_models, batch_size, args):
    """
    use teacher models to generate train dataset for student model

    :param partition_sets: K sub-datasets
    :param teacher_models: K trained teacher models
    :param batch_size: batch size of to train student model
    :param args: configuration
    :return: loader of train dataset for student model
    """
    x_list, y_list = [], []
    softmax = nn.Softmax(dim=1)
    for temp in teacher_models:
        temp.eval()

    # for each sub-dataset, use average of other K-1 teacher models' output as labels of student model's train dataset
    for i, dataset in enumerate(partition_sets):
        # get other K-1 teacher models
        temp_models = teacher_models[:i] + teacher_models[i+1:]
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # get average output of teacher models
        for x, _ in loader:
            if args.cuda:
                x = x.cuda()
            temp_y = []
            for temp_model in temp_models:
                output = temp_model(x)/T
                temp_y.append(softmax(output).unsqueeze(0))
            y = torch.cat(temp_y, dim=0)
            avg = torch.mean(y, dim=0)
            x_list.append(x)
            y_list.append(avg)

    ds = torch.utils.data.TensorDataset(torch.cat(x_list, dim=0), torch.cat(y_list, dim=0))
    return ds


def distillation(model, synthetic_set, batch_size, args):
    """
    train student model

    :param model: pre-round global model
    :param synthetic_set: train dataset generated by teacher models
    :param batch_size: batch size of train dataset
    :param args: configuration
    """
    model.train()
    losses = AverageMeter()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.target_learning_rate)
    log_softmax = nn.LogSoftmax(dim=1)
    criterion = nn.KLDivLoss()
    loader = torch.utils.data.DataLoader(synthetic_set, batch_size=batch_size, shuffle=True)
    for epoch in range(0, args.target_epochs):
        for batch_idx, (inputs, targets) in enumerate(loader):
            if args.cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            # compute output
            outputs = model(inputs)
            final_outputs = log_softmax(outputs / T)
            loss = criterion(final_outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def knowledge_distillation(train_loader, model, args):
    """
    use knowledge distillation to train student model as client's local model

    :param train_loader: loader of client's local dataset
    :param model: pre-round global model
    :param args: configuration
    :return: trained local model (student local)
    """
    # train k teacher models
    partition_sets, teacher_models = private_learning(train_loader, model, args)
    # use teacher models to generate train dataset for student model
    synthetic_set = pre_distillation(partition_sets, teacher_models, train_loader.batch_size, args)
    # train student model
    distillation(model, synthetic_set, train_loader.batch_size, args)
    # return student model
    return model