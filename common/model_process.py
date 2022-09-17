# -*- coding: utf-8 -*-
"""
Implementation of model training, testing and so on
"""
import copy
import logging
import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, classification_report

from common import CHECKPOINT_PATH
from model.alexnet import alexnet
from model.cnn_cifar import cnn_cifar
from model.cnn_mnist import cnn_mnist
from model.fcn_purchase import fcn_purchase
from model.fcn_texas import fcn_texas
from utils import AverageMeter, accuracy, attack_accuracy


def get_num_classes(dataset):
    """
    get number of dataset classes

    :param dataset: dataset name
    :return: number of dataset classes
    """
    data_dict = {
        'cifar10': 10,
        'cifar100': 100,
        'purchase100': 100,
        'texas100': 100,
        'mnist': 10
    }
    return data_dict[dataset]


def init_model(args):
    """
    initialize model

    :param args: configuration
    :return: model
    """
    num_classes = get_num_classes(args.dataset)
    if 'cifar' in args.dataset:
        if args.target_model == 'alexnet':
            return alexnet(num_classes=num_classes)
        elif args.target_model == 'cnn':
            return cnn_cifar(num_classes=num_classes)
    elif args.dataset == 'purchase100':
        # whether to use dropout defense
        dropout = True if 'dropout' in args.keys() and args.dropout else False
        # dropout ratio
        p = args.dropout_p if 'dropout' in args.keys() and args.dropout else None
        # whether to use adversarial regularization defense
        adv_reg = True if 'adv_reg' in args.keys() and args.adv_reg else False

        return fcn_purchase(num_classes=num_classes, dropout=dropout, p=p, adv_reg=adv_reg)
    elif args.dataset == 'texas100':
        return fcn_texas(num_classes=num_classes)
    elif args.dataset == 'mnist':
        if args.target_model == 'cnn':
            return cnn_mnist(num_classes=num_classes)


def adjust_learning_rate(optimizer, epoch, schedule, gamma, state):
    """
    adjust learning rate during training epochs, not used
    """
    if epoch in schedule:
        state['lr'] *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


def get_optimizer(model, learning_rate, args):
    """
    generate optimizer for model, used Adam for Purchase100, Texas100,
    and MNIST, used Adam or SGD for CIFAR according to model type

    :param model: model
    :param learning_rate: learning rate
    :param args: configuration
    :return: optimizer for model
    """
    optimizer = None
    if 'cifar' in args.dataset:
        if args.target_model == 'cnn':
            optimizer = torch.optim.Adam(model.parameters(),
                                        lr=learning_rate)
        elif args.target_model == 'alexnet' or args.target_model == 'densenet':
            optimizer = optim.SGD(model.parameters(),
                                  lr=learning_rate,
                                  momentum=args.target_momentum,
                                  weight_decay=args.target_wd)
    elif args.dataset == 'purchase100':
        optimizer = optim.Adam(model.parameters(),
                               lr=learning_rate)
    elif args.dataset == 'texas100':
        optimizer = optim.Adam(model.parameters(),
                               lr=learning_rate)
    elif args.dataset == 'mnist':
        if args.target_model == 'cnn':
            optimizer = optim.Adam(model.parameters(),
                                   lr=learning_rate,
                                   weight_decay=args.target_wd)
    return optimizer


def load_model(model, model_type, args):
    """
    load model from local file CHECKPOINT_PATH/"dataset"_"model name"/"model_type"

    :param model: model
    :param model_type: local file name
    :param args: configuration
    :return: loaded model
    """
    checkpoint_path = '{}/{}_{}'.format(CHECKPOINT_PATH, args.dataset, args.target_model)
    resume = '{}/{}'.format(checkpoint_path, model_type)
    if os.path.isfile(resume):
        logging.debug('==> Resuming from checkpoint..')
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['state_dict'])
        return model
    else:
        return None


def train_model(model,
                model_type='',
                train_loader=None,
                test_loader=None,
                args=None,
                learning_rate=None,
                load=False,
                norm='acc',
                dp=False,
                debug=False):
    """
    model training

    :param model: model
    :param model_type: model type, used to judge which model to train (target, shadow, or attack model), the local file name to save model
    :param train_loader: loader of train dataset
    :param test_loader: loader of test dataset, not evaluate test accuracy if is None
    :param args: configuration
    :param learning_rate: specified learning rate, use configuration if is None
    :param load: whether to load model from local file without training
    :param norm: metric to evaluate model performance, 'acc' (accuracy) or 'pre' (precision)
    :param dp: whether to use DP defense
    :param debug: whether to log debug information
    :return: model parameters over epochs if dp is True,
             else accuracy, precision, recall, and F1-score of the best (evaluated by norm) model during training
    """
    best = 0.0
    acc, pre, recall, f1 = 0.0, 0.0, 0.0, 0.0
    if args.cuda:
        model = model.cuda()

    checkpoint_file = '{}/{}_{}'.format(CHECKPOINT_PATH, args.dataset, args.target_model)
    # load model from local file without training
    if load:
        resume = '{}/{}'.format(checkpoint_file, model_type)
        if os.path.isfile(resume):
            logging.debug('==> Resuming from checkpoint..')
            checkpoint = os.path.dirname(resume)
            checkpoint = torch.load(resume)
            model.load_state_dict(checkpoint['state_dict'])
            return

    epochs, schedule, gamma, num_classes = 0, [], 0.1, 0
    optimizer, criterion = None, None
    # judge whether model is attack model based on model_type
    is_attack = False if 'target' in model_type or 'shadow' in model_type else True

    # initialize hyper-parameters, optimizer, and criterion of target or shadow model
    if not is_attack:
        gamma = args.target_gamma
        # initialize hyper-parameters of target model
        if 'target' in model_type:
            learning_rate, epochs, schedule = \
                args.target_learning_rate if learning_rate is None else learning_rate, \
                args.target_epochs, \
                args.target_schedule
        # initialize hyper-parameters of shadow model
        elif 'shadow' in model_type:
            learning_rate, epochs, schedule = \
                args.shadow_learning_rate if learning_rate is None else learning_rate, \
                args.shadow_epochs, \
                args.shadow_schedule
        num_classes = args.num_classes
        optimizer = get_optimizer(model, learning_rate, args)
        if not dp:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss(reduction='sum')
    # initialize hyper-parameters, optimizer, and criterion of attack model
    else:
        # initialize hyper-parameters, optimizer, and criterion of attack model in CS-MIA
        if 'cs_mia' in model_type:
            learning_rate, epochs, schedule, gamma = \
                args.cs_mia_learning_rate if learning_rate is None else learning_rate, \
                args.cs_mia_epochs, \
                args.cs_mia_schedule, \
                args.cs_mia_gamma
            optimizer = optim.Adam(model.parameters(),
                                   lr=learning_rate,
                                   weight_decay=args.cs_mia_wd)
        # initialize hyper-parameters, optimizer, and criterion of attack model in ML-Leaks
        elif 'ml_leaks' in model_type:
            learning_rate, epochs, schedule, gamma = \
                args.ml_leaks_learning_rate if learning_rate is None else learning_rate, \
                args.ml_leaks_epochs, \
                args.ml_leaks_schedule, \
                args.ml_leaks_gamma
            optimizer = optim.Adam(model.parameters(),
                                   lr=learning_rate,
                                   weight_decay=args.ml_leaks_wd)
        # initialize hyper-parameters, optimizer, and criterion of attack model in White-box attack
        elif 'whitebox' in model_type:
            learning_rate, epochs = \
                args.whitebox_learning_rate if learning_rate is None else learning_rate, \
                args.whitebox_epochs
            optimizer = optim.Adam(model.parameters(),
                                   lr=learning_rate)
        num_classes = 2
        # use BCELoss for all attack models
        criterion = nn.BCELoss()

    state = dict()
    state['lr'] = learning_rate

    epoch_list = list()
    train_losses = list()
    test_losses = list()

    parameters_list = list()
    for epoch in range(0, epochs):
        adjust_learning_rate(optimizer, epoch, schedule, gamma, state)
        # train model in one epoch
        train_loss, train_acc = train(train_loader, model, criterion,
                                      optimizer, args.cuda, is_attack)

        # save model parameters over epochs if use DP defense
        if dp:
            parameters_list.append([p for p in copy.deepcopy(model).parameters()])

        if debug:
            logging.debug('train loss: {}, accuracy: {}'.format(train_loss, train_acc))
        epoch_list.append(epoch)
        train_losses.append(train_loss)

        # evaluate performance of current model
        if test_loader is not None:
            test_loss, test_acc, test_pre, test_recall, test_f1\
                = test(test_loader, model, criterion, args.cuda, num_classes, args.dataset, is_attack)
            test_losses.append(test_loss)
            current = 0.0
            if norm == 'acc':
                current = test_acc
            elif norm == 'pre':  # precision of membership inference attack
                current = test_pre
            if debug:
                logging.debug('test loss: {}, test acc: {}, test pre: {}, test recall: {}, test f1: {}'.
                             format(test_loss, test_acc, test_pre, test_recall, test_f1))

            # judge whether current model is best according to norm
            save_model = False
            best_temp = max(current, best)
            if best_temp == current:
                if best_temp != best:
                    save_model = True
                else:
                    if norm == 'acc':
                        if pre < test_pre:
                            save_model = True
                    elif norm == 'pre':
                        if acc < test_acc:
                            save_model = True
                best = best_temp

            # save best model performance
            if save_model:
                acc, pre, recall, f1 = test_acc, test_pre, test_recall, test_f1
                # save best model parameters
                if args.save_model:
                    save_checkpoint({
                        'state_dict': model.state_dict()
                    }, False, filename=model_type, checkpoint=checkpoint_file)

    if test_loader is not None:
        logging.debug('Best {}: {}'.format(norm, best))
    elif args.save_model:
        save_checkpoint({
            'state_dict': model.state_dict()
        }, False, filename=model_type, checkpoint=checkpoint_file)

    if dp:
        return parameters_list
    else:
        return acc, pre, recall, f1


def active_train_model(model, train_loader, test_loader, args):
    """
    local attacker performs active attack, not used by CS-MIA
    refers to Gradient Ascent Attacker in “Comprehensive Privacy Analysis of Deep Learning: Passive and Active White-box Inference Attacks against Centralized and Federated Learning”

    :param model: model
    :param train_loader: loader of test members
    :param test_loader: loader of test non-members
    :param args: configuration
    """
    if args.cuda:
        model = model.cuda()

    # modify batch size of train and test loader, process all data in one batch
    train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=len(train_loader.dataset), shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_loader.dataset, batch_size=len(test_loader.dataset), shuffle=True)

    learning_rate = args.target_learning_rate
    optimizer = get_optimizer(model, learning_rate, args)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(0, 1):
        # gradient ascent on test members
        active_train(train_loader, model, criterion, optimizer, args.cuda)
        # gradient ascent on test non-members
        active_train(test_loader, model, criterion, optimizer, args.cuda)


def active_train(train_loader, model, criterion, optimizer, use_cuda):
    """
    gradient ascent, used by active local attacker

    :param train_loader: loader of dataset
    :param model: model
    :param criterion: loss function of model
    :param optimizer: optimizer of model
    :param use_cuda: whether to use cuda
    """
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        # compute output
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            outputs, _ = outputs
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        # modify sign of gradients to perform gradient ascent
        for p in model.parameters():
            p.grad = -p.grad

        optimizer.step()
    return


def train_for_gradient(
        model,
        train_loader=None,
        args=None,
        dp=False):
    """
    train model and get gradients for FL participant, used by FedSGD

    :param model: model
    :param train_loader: loader of train dataset for FL participant
    :param args: configuration
    :param dp: whether to use DP defense, not used
    :return: model gradients after training
    """
    if args.cuda:
        model = model.cuda()

    if not dp:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(reduction='sum')

    # modify batch size of train loader, train all data in one batch, required in FedSGD
    train_dataset = train_loader.dataset
    one_batch_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)

    grad_dict = dict()
    model.train()
    for batch_idx, (inputs, targets) in enumerate(one_batch_train_loader):
        if args.cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
    for name, param in model.named_parameters():
        grad_dict[name] = param.grad.detach().clone()
    # torch.cuda.empty_cache()
    return grad_dict


def test_model(model, test_loader, args, num_classes):
    """
    evaluate model performance on test dataset

    :param model: model
    :param test_loader: loader of test dataset
    :param args: configuration
    :param num_classes: number of dataset classes
    :return: loss, accuracy, precision, recall, and F1-score
    """
    criterion = nn.CrossEntropyLoss()
    if args.cuda:
        model = model.cuda()
    test_loss, test_acc, test_pre, test_recall, test_f1\
        = test(test_loader, model, criterion, args.cuda, num_classes, args.dataset)
    return test_loss, test_acc, test_pre, test_recall, test_f1


def train(train_loader, model, criterion, optimizer, use_cuda, is_attack=False):
    """
    train model in one epoch

    :param train_loader: loader of train dataset
    :param model: model
    :param criterion: loss function of model
    :param optimizer: optimizer of model
    :param use_cuda: whether to use cuda
    :param is_attack: whether model is attack model
    :return: model loss and accuracy in current epoch
    """
    # switch to train mode
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        # process output of attack model for computing BCELoss
        if is_attack:
            outputs = torch.squeeze(model(inputs))
            targets = targets.to(torch.float32)
        else:
            outputs = model(inputs)
            # get output of target model if use adversarial regularization defense
            if isinstance(outputs, tuple):
                outputs, _ = outputs
        loss = criterion(outputs, targets)
        if is_attack:
            acc = attack_accuracy(outputs.data, targets.data)
        else:
            acc = accuracy(outputs.data, targets.data)[0]
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc, inputs.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return losses.avg, top1.avg


def gaussian_noise(data_shape, sigma, cuda):
    """
    Gaussian noise for CDP-FedAVG-LS Algorithm, not used
    """
    result = torch.normal(0, sigma, data_shape)
    if cuda:
        result = result.cuda()
    return result


def test(test_loader, model, criterion, use_cuda, num_classes, dataset, is_attack=False):
    """
    evaluate model performance of test dataset

    :param test_loader: loader of test dataset
    :param model: model
    :param criterion: loss function of model
    :param use_cuda: whether to use cuda
    :param num_classes: number of dataset classes
    :param dataset: dataset name, not used
    :param is_attack: whether model is attack model
    :return: loss, accuracy, precision, recall, and F1-score
    """
    losses = AverageMeter()
    # switch to evaluate mode
    model.eval()

    labels = list(range(num_classes))

    y_predict = []
    y_true = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                inputs = torch.autograd.Variable(inputs)
            targets = torch.autograd.Variable(targets)
            # process output of attack model for computing BCELoss
            if is_attack:
                outputs = torch.squeeze(model(inputs))
                targets = targets.to(torch.float32)
            else:
                outputs = model(inputs)
                # get output of target model if use adversarial regularization defense
                if isinstance(outputs, tuple):
                    outputs, _ = outputs
            loss = criterion(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            y_true += targets.data.tolist()
            # label member or non-member by comparing whether model output probability is greater than 0.5
            if is_attack:
                y_predict += (torch.where(outputs > 0.5, torch.ones_like(outputs), torch.zeros_like(outputs))).tolist()
            # pick the class with the largest probability as the prediction label
            else:
                _, predicted = torch.max(outputs.data, 1)
                y_predict += predicted.tolist()
    if is_attack:
        recall = recall_score(y_true, y_predict)
        precision = precision_score(y_true, y_predict)
        F1 = f1_score(y_true, y_predict)
        acc = accuracy_score(y_true, y_predict)
        # tn, fp, fn, tp = confusion_matrix(y_true, y_predict).ravel()
        # print('tn: {}, fp: {}, fn: {}, tp: {}'.format(tn, fp, fn, tp))
    else:
        recall = recall_score(y_true, y_predict, labels=labels, average='macro')
        precision = precision_score(y_true, y_predict, labels=labels, average='macro')
        F1 = f1_score(y_true, y_predict, labels=labels, average='macro')
        acc = accuracy_score(y_true, y_predict)
    # 精度调整
    recall = (recall * 100).round(3)
    precision = (precision * 100).round(3)
    F1 = (F1 * 100).round(3)
    acc = (acc * 100).round(3)
    if is_attack:
        classification_report(y_true, y_predict)
    return losses.avg, acc, precision, recall, F1


def save_checkpoint(model_state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    """
    save model to local file

    :param model_state: model parameters
    :param is_best: not used
    :param checkpoint: local file path
    :param filename: local file name
    :return:
    """
    filepath = os.path.join(checkpoint, filename)
    torch.save(model_state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
