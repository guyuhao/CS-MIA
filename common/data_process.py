# -*- coding: utf-8 -*-
"""
Implementation of dataset loading
"""

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler

from dataset.text_dataset import TextDataset
from .constants import *

dataset_path = '../../data'
train_dataset = None
test_dataset = None
current_dataset = None


def load_data(dataset, indices_file, batch_size, type=None):
    """
    get loader of train and test dataset according to indices saved in local file

    :param dataset: dataset name
    :param indices_file: the file path that saves indices
    :param batch_size: batch size to load data
    :param type: 'target' or 'train-test', 'target' means that local file only saves indices of train dataset,
                 'train-test' means that local file saved indices of train and test dataset
    :return: tuple containing loader of train and test dataset, test loader is None when type is 'target'
    """
    train_loader, test_loader = None, None
    get_dataset(dataset)

    if indices_file is None:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    else:
        # 目标模型数据集
        if type == 'target':
            data_indices = np.load(DATA_PATH + dataset + '/' + indices_file)['target_indices']
            sub_train_dataset = torch.utils.data.Subset(train_dataset, data_indices)
            train_loader = torch.utils.data.DataLoader(sub_train_dataset, batch_size=batch_size, shuffle=True)
        if type == 'train-test':
            data_indices = np.load(DATA_PATH + dataset + '/' + indices_file)
            train_indices = data_indices['train_indices']
            test_indices = data_indices['test_indices']
            if len(train_indices) > 0:
                sub_train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
                train_loader = torch.utils.data.DataLoader(sub_train_dataset, batch_size=batch_size, shuffle=True)
            if len(test_indices) > 0:
                sub_test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
                test_loader = torch.utils.data.DataLoader(sub_test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def load_data_from_indices(dataset, train_indices, test_indices, batch_size):
    """
    get loader of train and test dataset according to indices

    :param dataset: dataset name
    :param train_indices: indices of train dataset
    :param test_indices: indices of test dataset
    :param batch_size: batch size to load data
    :return: tuple containing: train loader and test loader
    """
    # get dataset according to name
    get_dataset(dataset)

    train_loader, test_loader = None, None
    if train_indices is not None:
        sub_train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        train_loader = torch.utils.data.DataLoader(sub_train_dataset, batch_size=batch_size, shuffle=True)
    if test_indices is not None:
        sub_test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
        test_loader = torch.utils.data.DataLoader(sub_test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_dataset_labels(dataset, name):
    """
    get labels of dataset

    :param dataset: dataset
    :param name: dataset name
    :return: labels
    """
    if 'cifar' in name or name in ['mnist']:
        return dataset.targets
    elif name in ['purchase100', 'texas100']:
        return dataset.y


def get_dataset(name):
    """
    get train and test dataset based on dataset name

    :param name: dataset name
    :return: tuple containing: train dataset and test dataset
    """
    global train_dataset
    global test_dataset
    global current_dataset
    if name not in ['cifar10', 'cifar100', 'purchase100', 'texas100', 'mnist']:
        raise TypeError('dataset should be a string, including cifar10, cifar100, purchase100, texas100, mnist. ')

    # get CIFAR dataset
    if 'cifar' in name:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        if name == 'cifar10':
            if train_dataset is None or name != current_dataset:
                train_dataset = torchvision.datasets.CIFAR10(root=dataset_path, train=True,
                                                             download=True, transform=transform)
            if test_dataset is None or name != current_dataset:
                test_dataset = torchvision.datasets.CIFAR10(root=dataset_path, train=False,
                                                            download=True, transform=transform)
        elif name == 'cifar100':
            if train_dataset is None or name != current_dataset:
                train_dataset = torchvision.datasets.CIFAR100(root=dataset_path, train=True,
                                                              download=True, transform=transform)
            if test_dataset is None or name != current_dataset:
                test_dataset = torchvision.datasets.CIFAR100(root=dataset_path, train=False,
                                                             download=True, transform=transform)
    # get feature dataset
    elif name in ['purchase100', 'texas100']:
        function = None
        if name == 'purchase100':
            function = get_purchase_data
        elif name == 'texas100':
            function = get_texas100_data
        if train_dataset is None or name != current_dataset:
            train_x, train_y = function('train')
            train_dataset = TextDataset(train_x, train_y)
        if test_dataset is None or name != current_dataset:
            test_x, test_y = function('test')
            test_dataset = TextDataset(test_x, test_y)
    # get MNIST dataset
    elif name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        if train_dataset is None or name != current_dataset:
            train_dataset = torchvision.datasets.MNIST(root=dataset_path, train=True,
                                                       download=True, transform=transform)
        if test_dataset is None or name != current_dataset:
            test_dataset = torchvision.datasets.MNIST(root=dataset_path, train=False,
                                                      download=True, transform=transform)

    current_dataset = name
    return train_dataset, test_dataset


def get_purchase_data(name):
    """
    load train or test data of Purchase100 from local file

    :param name: 'train' or 'test'
    :return: tuple containing features and labels
    """
    data = pd.read_csv(dataset_path + '/purchase/dataset_purchase_' + name,
                       header=None,
                       skipinitialspace=True)
    x = data.drop(columns=[0]).values.astype('float32')
    y = np.squeeze(data.drop(columns=list(range(1, 601))).values.astype('int'))-1
    return x, y


def get_texas100_data(name):
    """
    load train or test data of Texas100 from local file

    :param name: 'train' or 'test'
    :return: tuple containing features and labels
    """
    x, y = list(), list()
    data = pd.read_csv(dataset_path + '/texas100/feats_' + name,
                       header=None,
                       skipinitialspace=True)
    x = data.values.astype('float32')
    with open(dataset_path + '/texas100/labels_' + name, "r") as f:
        inputs = f.readlines()
    for datapoint in inputs:
        y.append(int(datapoint)-1)
    return x, y
