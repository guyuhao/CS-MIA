# -*- coding: utf-8 -*-
"""
Implementation of ML-Leaks attack
refers to “ML-Leaks: Model and Data Independent Membership Inference Attacks and Defenses on Machine Learning Models”
"""

import copy

import torch.nn

from common.data_process import *
from common.model_process import *
from .mia_dataset import *
from .ml_leaks_model import ml_leaks_model


def train_attack_model(train_dataset, test_dataset, args):
    """
    train and evaluate attack model of ML-Leaks

    :param train_dataset: train dataset for attack model
    :param test_dataset: test dataset for attack model
    :param args: configuration
    :return: accuracy, precision, recall, F1 score of ML-Leaks on test dataset
    """
    train_x, train_y = train_dataset
    test_x, test_y = test_dataset

    # pick top-3 high probability of model output as features for attack model
    top = 3
    softmax = nn.Softmax(dim=1)
    train_sort, _ = torch.sort(softmax(train_x), descending=True)
    test_sort, _ = torch.sort(softmax(test_x), descending=True)
    train_top_k = train_sort[:, :top].clone()
    test_top_k = test_sort[:, :top].clone()
    # generate loader of train and test dataset for attack model
    train_mia_dataset = MiaDataset(train_top_k, train_y)
    test_mia_dataset = MiaDataset(test_top_k, test_y)
    train_loader = torch.utils.data.DataLoader(train_mia_dataset, batch_size=args.ml_leaks_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_mia_dataset, batch_size=args.ml_leaks_batch_size, shuffle=True)

    # initialize attack model
    n_in = train_top_k.shape[1]
    n_out = len(np.unique(train_y))
    model = ml_leaks_model(
        n_in=n_in,
        n_hidden=args.ml_leaks_n_hidden,
        n_out=n_out)

    # train and evaluate attack model
    acc, pre, recall, f1 = train_model(
        model=model,
        model_type='ml_leaks',
        train_loader=train_loader,
        test_loader=test_loader,
        args=args,
        norm='acc')
    return acc, pre, recall, f1


def train_shadow_model(target_model, train_loader, test_loader, args):
    """
    train shadow model with the same architecture as target model, and save it in local file

    :param target_model: attacker's target model
    :param train_loader: loader of train dataset for shadow model
    :param test_loader: not used
    :param args: configuration
    """
    model = copy.deepcopy(target_model)
    model.zero_grad()
    save_model = args.save_model
    args.save_model = True
    train_model(model=model,
                model_type='shadow_ml_leaks',
                train_loader=train_loader,
                test_loader=test_loader,
                args=args,
                load=args.load_shadow)
    args.save_model = save_model


class MLLeaks:
    def __init__(self):
        super(MLLeaks, self).__init__()
        self.attack_train_x = None
        self.attack_train_y = None

    def train_shadow_models(self, target_model, args):
        """
        train shadow model to get train dataset for attack model, and save in attack_train_x and attack_train_y

        :param target_model: attacker's target model
        :param args: configuration
        """
        non_iid_prefix = '_non_iid' if args.non_iid else ''
        attack_x, attack_y = [], []

        # load auxiliary dataset for shadow model from local file
        indices_file = SHADOW_INDICES_FILE + '{}_{}_{}_{}.npz'.format(
            non_iid_prefix, args.dataset, args.attack_train_m_size, args.attack_train_nm_size)
        train_loader, test_loader = load_data(
            dataset=args.dataset,
            indices_file=indices_file,
            batch_size=args.target_batch_size,
            type='train-test')

        if args.load_shadow == 0:
            # train shadow model with the same architecture as target model
            # save trained shadow model into local model
            logging.debug('Training shadow model')
            train_shadow_model(target_model, train_loader, None, args)

        model = copy.deepcopy(target_model)
        model.zero_grad()
        if args.cuda:
            model = model.cuda()
        # load trained shadow model from local file
        train_model(model=model,
                    model_type='shadow_ml_leaks',
                    train_loader=train_loader,
                    test_loader=None,
                    args=args,
                    load=True)

        self.attack_train_x = torch.cat(attack_x, dim=0)
        self.attack_train_y = torch.cat(attack_y)

    def get_attack_dataset(self, target_model, train_loader, test_loader, args):
        """
        generate dataset for attack model, whose feature is confidence series, label is membership information (0 or 1)

        :param models: malicious attacker's local models during rounds for train dataset, or target client's local models for test dataset
        :param train_loader: loader of members for target model
        :param test_loader: loader of non-members for target model
        :param args: configuration
        :return: tuple containing features and labels
        """
        attack_x, attack_y = [], []
        with torch.no_grad():
            # compute shadow model output on test members, and label 1
            for i, (train_x, train_y) in enumerate(train_loader):
                if args.cuda:
                    train_x, train_y = train_x.cuda(), train_y.cuda()
                temp = target_model(train_x)
                if isinstance(temp, tuple):
                    temp, _ = temp
                attack_x.append(temp)
                attack_y.append(torch.ones(temp.shape[0], dtype=torch.long))

            # compute shadow model output on test non-members, and label 0
            for i, (test_x, test_y) in enumerate(test_loader):
                if args.cuda:
                    test_x, test_y = test_x.cuda(), test_y.cuda()
                temp = target_model(test_x)
                if isinstance(temp, tuple):
                    temp, _ = temp
                attack_x.append(temp)
                attack_y.append(torch.zeros(temp.shape[0], dtype=torch.long))

            attack_x = torch.cat(attack_x, dim=0)
            attack_y = torch.cat(attack_y)
        return attack_x, attack_y

    def get_test_data(self, target_model, train_loader, test_loader, args):
        """
        get test dataset for attack model

        :param target_model: attacker's target model
        :param train_loader: loader of test members data
        :param test_loader: loader of test non-members data
        :param args: configuration
        :return: tuple containing features and labels of test dataset for attack model
        """
        if args.cuda:
            target_model = target_model.cuda()
        attack_x, attack_y = self.get_attack_dataset(target_model, train_loader, test_loader, args)

        if args.save_data:
            np.savez(DATA_PATH + 'ml_leaks_attack_test_data.npz',
                     attack_x=attack_x.cpu().detach(),
                     attack_y=attack_y.cpu().detach())
        return attack_x, attack_y

    def attack(self, test_dataset, args):
        """
        perform ML-Leaks attack

        :param test_dataset: test dataset for attack model
        :param args: configuration
        :return: accuracy, precision, recall, F1 score of ML-Leaks on test dataset
        """
        train_dataset = (self.attack_train_x, self.attack_train_y)
        logging.debug('*'*15 + 'membership attack testing' + '*'*15)
        acc, pre, recall, f1 = train_attack_model(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            args=args)
        return acc, pre, recall, f1
