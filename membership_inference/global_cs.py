# -*- coding: utf-8 -*-
"""
Implementation of global CS-MIA
"""

import copy

import torch.nn

from common.data_process import *
from common.model_process import *
from .assurance import cal_m_entr_assurance, cal_conf_assurance
from .attack_model import *
from .mia_dataset import *


def train_attack_model(train_dataset, test_dataset, args):
    """
    train and evaluate attack model of global CS-MIA

    :param train_dataset: train dataset for attack model
    :param test_dataset: test dataset for attack model
    :param args: configuration
    :return: accuracy, precision, recall, F1 score of CS-MIA on test dataset
    """
    # generate loader of train and test dataset for attack model
    train_x, train_y = train_dataset
    test_x, test_y = test_dataset
    train_mia_dataset = MiaDataset(train_x, train_y)
    test_mia_dataset = MiaDataset(test_x, test_y)
    train_loader = torch.utils.data.DataLoader(train_mia_dataset, batch_size=args.cs_mia_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_mia_dataset, batch_size=args.cs_mia_batch_size, shuffle=True)

    # initialize attack model
    n_in = train_x.shape[1]
    model = attack_model(
        n_in=n_in,
        n_hidden=args.cs_mia_n_hidden)

    # train and evaluate attack model
    acc, pre, recall, f1 = train_model(
        model=model,
        model_type='cs_mia',
        train_loader=train_loader,
        test_loader=test_loader,
        args=args,
        norm='acc',
        debug=False
    )
    return acc, pre, recall, f1


class GlobalCS:
    def __init__(self, assurance_type='mentr'):
        super(GlobalCS, self).__init__()
        self.attack_train_x = None
        self.attack_train_y = None
        self.assurance_type = assurance_type
        self.model_list = list()

    def get_train_dataset(self, models, args):
        """
        get train dataset for attack model, and save in attack_train_x and attack_train_y

        :param models: malicious server's local models during round
        :param args: configuration
        """
        x, y = None, None
        non_iid_prefix = '_non_iid' if args.non_iid else ''
        file = DATA_PATH + 'cs_mia_server_train_data.npz'

        # load train dataset from local file directly
        if args.load_data and os.path.isfile(file):
            temp = np.load(file)
            x = torch.from_numpy(temp['attack_x'])
            y = torch.from_numpy(temp['attack_y'])
            if args.cuda:
                x = x.cuda()
                y = y.cuda()
        # generate train dataset based on local models
        else:
            # load train members and non-members data
            indices_file = SHADOW_INDICES_FILE + '{}_{}_{}_{}.npz'.format(
                non_iid_prefix, args.dataset, args.attack_train_m_size, args.attack_train_nm_size)
            train_loader, test_loader = load_data(
                dataset=args.dataset,
                indices_file=indices_file,
                batch_size=args.target_batch_size,
                type='train-test')

            # use local models to generate train dataset for attack model based on train members and non-members
            # feature is confidence series, label is membership information (0 or 1)
            x, y = self.get_dataset(models, train_loader, test_loader, args)
            if args.save_data:
                np.savez(DATA_PATH + 'cs_mia_server_train_data.npz',
                         attack_x=x.detach(),
                         attack_y=y.detach())
        self.attack_train_x = x
        self.attack_train_y = y


    def get_dataset(self, models, train_loader, test_loader, args):
        """
        generate dataset for attack model, whose feature is confidence series, label is membership information (0 or 1)

        :param models: malicious attacker's local models during rounds for train dataset, or target client's local models for test dataset
        :param train_loader: loader of members for target model
        :param test_loader: loader of non-members for target model
        :param args: configuration
        :return: tuple containing features and labels
        """
        # decide how to compute prediction confidence
        attack_train_x = []
        cal_assurance_function = None
        if self.assurance_type == 'mentr':
            cal_assurance_function = cal_m_entr_assurance
        elif self.assurance_type == 'conf':
            cal_assurance_function = cal_conf_assurance

        # compute confidence series of members
        for model in models:
            if isinstance(model, dict):
                model = model['model']
            assurance = cal_assurance_function(model, train_loader, args.num_classes, args.cuda)
            attack_train_x.append(assurance.cpu().detach().numpy().tolist())
        attack_train_x = torch.tensor(attack_train_x).t()
        # assign confidence series of members with label 1
        attack_train_y = torch.ones(attack_train_x.shape[0], dtype=torch.long)

        # compute confidence series of non-members
        attack_test_x = []
        for model in models:
            if isinstance(model, dict):
                model = model['model']
            assurance = cal_assurance_function(model, test_loader, args.num_classes, args.cuda)
            attack_test_x.append(assurance.cpu().detach().numpy().tolist())
        attack_test_x = torch.tensor(attack_test_x).t()
        # assign confidence series of non-members with label 0
        attack_test_y = torch.zeros(attack_test_x.shape[0], dtype=torch.long)

        # combine members and non-members as dataset for attack model
        x = torch.cat((attack_train_x, attack_test_x), 0)
        y = torch.cat((attack_train_y, attack_test_y), 0)
        return x, y

    def get_test_dataset(self, models, train_loader, test_loader, args):
        """
        get test dataset for attack model

        :param models: target client's local models during rounds
        :param train_loader: loader of test members data
        :param test_loader: loader of test non-members data
        :param args: configuration
        :return: tuple containing features and labels of test dataset for attack model
        """
        x, y = None, None
        file = DATA_PATH + 'cs_mia_server_test_data.npz'
        # load test dataset from local file directly
        if args.load_data and os.path.isfile(file):
            temp = np.load(file)
            x = torch.from_numpy(temp['attack_x'])
            y = torch.from_numpy(temp['attack_y'])
            if args.cuda:
                x = x.cuda()
                y = y.cuda()
        # generate test dataset based on local models
        else:
            x, y = self.get_dataset(models, train_loader, test_loader, args)
            if args.save_data:
                np.savez(DATA_PATH + 'cs_mia_server_test_data.npz',
                         attack_x=x.detach(),
                         attack_y=y.detach())
        return x, y

    def attack(self, test_dataset, args):
        """
        perform global CS-MIA

        :param test_dataset: test dataset for attack model
        :param args: configuration
        :return: accuracy, precision, recall, F1 score of CS-MIA on test dataset
        """
        logging.debug("Training the attack model...")
        train_dataset = (self.attack_train_x, self.attack_train_y)
        logging.debug('*'*15 + 'membership attack testing' + '*'*15)
        acc, pre, recall, f1 = train_attack_model(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            args=args)
        return acc, pre, recall, f1
