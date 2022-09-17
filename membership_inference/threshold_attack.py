# -*- coding: utf-8 -*-
"""
Implementation of Threshold-based attack, refers to “Systematic Evaluation of Privacy Risks of Machine Learning Models”
"""

import torch.nn

from common.data_process import *
from common.model_process import *
from .assurance import cal_m_entr_assurance, cal_conf_assurance


class ThresholdAttack:
    def __init__(self, metric='mentr'):
        super(ThresholdAttack, self).__init__()
        self.shadow_train = None
        self.shadow_test = None
        self.cal_assurance_function = cal_conf_assurance if metric == 'conf' else cal_m_entr_assurance

    def _thre_setting(self, tr_values, te_values):
        """
        select one of the computed confidences which achieves the highest attack accuracy on shadow dataset as threshold

        :param tr_values: confidences of train members
        :param te_values: confidences of train non-members
        :return: threshold
        """
        value_list = np.concatenate((tr_values, te_values))
        thre, max_acc = 0, 0
        for value in value_list:
            tr_ratio = np.sum(tr_values >= value)/(len(tr_values)+0.0)
            te_ratio = np.sum(te_values < value)/(len(te_values)+0.0)
            acc = 0.5*(tr_ratio + te_ratio)
            if acc > max_acc:
                thre, max_acc = value, acc
        return thre

    def get_train_dataset(self, model, args):
        """
        get confidences of train members and non-members dataset

        :param model: attacker's target model, global model in local attack,
                        or target client's local model in global attack (impractical for global attacker, but achieve better attack accuracy)
        :param args: configuration
        :return: tuple containing confidences of train members and non-members
        """
        non_iid_prefix = '_non_iid' if args.non_iid else ''
        # load train members from local file directly
        indices_file = 'client{}_cs_mia_indices{}_{}_{}_{}.npz'.format(
            args.cs_mia_client, non_iid_prefix, args.dataset, args.client_train_size, args.attack_train_m_size)
        train_loader, _ = load_data(dataset=args.dataset,
                                    indices_file=indices_file,
                                    batch_size=128,
                                    type='target')  # batch_size不影响结果

        # load train non-members from local file directly
        indices_file = SHADOW_INDICES_FILE + '{}_{}_{}_{}.npz'. \
            format(non_iid_prefix, args.dataset, args.attack_train_m_size, args.attack_train_nm_size)
        _, test_loader = load_data(dataset=args.dataset,
                                   indices_file=indices_file,
                                   batch_size=128,
                                   type='train-test')

        # compute confidences of train members
        train_x = self.cal_assurance_function(model, train_loader, args.num_classes, args.cuda)

        # compute confidences of train non-members
        test_x = self.cal_assurance_function(model, test_loader, args.num_classes, args.cuda)

        return train_x, test_x

    def get_test_dataset(self, model, train_loader, test_loader, args):
        """
        get test dataset for attack

        :param model: attacker's target model
        :param train_loader: loader of test members data
        :param test_loader: loader of test non-members data
        :param args: configuration
        :return: tuple containing features (confidences) and labels of test dataset for attack
        """
        attack_train_x = self.cal_assurance_function(model, train_loader, args.num_classes, args.cuda)
        attack_train_y = torch.ones(attack_train_x.shape[0], dtype=torch.long)
        attack_test_x = self.cal_assurance_function(model, test_loader, args.num_classes, args.cuda)
        attack_test_y = torch.zeros(attack_test_x.shape[0], dtype=torch.long)
        return torch.cat((attack_train_x, attack_test_x), 0), torch.cat((attack_train_y, attack_test_y), 0)

    def attack(self, train_dataset, test_dataset, args):
        """
        perform threshold-based attack

        :param train_dataset: train dataset for attack
        :param test_dataset: test dataset for attack
        :param args: configuration, not used
        :return: accuracy, precision, recall, F1 score of CS-MIA on test dataset
        """
        # select threshold from train dataset
        tr_values, te_values = train_dataset
        threshold = self._thre_setting(tr_values.cpu().numpy(), te_values.cpu().numpy())

        # infer membership by comparing confidence with threshold
        inputs, labels = test_dataset
        predictions = list()
        for temp in inputs:
            if temp >= threshold:
                predictions.append(1)
            else:
                predictions.append(0)

        acc = accuracy_score(labels, predictions)
        pre = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)
        f1 = f1_score(labels, predictions)

        recall = (recall * 100).round(3)
        pre = (pre * 100).round(3)
        f1 = (f1 * 100).round(3)
        acc = (acc * 100).round(3)
        return acc, pre, recall, f1
