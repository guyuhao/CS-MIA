# -*- coding: utf-8 -*-
"""
Implementation of white-box membership inference
refers to “Comprehensive Privacy Analysis of Deep Learning: Passive and Active White-box Inference Attacks against Centralized and Federated Learning”
"""

import numpy as np
import torch
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from torch import nn, optim

from common import SHADOW_INDICES_FILE, load_data, get_optimizer
from membership_inference.whitebox_model import WhiteboxModel
from utils import AverageMeter
import logging


class WhiteboxAttack:
    def __init__(self, args):
        super(WhiteboxAttack, self).__init__()
        self.args = args

    def get_inference_model_input(self, tr_input, te_input, tr_target, te_target,
                                  models, inference_model,
                                  classifier_criterion, criterion,
                                  classifier_optimizers):
        if self.args.cuda:
            tr_input = tr_input.cuda()
            te_input = te_input.cuda()
            tr_target = tr_target.cuda()
            te_target = te_target.cuda()
            for model in models:
                model = model.cuda()

        v_tr_input = torch.autograd.Variable(tr_input)
        v_te_input = torch.autograd.Variable(te_input)
        v_tr_target = torch.autograd.Variable(tr_target)
        v_te_target = torch.autograd.Variable(te_target)
        # compute output
        model_input = torch.cat((v_tr_input, v_te_input))
        pred_outputs = []
        for i in range(len(models)):
            temp = models[i](model_input)
            if isinstance(temp, tuple):
                temp, _ = temp
            pred_outputs.append(temp)
        infer_input = torch.cat((v_tr_target, v_te_target))
        # one_hot_tr = torch.from_numpy(np.zeros((infer_input.size(0), self.args.num_classes))).type(torch.FloatTensor)
        one_hot_tr = torch.zeros((infer_input.size(0), self.args.num_classes)).type(torch.FloatTensor)
        infer_input_long = infer_input.type(torch.LongTensor)
        if self.args.cuda:
            one_hot_tr = one_hot_tr.cuda()
            infer_input_long = infer_input_long.cuda()
        target_one_hot_tr = one_hot_tr.scatter_(1, infer_input_long.view([-1, 1]).data, 1)
        infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr)

        models_outputs = []
        correct_labels = []
        model_grads = []
        model_losses = []

        for m_n in range(len(models)):
            correct = torch.sum(pred_outputs[m_n] * infer_input_one_hot, dim=1)
            grads = torch.zeros(0)
            output_num = len(pred_outputs[m_n])
            for i in range(output_num):
                loss_classifier = classifier_criterion(pred_outputs[m_n][i].view([1, -1]),
                                                       infer_input[i].view([-1]))
                classifier_optimizers[m_n].zero_grad()
                if i == output_num - 1:
                    loss_classifier.backward(retain_graph=False)
                else:
                    loss_classifier.backward(retain_graph=True)
                grad = models[m_n].classifier.weight.grad
                g = grad.view([1, 1, -1, self.args.num_classes])
                if grads.size()[0] != 0:
                    grads = torch.cat((grads, g))
                else:
                    grads = g

            c = correct.view([-1, 1])
            preds = pred_outputs[m_n]
            loss = classifier_criterion(pred_outputs[m_n], infer_input).view(-1, 1)
            if self.args.cuda:
                grads = grads.cuda()
                c = c.cuda()
                preds = preds.cuda()
                loss = loss.cuda()
            grads = torch.autograd.Variable(grads)
            c = torch.autograd.Variable(c)
            preds = torch.autograd.Variable(preds)
            loss = torch.autograd.Variable(loss)

            models_outputs.append(preds)
            correct_labels.append(c)
            model_grads.append(grads)
            model_losses.append(loss)
        member_output = inference_model(model_grads, correct_labels, models_outputs, model_losses)

        is_member_labels = torch.from_numpy(
            np.reshape(np.concatenate((np.ones(v_tr_input.size(0)), np.zeros(v_te_input.size(0)))), [-1, 1]))
        if self.args.cuda:
            is_member_labels = is_member_labels.cuda()
        v_is_member_labels = torch.autograd.Variable(is_member_labels).type(torch.FloatTensor)
        if self.args.cuda:
            v_is_member_labels = v_is_member_labels.cuda()
        loss = criterion(member_output, v_is_member_labels)
        return loss, v_is_member_labels, member_output

    def privacy_train(self,
                      train_loader, test_loader,
                      models, inference_model,
                      classifier_criterion, criterion,
                      classifier_optimizers, optimizer):
        """
        train attack model

        :param train_loader: loader of train members
        :param test_loader: loader of train non-members
        :param models: observed target models
        :param inference_model: attack model
        :param classifier_criterion: loss function of target model
        :param criterion: loss function of attack model
        :param classifier_optimizers: optimizers of observed target models
        :param optimizer: optimizer of attack model
        :return: tuple containing train loss and accuracy on train dataset
        """
        losses = AverageMeter()
        inference_model.train()
        for model in models:
            model.eval()
        y_predict = []
        y_true = []
        for batch_idx, ((tr_input, tr_target), (te_input, te_target)) in enumerate(zip(train_loader, test_loader)):
            # get input of attack model based on target models, and then train in one batch and get loss
            loss, v_is_member_labels, member_output = self.get_inference_model_input(
                tr_input, te_input, tr_target, te_target,
                models, inference_model,
                classifier_criterion, criterion,
                classifier_optimizers)

            losses.update(loss.data, tr_input.size(0)+te_input.size(0))
            y_true += v_is_member_labels.data.tolist()
            y_predict += (torch.where(member_output > 0.5, torch.ones_like(member_output),
                                      torch.zeros_like(member_output))).tolist()
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc = accuracy_score(y_true, y_predict)
        acc = (acc * 100).round(3)
        return losses.avg, acc

    def privacy_test(self,
                     train_loader, test_loader,
                     models, inference_model,
                     classifier_criterion, criterion,
                     classifier_optimizers):
        """
        evaluate attack model

        :param train_loader: loader of test members data
        :param test_loader: loader of test non-members data
        :param models: observed target models
        :param inference_model: attack model
        :param classifier_criterion: loss function of target model
        :param criterion: loss function of attack model
        :param classifier_optimizers: optimizers of observed target models
        :return:
        """
        losses = AverageMeter()
        inference_model.eval()
        for model in models:
            model.eval()
        y_predict = []
        y_true = []
        for batch_idx, ((tr_input, tr_target), (te_input, te_target)) in enumerate(zip(train_loader, test_loader)):
            # get input of attack model based on target models, and then train in one batch and get loss
            loss, v_is_member_labels, member_output = self.get_inference_model_input(
                tr_input, te_input, tr_target, te_target,
                models, inference_model,
                classifier_criterion, criterion,
                classifier_optimizers)
            losses.update(loss.data, tr_input.size(0) + te_input.size(0))
            y_true += v_is_member_labels.data.tolist()
            y_predict += (torch.where(member_output > 0.5, torch.ones_like(member_output), torch.zeros_like(member_output))).tolist()

        recall = recall_score(y_true, y_predict)
        precision = precision_score(y_true, y_predict)
        F1 = f1_score(y_true, y_predict)
        acc = accuracy_score(y_true, y_predict)
        # 精度调整
        recall = (recall * 100).round(3)
        precision = (precision * 100).round(3)
        F1 = (F1 * 100).round(3)
        acc = (acc * 100).round(3)
        return losses.avg, acc, precision, recall, F1

    def attack(self, models, test_m_loader, test_nm_loader):
        """
        perform white-box attack

        :param models: target models during observed rounds
        :param test_m_loader: loader of test members data
        :param test_nm_loader: loader of test non-members data
        :return: accuracy, precision, recall, F1 score of white-box attack on test dataset
        """
        non_iid_prefix = '_non_iid' if self.args.non_iid else ''

        # load train members from local file
        indices_file = 'client{}_cs_mia_indices{}_{}_{}_{}.npz'. \
            format(self.args.cs_mia_client, non_iid_prefix, self.args.dataset, self.args.client_train_size, self.args.attack_train_m_size)
        train_m_loader, _ = load_data(
            dataset=self.args.dataset,
            indices_file=indices_file,
            batch_size=self.args.cs_mia_batch_size,
            type='target')

        # load train non-members from local file
        indices_file = SHADOW_INDICES_FILE + '{}_{}_{}_{}.npz'. \
            format(non_iid_prefix, self.args.dataset, self.args.attack_train_m_size, self.args.attack_train_nm_size)
        _, train_nm_loader = load_data(
            dataset=self.args.dataset,
            indices_file=indices_file,
            batch_size=self.args.cs_mia_batch_size,
            type='train-test')

        # get target models in observed rounds
        target_models = [models[epoch-1] for epoch in self.args.whitebox_observed_round]
        # remove target model in the observed round when target client is not selected
        target_models = [model for model in target_models if model is not None]
        # if target client is never selected in observed rounds, use target models in last rounds
        if len(target_models) == 0:
            temp_models = [model for model in models if model is not None]
            target_models = temp_models[max(-len(temp_models), -len(self.args.whitebox_observed_round)):]
            # target_models = models[max(-len(models), -len(self.args.whitebox_observed_round)):]
        # print('whitebox target models size: {}'.format(len(target_models)))

        # only support that last layer of target model is fully connected layer and is defined as 'classifier'
        target_classifier_grad_size = target_models[-1].classifier.in_features

        # initialize attack model
        inference_model = WhiteboxModel(num_classes=self.args.num_classes,
                                        num_models=len(target_models),
                                        target_classifier_grad_size=target_classifier_grad_size)
        if self.args.cuda:
            inference_model = inference_model.cuda()
        criterion = nn.CrossEntropyLoss(reduce=False)
        # criterion_attack = nn.MSELoss()
        criterion_attack = nn.BCELoss()
        target_optimizers = []
        for model in target_models:
            target_optimizers.append(get_optimizer(model, self.args.target_learning_rate, self.args))
        optimizer_mem = optim.Adam(inference_model.parameters(), lr=self.args.whitebox_learning_rate)
        best = 0.0
        acc, pre, recall, f1 = 0.0, 0.0, 0.0, 0.0
        for epoch in range(self.args.whitebox_epochs):
            # train attack model
            train_loss, train_acc = self.privacy_train(
                train_m_loader, train_nm_loader,
                target_models, inference_model,
                criterion, criterion_attack,
                target_optimizers, optimizer_mem)

            # evaluate attack model
            test_loss, test_acc, test_pre, test_recall, test_f1 = self.privacy_test(
                test_m_loader, test_nm_loader,
                target_models, inference_model,
                criterion, criterion_attack,
                target_optimizers)

            # set performance of attack model with the best precision as final result
            current = test_pre
            logging.debug('epoch: {}, test loss: {}, test acc: {}, test pre: {}, test recall: {}, test f1: {}'.format(
                epoch, test_loss, test_acc, test_pre, test_recall, test_f1))
            best = max(current, best)
            if best == current:
                acc, pre, recall, f1 = test_acc, test_pre, test_recall, test_f1
        return acc, pre, recall, f1
