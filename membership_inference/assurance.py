# -*- coding: utf-8 -*-
"""
Implementation of computation of prediction confidence
"""

import numpy as np
import torch
from torch import nn


def cal_m_entr_assurance(model, data_loader, num_classes, cuda):
    """
    computation of Mentr prediction confidence, refers to "Systematic Evaluation of Privacy Risks of Machine Learning Models"

    :param model: model
    :param data_loader: loader of dataset to compute confidence
    :param num_classes: number of dataset classes
    :param cuda: whether to use cuda
    :return: Mentr confidence
    """
    def softmax_by_row(logits, T=1.0):
        mx = np.max(logits, axis=-1, keepdims=True)
        exp = np.exp((logits - mx) / T)
        denominator = np.sum(exp, axis=-1, keepdims=True)
        return exp / denominator

    def _log_value(probs, small_value=1e-30):
        return -np.log(np.maximum(probs, small_value))

    def _model_predictions(model, dataloader):
        if cuda:
            model = model.cuda()
        model.eval()
        with torch.no_grad():
            return_outputs, return_labels = [], []
            for (inputs, labels) in dataloader:
                return_labels.append(labels.numpy())
                if cuda:
                    inputs = inputs.cuda()
                outputs = model.forward(inputs)
                if isinstance(outputs, tuple):
                    outputs, _ = outputs
                return_outputs.append(softmax_by_row(outputs.data.cpu().numpy()))
            return_outputs = np.concatenate(return_outputs)
            return_labels = np.concatenate(return_labels)
        model = model.cpu()
        return return_outputs, return_labels

    probs, true_labels = _model_predictions(model, data_loader)
    log_probs = _log_value(probs)
    reverse_probs = 1-probs
    log_reverse_probs = _log_value(reverse_probs)
    modified_probs = np.copy(probs)
    modified_probs[range(true_labels.size), true_labels] = reverse_probs[range(true_labels.size), true_labels]
    modified_log_probs = np.copy(log_reverse_probs)
    modified_log_probs[range(true_labels.size), true_labels] = log_probs[range(true_labels.size), true_labels]
    result = np.sum(np.multiply(modified_probs, modified_log_probs), axis=1)
    result = -torch.tensor(result)
    return result


def cal_conf_assurance(model, data_loader, num_classes, cuda):
    """
    computation of Prediction probability of ground-truth label,
    refers to "Privacy Risks of Securing Machine Learning Models against Adversarial Examples"

    :param model: model
    :param data_loader: loader of dataset to compute confidence
    :param num_classes: number of dataset classes
    :param cuda: whether to use cuda
    :return: Pr confidence
    """
    final_result = []
    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(data_loader):
            if cuda:
                inputs = inputs.cuda()
                model = model.cuda()
            predictions = model(inputs)
            # log_softmax = nn.LogSoftmax(dim=1)
            log_softmax = nn.Softmax(dim=1)
            predictions = log_softmax(predictions)
            result = torch.zeros(predictions.shape[0], dtype=torch.float32)
            for index, temp in enumerate(predictions):
                result[index] = temp[targets[index]]
            final_result += result.numpy().tolist()
    return torch.Tensor(final_result)

