from __future__ import print_function, absolute_import

__all__ = ['accuracy', 'attack_accuracy']

import torch


def accuracy(output, target, topk=(1,)):
    """
    compute top-k accuracy

    :param output: model output
    :param target: ground-truth labels
    :param topk: top k
    :return: accuracy
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def attack_accuracy(output, target):
    """
    compute attack accuracy

    :param output: attack model output
    :param target: ground-truth membership labels
    :return: attack accuracy
    """
    predict = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
    correct = (predict == target).sum()
    total = target.size(0)
    res = 100 * correct / total
    return res

