# -*- coding: utf-8 -*-
"""
Implementation of BlindMI attack, refers to "Practical blind membership inference attack via differential comparisons"
"""
from functools import partial

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from torch.utils import data

from common import load_data, SHADOW_INDICES_FILE


def compute_pairwise_distances(x, y):
    """Computes the squared pairwise Euclidean distances between x and y.
    Args:
      x: a tensor of shape [num_x_samples, num_features]
      y: a tensor of shape [num_y_samples, num_features]
    Returns:
      a distance matrix of dimensions [num_x_samples, num_y_samples].
    Raises:
      ValueError: if the inputs do no matched the specified dimensions.
    """

    if not len(x.shape) == len(y.shape) == 2:
        raise ValueError('Both inputs should be matrices.')

    norm = lambda x: torch.sum(x.pow(2), 1)

    return (norm(x.unsqueeze(2) - y.t())).t()



def gaussian_kernel_matrix(x, y, sigmas):
    r"""Computes a Guassian Radial Basis Kernel between the samples of x and y.
    We create a sum of multiple gaussian kernels each having a width sigma_i.
    Args:
      x: a tensor of shape [num_samples, num_features]
      y: a tensor of shape [num_samples, num_features]
      sigmas: a tensor of floats which denote the widths of each of the
        gaussians in the kernel.
    Returns:
      A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
    """
    sigmas = sigmas.to(x.device)
    beta = 1. / (2. * (sigmas.unsqueeze(1)))
    dist = compute_pairwise_distances(x, y)
    dist_shape = dist.shape
    s = torch.mm(beta, dist.reshape(1, -1))
    result = torch.sum((-s).exp(), dim=0)
    return result.reshape(dist_shape)


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
    '''
    Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
    Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
    the distributions of x and y. Here we use the kernel two sample estimate
    using the empirical mean of the two distributions.
    MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
                = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },
    where K = <\phi(x), \phi(y)>,
      is the desired kernel function, in this case a radial basis kernel.
    Args:
        x: a tensor of shape [num_samples, num_features]
        y: a tensor of shape [num_samples, num_features]
        kernel: a function which computes the kernel in MMD. Defaults to the
                GaussianKernelMatrix.
    Returns:
        a scalar denoting the squared maximum mean discrepancy loss.
    '''
    # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
    cost = torch.mean(kernel(x, x))
    cost += torch.mean(kernel(y, y))
    cost -= 2 * torch.mean(kernel(x, y))
    # We do not allow the loss to become negative.
    cost = torch.where(cost > 0., cost, torch.tensor(0.).to(cost.device))
    return cost


def mmd_loss(source_samples, target_samples, weight, scope=None):
    """Adds a similarity loss term, the MMD between two representations.
    This Maximum Mean Discrepancy (MMD) loss is calculated with a number of
    different Gaussian kernels.
    Args:
      source_samples: a tensor of shape [num_samples, num_features].
      target_samples: a tensor of shape [num_samples, num_features].
      weight: the weight of the MMD loss.
      scope: optional name scope for summary tags.
    Returns:
      a scalar tensor representing the MMD loss value.
    """
    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    gaussian_kernel = partial(
        gaussian_kernel_matrix, sigmas=torch.tensor(sigmas))

    loss_value = maximum_mean_discrepancy(
        source_samples, target_samples, kernel=gaussian_kernel)
    loss_value = torch.max(torch.tensor(1e-4).to(loss_value.device), loss_value) * weight
    return loss_value.item()


def sobel(img_set):
    ret = np.empty(img_set.shape)
    for i, img in enumerate(img_set):
        grad_x = cv.Sobel(np.float32(img), cv.CV_32F, 1, 0)
        grad_y = cv.Sobel(np.float32(img), cv.CV_32F, 0, 1)
        gradx = cv.convertScaleAbs(grad_x)
        grady = cv.convertScaleAbs(grad_y)
        gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
        ret[i, :] = gradxy
    return ret


class BlindMI_Diff_W:
    def __init__(self):
        super(BlindMI_Diff_W, self).__init__()
        self.non_member_x = None
        self.non_member_y = None

    # def get_non_member_dataset(self, args):
    def get_non_member_dataset(self, args, train_loader=None, test_loader=None):
        non_iid_prefix = '_non_iid' if args.non_iid else ''
        indices_file = SHADOW_INDICES_FILE + '{}_{}_{}_{}.npz'. \
            format(non_iid_prefix, args.dataset, args.attack_train_m_size, args.attack_train_nm_size)
        _, test_loader = load_data(dataset=args.dataset,
                                   indices_file=indices_file,
                                   batch_size=args.cs_mia_batch_size,
                                   type='train-test')
        non_member_x, non_member_y = [], []
        for i, (test_x, test_y) in enumerate(test_loader):
            if args.cuda:
                test_x, test_y = test_x.cuda(), test_y.cuda()
            non_member_x.append(test_x)
            non_member_y.append(test_y)
        self.non_member_x = torch.cat(non_member_x, dim=0)
        self.non_member_y = torch.cat(non_member_y, dim=0)

    def get_test_data(self, train_loader, test_loader, args):
        self.get_non_member_dataset(args)
        attack_x, attack_y, attack_m = [], [], []
        with torch.no_grad():
            # data used in training, label is 1
            for i, (train_x, train_y) in enumerate(train_loader.dataset):
                if args.cuda:
                    train_x = train_x.cuda()
                attack_x.append(train_x.unsqueeze(0))
                attack_y.append(train_y)
                attack_m.append(1)
            # data not used in training, label is 0
            for i, (test_x, test_y) in enumerate(test_loader.dataset):
                if args.cuda:
                    test_x = test_x.cuda()
                attack_x.append(test_x.unsqueeze(0))
                attack_y.append(test_y)
                attack_m.append(0)
            attack_x = torch.cat(attack_x, dim=0)
            attack_y = torch.tensor(attack_y)
            attack_m = torch.tensor(attack_m)
        return attack_x, attack_y, attack_m

    def diff_Mem_attack(self, x_, y_true, m_true, target_model, args):
        '''
        Attck the target with BLINDMI-DIFF-W, BLINDMI-DIFF with gernerated non-member.
        The non-member is generated by randomly chosen data and the number is 20 by default.
        If the data has been shuffled, please directly remove the process of shuffling.
        :param target_model: the model that will be attacked
        :param x_: the data that target model may used for training
        :param y_true: the label of x_
        :param m_true: one of 0 and 1, which represents each of x_ has been trained or not.
        :param non_Mem_Generator: the method to generate the non-member data. The default non-member generator
        is Sobel.
        :return:  Tensor arrays of results
        '''
        device = self.non_member_x.device
        with torch.no_grad():
            target_model.eval()
            y_true = y_true.cpu().numpy()
            y_true_one_hot = to_categorical(y_true, num_classes=args.num_classes)
            if args.cuda:
                target_model = target_model.cuda()
                x_ = x_.cuda()
                self.non_member_x = self.non_member_x.cuda()
            softmax = nn.Softmax(dim=1)
            temp = target_model(x_)
            if isinstance(temp, tuple):
                temp, _ = temp
            y_pred = softmax(temp).cpu().numpy()
            mix = np.c_[y_pred[y_true_one_hot.astype(bool)], np.sort(y_pred, axis=1)[:, ::-1][:, :2]]

            self.non_member_y = self.non_member_y.cpu().numpy()
            non_member_y_one_hot = to_categorical(self.non_member_y, num_classes=args.num_classes)
            temp = target_model(self.non_member_x)
            if isinstance(temp, tuple):
                temp, _ = temp
            nonMem_pred = softmax(temp).cpu().numpy()
            nonMem = torch.tensor(np.c_[nonMem_pred[non_member_y_one_hot.astype(bool)],
                                        np.sort(nonMem_pred, axis=1)[:, ::-1][:, :2]]).to(device)

            ds = data.TensorDataset(torch.tensor(mix), torch.tensor(m_true))
            dl = data.DataLoader(dataset=ds,
                                 batch_size=20,
                                 shuffle=True,
                                 drop_last=False)

            m_pred, m_true = [], []
            mix_shuffled = []
            for i, (mix_batch, m_true_batch) in enumerate(dl):
                mix_batch = mix_batch.to(device)
                m_true_batch = m_true_batch.to(device)
                m_pred_batch = np.ones(mix_batch.shape[0])
                m_pred_epoch = np.ones(mix_batch.shape[0])
                nonMemInMix = True
                while nonMemInMix:
                    mix_epoch_new = mix_batch[m_pred_epoch.astype(bool)]
                    dis_ori = mmd_loss(nonMem, mix_epoch_new, weight=1)
                    nonMemInMix = False
                    for index, item in enumerate(mix_batch):
                        if m_pred_batch[index] == 1:
                            nonMem_batch_new = torch.cat([nonMem, mix_batch[index: index+1]], dim=0)
                            mix_batch_new = torch.cat([mix_batch[:index], mix_batch[index+1:]], dim=0)
                            m_pred_without = np.r_[m_pred_batch[:index], m_pred_batch[index+1:]]
                            mix_batch_new = mix_batch_new[m_pred_without.astype(bool, copy=True)]
                            dis_new = mmd_loss(nonMem_batch_new, mix_batch_new, weight=1)
                            if dis_new > dis_ori:
                                nonMemInMix = True
                                m_pred_epoch[index] = 0
                    m_pred_batch = m_pred_epoch.copy()

                mix_shuffled.append(mix_batch)
                m_pred.append(m_pred_batch)
                m_true.append(m_true_batch.cpu().numpy())

        y_label = np.concatenate(m_true, axis=0)
        y_predict = np.concatenate(m_pred, axis=0)

        recall = recall_score(y_label, y_predict)
        precision = precision_score(y_label, y_predict)
        F1 = f1_score(y_label, y_predict)
        acc = accuracy_score(y_label, y_predict)
        # 精度调整
        recall = (recall * 100).round(3)
        precision = (precision * 100).round(3)
        F1 = (F1 * 100).round(3)
        acc = (acc * 100).round(3)
        return acc, precision, recall, F1


