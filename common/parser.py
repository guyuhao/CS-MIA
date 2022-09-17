# -*- coding: utf-8 -*-
"""
Implementation of parsing the configuration file
"""

import argparse

import yaml

from common import get_num_classes
import logging


def get_args():
    """
    parse configuration yaml file

    :return: configuration
    """
    defense_list = ['sec_agg', 'know_dist', 'dropout', 'dp', 'adv_reg']
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_template.yaml')
    parser.add_argument('--blindmi', type=int, default=0)

    parser.add_argument('--defense', type=str, default='',
                        help='Support sec_agg(secure aggregation), know_dist(knowledge distillation)')
    parser.add_argument('--metric', type=str, default='mentr',
                        help='Support conf, mentr')

    temp = parser.parse_args()
    yaml.warnings({'YAMLLoadWarning': False})
    f = open(temp.config, 'r', encoding='utf-8')
    cfg = f.read()
    args = dict_to_object(yaml.load(cfg))
    f.close()
    args.num_classes = get_num_classes(args.dataset)

    if temp.defense in defense_list:
        args[temp.defense] = 1

    args['blindmi'] = temp.blindmi
    args['metric'] = temp.metric

    if temp.defense in args.keys() and args[temp.defense]:
        args.log = temp.defense + '_' + args.log
    set_logging(args.log)

    return args


def set_logging(log_file):
    """
        configure logging INFO messaged located in tests/result

        :param str log_file: path of log file
        """
    logging.basicConfig(
        level=logging.DEBUG,
        filename='./result/{}'.format(log_file),
        filemode='w',
        format='[%(asctime)s| %(levelname)s| %(processName)s] %(message)s' # 日志格式
    )


# transfer configuration Dict to Class, which use . to access configuration property, such args.dataset
class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d


def dict_to_object(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    inst = Dict()
    for k, v in dictObj.items():
        inst[k] = dict_to_object(v)
    return inst
