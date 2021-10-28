'''
Helper functions.
'''

import os
from textwrap import indent
import torch
import torch.nn.functional as F


def check_path(path):
    '''
    Checks if a given path exists. If not, create the directory.
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    return None


def get_device():
    '''
    Checks if GPU is available to be used. If not, CPU is used.
    '''
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def onehot_encode(tensor, num_classes, device):
    '''
    Encodes the given tensor into one-hot vectors.
    '''
    # return F.one_hot(tensor, num_classes).to(device)
    return torch.eye(num_classes).to(device).index_select(dim=0, index=tensor.to(device))


def accuracy_calc(predictions, labels):
    '''
    Calculates prediction accuracy.
    '''

    num_data = labels.size()[0]
    correct_pred = torch.sum(predictions == labels)
    accuracy = (correct_pred.item()*100)/num_data

    return accuracy


def get_learning_rate(optimizer):
    '''
    Returns the current LR from the optimizer module.
    '''

    for params in optimizer.param_groups:
        return params['lr']
