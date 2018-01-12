import torch
import qelos as q

# regularization

def add(x):     # x's are regularization variables
    pass

def l1(*x):
    acc = 0
    for x_e in x:
        acc += torch.abs(x_e)
    return acc


def l2(*x):
    acc = 0
    for x_e in x:
        acc += (x_e ** 2).sum()
    return acc

