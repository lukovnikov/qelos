import torch
import qelos as q

# regularization

def add(x):     # x's are regularization variables


def l1(*x, **kw):
    l = q.getkw(kw, "l", 0)
    acc = 0
    for x_e in x:
        acc += l * torch.abs(x_e)
    return acc


def l2(*x, **kw):
    l = q.getkw(kw, "l", 0)
    acc = 0
    for x_e in x:
        acc += l * torch.norm(x_e, 2)
    return acc

