import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import qelos as q


class var(object):
    all_cuda = False

    def __init__(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        self.v = Variable(x)

    def cuda(self, crit=None):
        if crit is False:
            if self.v.is_cuda:
                self.v = self.v.cpu()
        elif crit is True:
            if not self.v.is_cuda:
                self.v = self.v.cuda()
        elif hasattr(crit, "is_cuda"):
            self.cuda(crit=crit.is_cuda)
        elif crit is None:
            self.cuda(crit=var.all_cuda)
        return self


class val(object):
    def __init__(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        self.v = nn.Parameter(x, requires_grad=False)


def name2fn(x):
    mapping = {"tanh": nn.Tanh,
               "sigmoid": nn.Sigmoid,
               "relu": nn.ReLU,
               "linear": nn.Linear,
               "elu": nn.ELU,
               "selu": nn.SELU,
               "crelu": q.CReLU,
               None: q.Identity}
    if x not in mapping:
        raise Exception("unknown activation function name")
    return mapping[x]