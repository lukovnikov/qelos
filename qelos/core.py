import torch
from torch.autograd import Variable
from torch import nn
import numpy as np
from qelos.util import isnumber, isstring


class param(object):        # TODO hook in somehow
    def __init__(self, shape, lrmul=1., regmul=1., name=None):
        self.shape = shape
        self.value = nn.Parameter(torch.FloatTensor(*shape))

    def uniform(self, range=0.01, std=None, mean=0.0):
        if std is not None:
            a = mean - np.sqrt(3) * std
            b = mean + np.sqrt(3) * std
        else:
            try:
                a, b = range  # range is a tuple
            except TypeError:
                a, b = -range, range  # range is a number
        nn.init.uniform(self.value, -a, +a)
        return self.value

    def normal(self, std=0.01, mean=0.0):
        nn.init.normal(self.value, mean, std)
        return self.value

    def glorotnormal(self, arg=1.0):
        def inner():
            if isstring(arg):
                gain = nn.init.calculate_gain(arg)
            elif isnumber(arg):
                gain = arg
            else:
                raise Exception("unexpected arg type")
            nn.init.xavier_normal(self.value, gain)

        inner()
        return self.value

    def glorotuniform(self, arg=1.0):
        def inner():
            if isstring(arg):
                gain = nn.init.calculate_gain(arg)
            elif isnumber(arg):
                gain = arg
            else:
                raise Exception("unexpected arg type")
            nn.init.xavier_uniform(self.value, gain)
        inner()
        return self.value

    def henormal(self, gain=1.0, c01b=False):
        return None     # TODO

    def heuniform(self, gain=1.0, c01b=False):
        return None     # TODO

    def constant(self, val=0.0):
        nn.init.constant(self.value, val)
        return self.value

    def sparse(self, sparsity=0.1, std=0.01):
        nn.init.sparse(self.value, sparsity=sparsity, std=std)
        return self.value

    def orthogonal(self, gain=1.0):
        nn.init.orthogonal(self.value, gain=gain)
        return self.value