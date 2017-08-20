from torch import nn
from torch.autograd import Variable

import qelos
from qelos.basic import Identity


class var(object):
    all_cuda = False

    def __init__(self, torchtensor):
        self.v = Variable(torchtensor)

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
    def __init__(self, tensor):
        self.v = nn.Parameter(tensor, requires_grad=False)


def name2fn(x):
    mapping = {"tanh": nn.Tanh,
               "sigmoid": nn.Sigmoid,
               "relu": nn.ReLU,
               "linear": nn.Linear,
               "elu": nn.ELU,
               "selu": nn.SELU,
               "crelu": qelos.basic.CReLU,
               None: Identity}
    if x not in mapping:
        raise Exception("unknown activation function name")
    return mapping[x]