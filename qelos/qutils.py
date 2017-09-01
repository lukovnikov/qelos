import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import qelos as q


class var(object):
    all_cuda = False

    def __init__(self, x, requires_grad=False, volatile=False):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        self.v = Variable(x, requires_grad=requires_grad, volatile=volatile)

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


def seq_pack(x, mask):  # mask: (batsize, seqlen)
    # 1. get lengths
    lens = torch.sum(mask, 1)
    # 2. sort by length
    # 3. pack
    return x


def seq_unpack(x):
    return x, mask


class argmap(nn.Module):
    def __init__(self, f=lambda *args, **kwargs: (args, kwargs)):
        super(argmap, self).__init__()
        self.f = f

    def forward(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    @staticmethod
    def from_spec(*argspec, **kwargspec):
        """ specs for output args and kwargs"""

        def dict_arg_map(*args, **kwargs):
            def get_(spece, a, k):
                if isinstance(spece, set):
                    return k[list(spece)[0]]
                else:
                    return a[spece]

            outargs = []
            outkwargs = {}
            for argspec_e in argspec:
                outargs.append(get_(argspec_e, args, kwargs))
            for kwargspec_k, kwargspec_v in kwargspec.items():
                outkwargs[kwargspec_k] = get_(kwargspec_v, args, kwargs)
            return outargs, outkwargs

        mapfn = dict_arg_map
        return argmap(mapfn)