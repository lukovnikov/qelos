import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import qelos as q
from torch.utils.data import DataLoader


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


def dataload(*tensors, **kw):
    tensordataset = q.TensorDataset(*tensors)
    dataloader = DataLoader(tensordataset, **kw)
    return dataloader


def params_of(module):
    params = module.parameters()
    params = filter(lambda x: x.requires_grad == True, params)
    return params


def batchablesparse2densemask(bs):      # 2D !!!
    bsshape = None
    if bs.dim() > 2:
        bsshape = bs.size()
        bs = bs.contiguous().view(-1, bs.size(-1))
    vocsize = int(bs.data[0, 1])
    outmask_t = var(torch.ByteTensor(bs.size(0), vocsize + 1)).cuda(bs).v
    outmask_t.data.fill_(0)
    outmask_t.data.scatter_(1, bs.data[:, 2:].long(), 1)
    outmask_t.data = outmask_t.data[:, 1:]
    # assert(test_batchablesparse2densemask(bs, outmask_t))
    if bsshape is not None:
        outmask_t = outmask_t.contiguous().view(bsshape[:-1] + (outmask_t.size(-1),))
    return outmask_t


def test_batchablesparse2densemask(bs, outmask):
    bs = bs.data.numpy()
    outmask = outmask.data.numpy()
    npoutmask = np.zeros_like(outmask)
    for i in range(bs.shape[0]):
        for j in range(2, bs.shape[1]):
            k = bs[i, j]
            if k > 0:
                npoutmask[i, k-1] = 1
    assert(np.all(npoutmask == outmask))


import copy


def rec_clone(x):
    if isinstance(x, list):
        ret = copy.copy(x)
        for i in range(len(ret)):
            ret[i] = rec_clone(ret[i])
    elif isinstance(x, dict):
        ret = copy.copy(x)
        for k, v in ret.items():
            ret[k] = rec_clone(v)
    elif isinstance(x, set):
        ret = copy.copy(x)
        retelems = copy.copy(ret)
        ret.clear()
        for retelem in retelems:
            ret.add(rec_clone(retelem))
    elif isinstance(x, tuple):
        ret = tuple()
        for retelem in x:
            ret += (rec_clone(retelem),)
    elif isinstance(x, torch.autograd.Variable):
        ret = q.var(rec_clone(x.data)).v
    elif isinstance(x, torch.Tensor):
        ret = x.clone()
    elif isinstance(x, np.ndarray):
        ret = x.copy()
    else:
        raise q.SumTingWongException("not supported type: {}".format(type(x)))
    return ret


def intercat(tensors, axis=-1):
    if axis != -1 and axis != tensors[0].dim()-1:
        tensors = [tensor.transpose(axis, -1) for tensor in tensors]
    t = torch.stack(tensors, -1)
    t = t.view(t.size()[:-2] + (-1,))
    if axis != -1 and axis != tensors[0].dim()-1:
        t = t.transpose(axis, -1)
    return t


