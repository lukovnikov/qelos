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
        if isinstance(crit, int) and not isinstance(crit, bool):
            self.v = self.v.cuda(crit)
        elif crit is False:
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


class hyperparam(object):
    def __init__(self, initval):
        super(hyperparam, self).__init__()
        self._initval = initval
        self._v = initval

    def reset(self):
        self._v = self._initval

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, value):
        self._v = value


def v(x):
    if isinstance(x, hyperparam):
        return x._v
    elif isinstance(x, (var, val)):
        return x.v
    elif isinstance(x, torch.autograd.Variable):
        return x.data
    elif isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    else:
        return x


def add_tag(x, tag):
    assert(isinstance(x, Variable) and q.isstring(tag))
    add_qelos_key(x, "tags", set())
    x._qelos["tags"].add(tag)


def remove_tag(x, tag):
    assert(isinstance(x, Variable) and q.isstring(tag))
    if hasattr(x, "_qelos") and "tags" in x._qelos:
        x._qelos["tags"].remove(tag)


def get_tags(x):
    assert(isinstance(x, Variable))
    if hasattr(x, "_qelos") and "tags" in x._qelos:
        return x._qelos["tags"]


def filter_by_tag(xs, tag):
    assert(q.isstring(tag))
    for x in xs:
        if hasattr(x, "_qelos") and "tags" in x._qelos and tag in x._qelos["tags"]:
            yield x


def add_qelos_key(x, k, v=None):
    assert(isinstance(x, Variable) and q.isstring(k))
    if not hasattr(x, "_qelos"):
        x._qelos = {}
    if k not in x._qelos:
        x._qelos[k] = v


def has_qelos_key(x, k):
    assert (isinstance(x, Variable) and q.isstring(k))
    return hasattr(x, "_qelos") and k in x._qelos


def get_qelos_key(x, k):
    if has_qelos_key(x, k):
        return x._qelos[k]


def remove_qelos_key(x, k):
    if has_qelos_key(x, k):
        del x._qelos[k]


def gradmult(xs, frac=1.):  # supports hyperparam as frac
    def hookf(_grad):
        return _grad * q.v(frac)
    if isinstance(xs, Variable):
        xs = [xs]
    for xt in xs:
        remover = xt.register_hook(hookf)
        add_qelos_key(xt, "gradmult_removers", set())
        xt._qelos["gradmult_removers"].add(remover)


def remove_gradmult(xs):
    if isinstance(xs, Variable):
        xs = [xs]
    for xt in xs:
        if hasattr(xt, "_qelos") and "gradmult_removers" in xt._qelos:
            for rmvr in xt._qelos["gradmult_removers"]:
                rmvr()
            del xt._qelos["gradmult_removers"]


def set_lr(x, lr):
    if isinstance(x, nn.Module):
        for p in q.params_of(x):
            set_lr(p, lr)
    else:
        add_qelos_key(x, "lr", None)
        x._qelos["lr"] = lr


def remove_lr(x):
    if isinstance(x, nn.Module):
        for p in q.params_of(x):
            remove_lr(p)
    else:
        remove_qelos_key(x, "lr")


def set_l2(x, l2):
    if isinstance(x, nn.Module):
        for p in q.params_of(x):
            set_l2(p, l2)
    else:
        add_qelos_key(x, "l2", None)
        x._qelos["l2"] = l2


def remove_l2(x):
    if isinstance(x, nn.Module):
        for p in q.params_of(x):
            remove_l2(p)
    else:
        remove_qelos_key(x, "l2")


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
    x = x.float()
    mask = mask.float()
    # 1. get lengths
    lens = torch.sum(mask.float(), 1)
    # 2. sort by length
    assert(lens.dim() == 1)
    sortidxs = np.argsort(-lens.cpu().data.numpy())
    unsorter = np.zeros_like(sortidxs)
    unsorter[sortidxs] = np.arange(0, len(unsorter))
    unsorter = q.var(unsorter).cuda(x).v
    # 3. pack
    sortids = q.var(sortidxs).cuda(x).v
    sortedseq = torch.index_select(x, 0, sortids)
    sortedmsk = torch.index_select(mask, 0, sortids)
    sortedlens = sortedmsk.long().sum(1)
    sortedlens = list(sortedlens.cpu().data.numpy())
    packedseq = torch.nn.utils.rnn.pack_padded_sequence(sortedseq, sortedlens, batch_first=True)
    # packedmsk = torch.nn.utils.rnn.pack_padded_sequence(sortedmsk, sortedlens, batch_first=True)
    return packedseq, unsorter


def seq_unpack(x, order, padding_value=0):
    unpacked, lens = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, padding_value=padding_value)
    mask = np.zeros((len(lens), max(lens)), dtype="int64")
    for i, l in enumerate(lens):
        mask[i, :l] = 1
    mask = q.var(mask).cuda(unpacked).v
    out = torch.index_select(unpacked, 0, order)
    outmask = torch.index_select(mask, 0, order)
    return out, outmask


def dataload(*tensors, **kw):
    tensordataset = q.TensorDataset(*tensors)
    dataloader = DataLoader(tensordataset, **kw)
    return dataloader


def params_of(m):
    params = m.parameters()
    params = filter(lambda x: x.requires_grad == True, params)
    return params


def paramgroups_of(m):
    params = q.params_of(m)
    default_group = {"params": []}
    paramgroups = []
    for param in params:
        g = None
        if q.has_qelos_key(param, "lr"):
            g = {"params": [param], "lr": q.get_qelos_key(param, "lr")}
        if q.has_qelos_key(param, "l2"):
            g = {"params": [param], "weight_decay": q.get_qelos_key(param, "l2")}
        if g is None:
            default_group["params"].append(param)
        else:
            paramgroups.append(g)
    paramgroups.append(default_group)
    return paramgroups


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


