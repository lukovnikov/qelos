# TODO: word embeddings, Glove etc.
import math
from collections import OrderedDict

import numpy as np
import os, pickle as pkl
from IPython import embed

import qelos as q
from qelos.util import ticktock, isnumber, issequence, isstring
from torch import nn
import torch
from torch.autograd import Variable


class WordVecBase(object):
    masktoken = "<MASK>"
    raretoken = "<RARE>"

    def __init__(self, worddic, **kw):
        super(WordVecBase, self).__init__(**kw)
        self.D = OrderedDict() if worddic is None else worddic

    # region NON-BLOCK API :::::::::::::::::::::::::::::::::::::
    def getindex(self, word):
        return self.D[word] if word in self.D else self.D[self.raretoken] if self.raretoken in self.D else -1

    def __mul__(self, other):
        return self.getindex(other)

    def __contains__(self, word):
        return word in self.D

    def getvector(self, word):
        raise NotImplemented()

    def __getitem__(self, word):
        return self.getvector(word)

    @property
    def shape(self):
        raise NotImplemented()

    def cosine(self, A, B):
        return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

    def getdistance(self, A, B, distance=None):
        if distance is None:
            distance = self.cosine
        return distance(self.getvector(A), self.getvector(B))

    def __mod__(self, other):
        if isinstance(other, (tuple, list)):  # distance
            assert len(other) > 1
            if len(other) == 2:
                return self.getdistance(other[0], other[1])
            else:
                y = other[0]
                return map(lambda x: self.getdistance(y, x), other[1:])
        else:  # embed
            return self.__getitem__(other)
    # endregion


class WordEmbBase(WordVecBase, nn.Module):
    def getvector(self, word):
        try:
            if isstring(word):
                word = self.D[word]
            wordid = q.var(torch.LongTensor([word])).v
            ret, _ = self(wordid)
            return ret.squeeze(0).data.numpy()
        except Exception:
            return None

    def adapt(self, wdic):  # adapts to given word-idx dictionary
        return AdaptedWordEmb(self, wdic)

    def override(self, wordemb,
                 which=None):  # uses override vectors instead of base vectors if word in override dictionary
        return OverriddenWordEmb(self, wordemb, which=which)


class WordEmb(WordEmbBase):
    """ is a VectorEmbed with a dictionary to map words to ids """
    def __init__(self, dim=50, value=None, worddic=None,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 sparse=False, freeze=False,
                 **kw):
        assert(worddic is not None)     # always needs a dictionary
        super(WordEmb, self).__init__(worddic, **kw)
        wdvals = worddic.values()
        assert(min(wdvals) >= 0)     # word ids must be positive

        # extract maskid and rareid from worddic
        maskid = worddic[self.masktoken] if self.masktoken in worddic else None
        rareid = worddic[self.raretoken] if self.raretoken in worddic else None
        self.maskid = maskid

        indim = max(worddic.values())+1        # to init from worddic
        self.embedding = nn.Embedding(indim, dim, padding_idx=maskid,
                                      max_norm=max_norm, norm_type=norm_type,
                                      scale_grad_by_freq=scale_grad_by_freq,
                                      sparse=sparse)
        if value is not None:
            self.embedding.weight = nn.Parameter(torch.from_numpy(value))
        if freeze is True:
            self.embedding.weight.requires_grad = False

    def forward(self, x):
        ret = self.embedding(x)
        mask = None
        if self.maskid is not None:
            mask = (x != self.maskid).int()
        return ret, mask


class AdaptedWordEmb(WordEmbBase):  # adapt to given dictionary, map extra words to rare
    def __init__(self, wordemb, wdic, **kw):
        D = wordemb.D
        assert(wordemb.raretoken in D)     # must have rareid in D to map extra words to it
        super(AdaptedWordEmb, self).__init__(wdic, **kw)
        self.inner = wordemb

        self.ad = {v: D[k] if k in D else D[self.raretoken]
                   for k, v in wdic.items()}

        valval = np.ones((max(self.ad.keys()) + 1,), dtype="int64")
        for i in range(valval.shape[0]):
            valval[i] = self.ad[i] if i in self.ad else D[wordemb.raretoken]
        self.adb = nn.Parameter(torch.from_numpy(valval), requires_grad=False)

    def forward(self, inp):
        # x = q.var(self.adb).cuda(inp).v.gather(0, inp)
        inpshape = inp.size()
        inp = inp.view(-1)
        x = self.adb.gather(0, inp)
        ret = self.inner(x)
        mask = None
        if issequence(ret):
            ret, mask = ret
            mask = mask.view(inpshape)
        ret = ret.view(*(inpshape+(-1,)))
        return ret, mask


class ComputedWordEmb(WordEmbBase):
    def __init__(self, data=None, computer=None, worddic=None):
        """
        :param data:
        :param computer:
        :param worddic:
        """
        super(ComputedWordEmb, self).__init__(worddic=worddic)
        self.data = nn.Parameter(torch.from_numpy(data), requires_grad=False)
        self.computer = computer
        self.weight = None
        wdvals = worddic.values()
        assert(min(wdvals) >= 0)     # word ids must be positive

        # extract maskid and rareid from worddic
        maskid = worddic[self.masktoken] if self.masktoken in worddic else None
        rareid = worddic[self.raretoken] if self.raretoken in worddic else None
        self.maskid = maskid
        # assert(maskid is None)
        # assert(rareid is None)
        self.indim = max(worddic.values())+1

    def forward(self, x):
        mask = None
        if self.maskid is not None:
            mask = x != self.maskid
        xshape = x
        x = x.view(-1)
        data = self.data.index_select(0, x)
        emb = self.computer(data)
        emb = emb.view(*(xshape + (-1,)))
        return emb, mask


class OverriddenWordEmb(WordEmbBase):
    def __init__(self, base, override, which=None, **kw):
        super(OverriddenWordEmb, self).__init__(base.D)
        self.base = base
        self.over = override
        # assert(base.outdim == override.outdim)  # ensure same output dimension
        baseindexes_val = np.arange(max(base.D.values()) + 1).astype("int64")
        self.baseindexes = q.val(torch.from_numpy(baseindexes_val)).v
        overridemask_val = np.zeros_like(baseindexes_val, dtype="float32")
        overrideindexes_val = np.zeros_like(baseindexes_val, dtype="int64")
        if which is None:   # which: list of words to override
            for k, v in base.D.items():     # for all symbols in base dic
                if k in override.D:         # if also in override dic
                    overrideindexes_val[v] = override.D[k]   # map base idx to ovrd idx
                    overridemask_val[v] = 1
        else:
            for k in which:
                if k in override.D:     # TODO: if k from which is missing from base.D
                    overrideindexes_val[base.D[k]] = override.D[k]
                    overridemask_val[base.D[k]] = 1
        self.overrideindexes = q.val(torch.from_numpy(overrideindexes_val)).v
        self.overridemask = q.val(torch.from_numpy(overridemask_val)).v

    def forward(self, x):
        xshape = x.size()
        x = x.view(-1)
        base_idx_select = torch.gather(self.baseindexes, 0, x)
        over_idx_select = torch.gather(self.overrideindexes, 0, x)
        over_msk_select = torch.gather(self.overridemask, 0, x)
        base_emb = self.base(base_idx_select)
        base_msk = None
        if isinstance(base_emb, tuple):
            base_emb, base_msk = base_emb
        over_emb = self.over(over_idx_select)
        if isinstance(over_emb, tuple):
            over_emb, over_msk = over_emb
        emb = base_emb * (1 - over_msk_select.unsqueeze(1)) + over_emb * over_msk_select.unsqueeze(1)
        emb = emb.view(*(xshape + (-1,)))
        msk = None
        if base_msk is not None:
            msk = base_msk.view(xshape)
        return emb, msk


class GloveVec(object):
    defaultpath = "../data/glove/glove.%dd"
    maskid = 0
    rareid = 1

    @classmethod
    def _get_path(cls, dim, path=None):
        # if dim=None, load all
        path = cls.defaultpath if path is None else path
        relpath = path % dim
        path = os.path.join(os.path.dirname(__file__), relpath)
        return path

    def loadvalue(self, path, dim, indim=None, maskid=None, rareid=None):
        tt = ticktock(self.__class__.__name__)
        tt.tick()
        W = np.load(open(path+".npy"))
        if indim is not None:
            W = W[:indim, :]
        if rareid is not None:
            W = np.concatenate([np.zeros_like(W[0, :])[np.newaxis, :], W], axis=0)
        if maskid is not None:
            W = np.concatenate([np.zeros_like(W[0, :])[np.newaxis, :], W], axis=0)
        tt.tock("vectors loaded")
        tt.tick()
        # dictionary
        words = pkl.load(open(path+".words"))
        D = OrderedDict()
        i = 0
        if maskid is not None:
            D["<MASK>"] = i; i+=1
        if rareid is not None:
            D["<RARE>"] = i; i+=1
        for j, word in enumerate(words):
            if indim is not None and j >= indim:
                break
            D[word] = i
            i += 1
        tt.tock("dictionary created")
        return W, D


class GloveEmb(WordEmb, GloveVec):

    def __init__(self, dim, vocabsize=None, path=None, freeze=True, maskid=None,
                 **kw):
        assert("worddic" not in kw)
        path = self._get_path(dim, path=path)
        maskid = self.maskid if maskid is None else maskid
        value, wdic = self.loadvalue(path, dim, indim=vocabsize, maskid=maskid, rareid=self.rareid)
        self.allwords = wdic.keys()
        super(GloveEmb, self).__init__(dim=dim, value=value,
                                       worddic=wdic, freeze=freeze, **kw)


class WordLinoutBase(WordVecBase, nn.Module):
    def getvector(self, word):
        try:
            if isstring(word):
                word = self.D[word]
            wordid = q.var(torch.LongTensor([[word]])).v
            ret, _ = self._getvector(wordid)
            return ret.squeeze().data.numpy()
        except Exception:
            return None

    def _getvector(self, wordid):
        raise NotImplemented()

    def adapt(self, wdic):  # adapts to given word-idx dictionary
        return AdaptedWordLinout(self, wdic)

    def override(self, wordemb,
                 which=None):  # uses override vectors instead of base vectors if word in override dictionary
        return OverriddenWordLinout(self, wordemb, which=which)


class WordLinout(WordLinoutBase):
    def __init__(self, outdim, worddic=None, weight=None, bias=True, freeze=False):
        super(WordLinout, self).__init__(worddic)
        wdvals = worddic.values()
        assert(min(wdvals) >= 0)     # word ids must be positive

        # extract maskid and rareid from worddic
        maskid = worddic[self.masktoken] if self.masktoken in worddic else None
        rareid = worddic[self.raretoken] if self.raretoken in worddic else None
        self.maskid = maskid

        indim = max(worddic.values())+1        # to init from worddic
        self.indim = indim
        self.lin = nn.Linear(indim, outdim, bias=bias)

        if weight is not None:
            self.lin.weight = nn.Parameter(torch.from_numpy(weight))
        if freeze is True:
            self.lin.weight.requires_grad = False
            if bias is True:
                self.lin.bias.requires_grad = False

    def _getvector(self, wordid):
        vec = self.lin.weight.index_select(0, wordid)
        return vec

    def forward(self, x, mask=None):    # TODO mask
        ret = self.lin(x)
        ret = ret * mask if mask is not None else mask


class ComputedWordLinout(WordLinoutBase):
    def __init__(self, data=None, computer=None, worddic=None, bias=True):
        super(ComputedWordLinout, self).__init__(worddic)
        self.data = q.val(torch.from_numpy(data)).v
        self.computer = computer

        wdvals = worddic.values()
        assert(min(wdvals) >= 0)     # word ids must be positive

        # extract maskid and rareid from worddic
        maskid = worddic[self.masktoken] if self.masktoken in worddic else None
        rareid = worddic[self.raretoken] if self.raretoken in worddic else None

        self.outdim = max(worddic.values())+1
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.outdim))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self.base_weight = None

    def reset_parameters(self):
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, mask=None):        # (batsize, indim), (batsize, outdim)
        if mask is not None:
            # select data, compute vectors, build switcher
            msk = mask.sum(0)       # --> (outdim,)
            compute_ids = msk.nonzero()
            data_select = self.data[compute_ids]
            comp_weight = self.computer(data_select)        # (num_data_select, indim)
            indim = comp_weight.size(1)
            if self.base_weight is None or self.base_weight.size(1) != indim:
                self.base_weight = q.var(torch.zeros(1, indim)).cuda(x).v
            weight = torch.cat([self.base_weight, comp_weight], 0)
            index_transform = (torch.cumsum(msk, 0) * msk).long()
            weight = weight.gather(0, index_transform)
        else:
            weight = self.computer(self.data)
        out = torch.mm(x, weight.t)
        if self.bias:
            bias = self.bias if mask is not None else self.bias * mask
            out += bias
        return out#, mask ?


class GloveLinout(WordLinout, GloveVec):
    def __init__(self, dim, vocabsize=None, path=None, freeze=True,
                 maskid=None, bias=False,
                 **kw):
        assert ("worddic" not in kw)
        path = self._get_path(dim, path=path)
        maskid = self.maskid if maskid is None else maskid
        value, wdic = self.loadvalue(path, dim, indim=vocabsize, maskid=maskid, rareid=self.rareid)
        self.allwords = wdic.keys()
        outdim = max(wdic.values()) + 1
        super(GloveLinout, self).__init__(outdim, value=value,
                                          worddic=wdic, freeze=freeze, bias=bias,
                                          **kw)


class AdaptedWordLinout(WordLinoutBase):
    def __init__(self, wordlinout, wdic, **kw):
        D = wordlinout.D
        assert (wordlinout.raretoken in D)  # must have rareid in D to map extra words to it
        assert(self.raretoken in wdic)
        super(AdaptedWordLinout, self).__init__(wdic, **kw)
        self.inner = wordlinout

        self.new_to_old_d = {v: D[k] if k in D else D[self.raretoken]
                   for k, v in wdic.items()}
        self.old_to_new_d = {v: wdic[k] if k in wdic else wdic[wordlinout.raretoken]
                             for k, v in D.items()}

        numnew = max(self.new_to_old_d.keys()) + 1
        numold = max(self.old_to_new_d.keys()) + 1

        new_to_old = np.zeros((numnew,), dtype="int64")
        for i in range(new_to_old.shape[0]):
            j = self.new_to_old_d[i] if i in self.new_to_old_d else D[wordlinout.raretoken]
            new_to_old[i] = j
        self.new_to_old = nn.Parameter(torch.from_numpy(new_to_old),
                                       requires_grad=False)  # for every new dic word id, contains old dic id

        old_to_new = np.zeros((numold,), dtype="int64")
        for i in range(old_to_new.shape[0]):
            j = self.old_to_new_d[i] if i in self.old_to_new_d else wdic[self.raretoken]
            old_to_new[i] = j
        self.old_to_new = nn.Parameter(torch.from_numpy(old_to_new),
                                       requires_grad=False)  # for every old dic word id, contains new dic id

    def forward(self, x, mask=None):       # (batsize, indim), (batsize, outdim)
        innermask = mask.index_select(1, self.new_to_old) if mask is not None else None
        baseout = self.inner(x, mask=innermask)     # (batsize, outdim) --> need to permute columns
        out = baseout.index_select(1, self.old_to_new)
        return out#, mask?


class OverriddenWordLinout(WordLinoutBase):
    def __init__(self, base, override, which=None, **kw):
        super(OverriddenWordLinout, self).__init__(base.D)
        self.base = base
        self.over = override.adapt(base.D)

        numout = max(base.D.values()) + 1

        overridemask_val = np.zeros((numout,), dtype="float32")
        if which is None:   # which: list of words to override
            for k, v in base.D.items():     # for all symbols in base dic
                if k in override.D:         # if also in override dic
                    overridemask_val[v] = 1
        else:
            for k in which:
                if k in override.D:     # TODO: if k from which is missing from base.D
                    overridemask_val[base.D[k]] = 1
        self.overridemask = q.val(torch.from_numpy(overridemask_val)).v

    def forward(self, x, mask=None):    # (batsize, indim), (batsize, outdim)
        baseres = self.base(x, mask=mask)
        overres = self.over(x, mask=mask)
        res = self.overridemask.unsqueeze(0) * overres \
              + (1 - self.overridemask.unsqueeze(0)) * baseres
        if mask is not None:
            res = res * mask
        return res#, mask