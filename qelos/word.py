# TODO: word embeddings, Glove etc.

from collections import OrderedDict

import numpy as np
import os, pickle as pkl
from IPython import embed

import qelos as q
from qelos.util import ticktock, isnumber, issequence, isstring
from torch import nn
import torch
from torch.autograd import Variable


class WordEmbBase(object):
    masktoken = "<MASK>"
    raretoken = "<RARE>"

    def __init__(self, worddic, **kw):
        super(WordEmbBase, self).__init__(**kw)
        self.D = OrderedDict() if worddic is None else worddic

    # region NON-BLOCK API :::::::::::::::::::::::::::::::::::::
    def getindex(self, word):
        return self.D[word] if word in self.D else self.D[self.raretoken] if self.raretoken in self.D else -1

    def __mul__(self, other):
        return self.getindex(other)

    def __contains__(self, word):
        return word in self.D

    def getvector(self, word):
        try:
            if isstring(word):
                return self.get_weight()[self.D[word]]
            elif isnumber(word):
                return self.get_weight()[word, :]
        except Exception:
            return None

    def __getitem__(self, word):
        v = self.getvector(word)
        return v if v is not None else self.get_weight()[0, :]

    def get_weight(self):
        return self.embedding.weight.data.cpu().numpy()

    @property
    def shape(self):
        return self.get_weight().shape

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

    @property
    def block(self):
        return self


class WordEmb(WordEmbBase, nn.Module):
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
        if self.maskid is not None:
            mask = (x != self.maskid).int()
            return ret, mask
        else:
            return ret

    def adapt(self, wdic):      # adapts to given word-idx dictionary
        return AdaptedWordEmb(self, wdic)

    def override(self, wordemb, which=None):    # uses override vectors instead of base vectors if word in override dictionary
        return OverriddenWordEmb(self, wordemb, which=which)

    def augment(self, wordemb):
        return AugmentedWordEmb(self, wordemb)


class AdaptedWordEmb(WordEmbBase, nn.Module):  # adapt to given dictionary, map extra words to rare
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

    def get_weight(self):
        return self.inner.get_weight()[self.adb.data.cpu().numpy()]

    def forward(self, inp):
        # x = q.var(self.adb).cuda(inp).v.gather(0, inp)
        x = self.adb.gather(0, inp)
        ret = self.inner(x)
        if issequence(ret):
            ret, mask = ret
            return ret, mask
        else:
            return ret


class OverriddenWordEmb(WordEmb):
    def __init__(self, base, override, which=None, **kw):
        assert(base.outdim == override.outdim)  # ensure same output dimension
        baseindexes_val = np.arange(max(base.D.values()) + 1).astype("int32")
        baseindexes = Val(baseindexes_val)
        basevar = base(baseindexes)     # slicing out all base vectors as they are
        overridemask_val = np.zeros_like(baseindexes_val, dtype="float32")
        overrideindexes_val = np.zeros_like(baseindexes_val, dtype="int32")
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
        overrideindexes = Val(overrideindexes_val)
        overridemask = Val(overridemask_val)
        overridevar = override(overrideindexes)
        v = T.switch(overridemask.dimadd(1), overridevar, basevar)
        super(OverriddenWordEmb, self).__init__(worddic=base.D, value=v,
               dim=base.outdim, **kw)


class AugmentedWordEmb(WordEmb):    # TODO: RARE TOKEN MGMT
    def __init__(self, base, augment, **kw):
        assert(base.outdim == augment.outdim)
        super(AugmentedWordEmb, self).__init__(worddic=base.D, value=False,
                dim=base.outdim, normalize=base.normalize,
                trainfrac=base.trainfrac, **kw)
        self.base = base
        self.augment = augment
        self.ad = {v: augment.D[k]
                    if k in augment.D
                    else 0
                   for k, v in base.D.items()}
        valval = np.zeros((max(self.ad.keys()) + 1,), dtype="int32")
        for i in range(valval.shape[0]):
            valval[i] = self.ad[i] if i in self.ad else 0
        self.adb = Val(valval)

    def apply(self, x):
        baseemb = self.base(x)
        augmemb = self.augment(self.adb[x])
        ret = T.concatenate([baseemb, augmemb], axis=1)
        self._maskfrom(ret, x)
        return ret

    @property
    def get_weight(self):
        return None         # TODO


class Glove(WordEmb):
    defaultpath = "../../../data/glove/glove.%dd"
    maskid = 0
    rareid = 1

    def __init__(self, dim, vocabsize=None, path=None, trainfrac=0.0, maskid=None,
                 **kw):
        assert("worddic" not in kw)
        path = self._get_path(dim, path=path)
        maskid = self.maskid if maskid is None else maskid
        value, wdic = self.loadvalue(path, dim, indim=vocabsize, maskid=maskid, rareid=self.rareid)
        self.allwords = wdic.keys()
        super(Glove, self).__init__(dim=dim, value=value,
                                    worddic=wdic, trainfrac=trainfrac, **kw)

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



