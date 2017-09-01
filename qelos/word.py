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
        """
        Takes a worddic and provides word id lookup and vector retrieval interface
        """
        super(WordVecBase, self).__init__(**kw)
        self.D = OrderedDict() if worddic is None else worddic

    # region NON-BLOCK API :::::::::::::::::::::::::::::::::::::
    def getindex(self, word):
        return self.D[word] if word in self.D else self.D[self.raretoken] if self.raretoken in self.D else -1

    def __mul__(self, other):
        """ given word (string), returns index (integer) based on dictionary """
        return self.getindex(other)

    def __contains__(self, word):
        return word in self.D

    def getvector(self, word):
        raise NotImplemented()

    def __getitem__(self, word):
        """ given word (string or index), returns vector """
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
        """ Given word (string or integer), returns vector.
            Does similarity computation if argument is a sequences. Compares first element of sequence with each of the following elements. """
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
    """
    All WordEmbs must be descendant.
    """
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
        """
        Adapts current word emb to a new dictionary
        """
        return AdaptedWordEmb(self, wdic)

    def override(self, wordemb,
                 which=None, whichnot=None):  # uses override vectors instead of base vectors if word in override dictionary
        """
        Overrides this wordemb's token vectors with the vectors from given wordemb.
        Optionally, restriction of which tokens to override can be specified by providing
            a list of tokens in which= argument.
        Optionally, exclusions can be made using whichnot
        """
        return OverriddenWordEmb(self, wordemb, which=which, whichnot=whichnot)

    def merge(self, wordemb, mode="sum"):
        """
        Merges this embedding with provided embedding using the provided mode.
        The dictionary of provided embedding must be identical to this embedding.
        """
        if not wordemb.D == self.D:
            raise q.SumTingWongException("must have identical dictionary")
        return MergedWordEmb(self, wordemb, mode=mode)


class WordEmb(WordEmbBase):
    """ is a VectorEmbed with a dictionary to map words to ids """
    def __init__(self, dim=50, value=None, worddic=None,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 sparse=False, fixed=False,
                 **kw):
        """
        Normal word embedder. Wraps nn.Embedding.

        :param dim: embedding vector dimension
        :param value: (optional) value to set the weight of nn.Embedding to
        :param worddic: worddic, must be provided
        :param max_norm: see nn.Embedding
        :param norm_type: see nn.Embedding
        :param scale_grad_by_freq: see nn.Embedding
        :param sparse: see nn.Embedding
        :param fixed: fixed embeddings
        :param kw:
        """
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
        if fixed is True:
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
        # assert(wordemb.raretoken in D)     # must have rareid in D to map extra words to it
        super(AdaptedWordEmb, self).__init__(wdic, **kw)
        self.inner = wordemb

        rareid = D[wordemb.raretoken] if wordemb.raretoken in D else 0

        # maps all idx from wdic (new) to idx in wordemb.D (old)
        # maps words from wdic (new) that are missing in wordemb.D (old)
        #   to wordemb.D's rare id

        self.ad = {v: D[k] if k in D else rareid for k, v in wdic.items()}

        valval = np.ones((max(self.ad.keys()) + 1,), dtype="int64")
        for i in range(valval.shape[0]):
            valval[i] = self.ad[i] if i in self.ad else rareid
        self.adb = q.val(valval).v

    def forward(self, inp):
        # x = q.var(self.adb).cuda(inp).v.gather(0, inp)
        inpshape = inp.size()
        inp = inp.view(-1)
        x = self.adb.gather(0, inp)
        ret, msk = self.inner(x)
        if msk is not None:
            msk = msk.view(inpshape)
        ret = ret.view(*(inpshape+(-1,)))
        return ret, msk


class ComputedWordEmb(WordEmbBase):
    def __init__(self, data=None, computer=None, worddic=None):
        """
        Takes some numpy tensor, a module and a worddic and computes token vectors on the fly.

        :param data: numpy tensor, wrapped with tensor.from_numpy(), so must watch dtype
        :param computer: nn.Module that takes (some of) the data and computes a vector for each data row
        :param worddic: dictionary of tokens to ids
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
        xshape = x.size()
        x = x.view(-1)
        data = self.data.index_select(0, x)
        emb = self.computer(data)
        emb = emb.contiguous()
        emb = emb.view(*(xshape + (-1,)))
        return emb, mask


class OverriddenWordVecBase(WordVecBase, nn.Module):
    def __init__(self, base, override, which=None, whichnot=None, **kw):
        super(OverriddenWordVecBase, self).__init__(base.D)
        self.base = base
        self.over = override.adapt(base.D)
        assert(not (which is not None and whichnot is not None))
        numout = max(base.D.values()) + 1
        whichnot = set()

        overridemask_val = np.zeros((numout,), dtype="float32")
        if which is None:   # which: list of words to override
            for k, v in base.D.items():     # for all symbols in base dic
                if k in override.D and k not in whichnot:         # if also in override dic
                    overridemask_val[v] = 1
        else:
            for k in which:
                if k in override.D:     # TODO: if k from which is missing from base.D
                    overridemask_val[base.D[k]] = 1
        self.overridemask = q.val(overridemask_val).v


class OverriddenWordEmb(OverriddenWordVecBase, WordEmbBase):
    def forward(self, x):
        xshape = x.size()
        x = x.view(-1)
        base_emb, base_msk = self.base(x)
        over_emb, over_msk = self.over(x)
        over_msk_select = torch.gather(self.overridemask, 0, x)
        emb = base_emb * (1 - over_msk_select.unsqueeze(1)) + over_emb * over_msk_select.unsqueeze(1)
        emb = emb.view(*(xshape + (-1,)))
        msk = None
        if base_msk is not None:
            msk = base_msk.view(xshape)
        return emb, msk


class MergedWordVecBase(WordVecBase):
    def __init__(self, base, merge, mode="sum"):
        super(MergedWordVecBase, self).__init__(base.D)
        self.base = base
        self.merg = merge
        self.mode = mode
        if not mode in ("sum", "cat"):
            raise q.SumTingWongException("{} merge mode not suported".format(mode))


class MergedWordEmb(MergedWordVecBase, WordEmbBase):
    def forward(self, x):
        base_emb, base_msk = self.base(x)
        merg_emb, merg_msk = self.merg(x)
        if self.mode == "sum":
            emb = base_emb + merg_emb
            msk = base_msk      # since dictionaries are identical
        elif self.mode == "cat":
            emb = torch.cat([base_emb, merg_emb], 1)
            msk = base_msk
        else:
            raise q.SumTingWongException()
        return emb, msk


class PretrainedWordVec(object):
    defaultpath = "../data/glove/glove.%dd"
    masktoken = "<MASK>"
    raretoken = "<RARE>"

    trylowercase=True

    loadcache = {}
    useloadcache = True

    @classmethod
    def _get_path(cls, dim, path=None):
        # if dim=None, load all
        path = cls.defaultpath if path is None else path
        relpath = path % dim
        path = os.path.join(os.path.dirname(__file__), relpath)
        return path

    def loadvalue(self, path, dim, indim=None, maskid=True, rareid=True):
        # TODO: nonstandard mask and rareid?
        tt = ticktock(self.__class__.__name__)
        tt.tick()
        # load weights
        if path not in self.loadcache:
            W = np.load(open(path+".npy"))
        else:
            W = self.loadcache[path][0]

        # adapt
        if indim is not None:
            W = W[:indim, :]
        if rareid:
            W = np.concatenate([np.zeros_like(W[0, :])[np.newaxis, :], W], axis=0)
        if maskid:
            W = np.concatenate([np.zeros_like(W[0, :])[np.newaxis, :], W], axis=0)
        tt.tock("vectors loaded")
        tt.tick()

        # load words
        if path not in self.loadcache:
            words = pkl.load(open(path+".words"))
        else:
            words = self.loadcache[path]

        # cache
        if self.useloadcache:
            self.loadcache[path] = (W, words)

        # dictionary
        D = OrderedDict()
        i = 0
        if maskid is not None:
            D[self.masktoken] = i; i+=1
        if rareid is not None:
            D[self.raretoken] = i; i+=1
        wordset = set(words)
        for j, word in enumerate(words):
            if indim is not None and j >= indim:
                break
            if word.lower() not in wordset and self.trylowercase:
                word = word.lower()
            D[word] = i
            i += 1
        tt.tock("dictionary created")
        return W, D


class PretrainedWordEmb(WordEmb, PretrainedWordVec):

    def __init__(self, dim, vocabsize=None, path=None, fixed=True, incl_maskid=True, incl_rareid=True, **kw):
        """
        WordEmb that sets the weight of nn.Embedder to loaded pretrained vectors.
        Adds a maskid and rareid as specified on the class.

        :param dim: token vector dimensions
        :param vocabsize: (optional) number of tokens to load
        :param path: (optional) where to load from.
                     Must be of format .../xxx%dxxx.
                     Files must be separated in .npy matrix and .words list.
                     Defaults to glove in qelos/data/.
        :param fixed: no learning
        :param incl_maskid: includes a <MASK> token in dictionary and assigns it id 0
        :param incl_rareid: includes a <RARE> token in dictionary and assigns it id 1 if incl_maskid was True, and id 0 otherwise
        """
        assert("worddic" not in kw)
        path = self._get_path(dim, path=path)
        value, wdic = self.loadvalue(path, dim, indim=vocabsize, maskid=incl_maskid, rareid=incl_rareid)
        self.allwords = wdic.keys()
        super(PretrainedWordEmb, self).__init__(dim=dim, value=value,
                                                worddic=wdic, fixed=fixed, **kw)


class WordLinoutBase(WordVecBase, nn.Module):
    def getvector(self, word):
        try:
            if isstring(word):
                word = self.D[word]
            wordid = q.var(torch.LongTensor([word])).v
            ret = self._getvector(wordid)
            return ret.squeeze().data.numpy()
        except Exception as e:
            return None

    def _getvector(self, wordid):
        raise NotImplemented()

    def adapt(self, wdic):  # adapts to given word-idx dictionary
        return AdaptedWordLinout(self, wdic)

    def override(self, wordemb,
                 which=None):  # uses override vectors instead of base vectors if word in override dictionary
        return OverriddenWordLinout(self, wordemb, which=which)

    def merge(self, x, mode="sum"):
        if not self.D == x.D:
            raise q.SumTingWongException()
        return MergedWordLinout(self, x, mode=mode)


class WordLinout(WordLinoutBase):
    def __init__(self, indim, worddic=None, weight=None, set_bias=None, bias=True, fixed=False):
        """
        Linear block to be used at the output for computing scores over a vocabulary of tokens. Usually followed by Softmax.

        :param indim: incoming dimension
        :param worddic: dictionary of words to ids
        :param weight: (optional) custom weight matrix. Must be numpy array. Watch the dtype
        :param set_bias: (optional) custom bias. Must be numpy array. Watch the dtype.
        :param bias: (optional) use bias
        :param fixed: (optional) don't train this
        """
        super(WordLinout, self).__init__(worddic)
        wdvals = worddic.values()
        assert(min(wdvals) >= 0)     # word ids must be positive

        # extract maskid and rareid from worddic
        maskid = worddic[self.masktoken] if self.masktoken in worddic else None
        rareid = worddic[self.raretoken] if self.raretoken in worddic else None
        self.maskid = maskid

        outdim = max(worddic.values())+1        # to init from worddic
        self.outdim = outdim
        self.indim = indim
        self.lin = nn.Linear(indim, outdim, bias=bias)

        if weight is not None:
            self.lin.weight = nn.Parameter(torch.from_numpy(weight))
        if set_bias is not None:
            self.lin.bias = nn.Parameter(torch.from_numpy(bias))
        if fixed is True:
            self.lin.weight.requires_grad = False
            if bias is True:
                self.lin.bias.requires_grad = False

    def _getvector(self, wordid):
        vec = self.lin.weight.index_select(0, wordid)
        return vec

    def forward(self, x, mask=None):
        ret = self.lin(x)
        ret = ret.mul(mask if mask is not None else 1)
        return ret#, mask ?


class ComputedWordLinout(WordLinoutBase):
    def __init__(self, data=None, computer=None, worddic=None, bias=False):
        """
        WordLinout that computes the weight matrix of the Linear transformation dynamically
        based on provided data and computer.
        :param data:    numpy array, 2D or more, one symbol data per row. Automatically wrapped so watch the dtype
        :param computer: module that builds vectors for rows of data
        :param worddic: token dictionary from token to id
        :param bias: (optional) use bias (not computed)
        """
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
            comp_weight = comp_weight.contiguous()
            indim = comp_weight.size(1)
            if self.base_weight is None or self.base_weight.size(1) != indim:
                self.base_weight = q.var(torch.zeros(1, indim)).cuda(x).v
            weight = torch.cat([self.base_weight, comp_weight], 0)
            index_transform = (torch.cumsum(msk, 0) * msk).long()
            weight = weight.gather(0, index_transform)
        else:
            weight = self.computer(self.data)
            weight = weight.contiguous()
        out = torch.mm(x, weight.t)
        if self.bias:
            bias = self.bias if mask is not None else self.bias * mask
            out += bias
        return out#, mask ?


class PretrainedWordLinout(WordLinout, PretrainedWordVec):
    def __init__(self, dim, vocabsize=None, path=None, fixed=True,
                 incl_maskid=True, incl_rareid=True, bias=False,
                 **kw):
        """
        WordLinout that sets the weight of the contained nn.Linear to loaded pretrained vectors.
        Adds a maskid and rareid as specified on the class.

        :param dim: token vector dimensions
        :param vocabsize: (optional) number of tokens to load
        :param path: (optional) where to load from.
                     Must be of format .../xxx%dxxx.
                     Files must be separated in .npy matrix and .words list.
                     Defaults to glove in qelos/data/.
        :param fixed: (optional) no learning. Disables bias automatically.
        :param incl_maskid: (optional) includes a <MASK> token in dictionary and assigns it id 0
        :param incl_rareid: (optional) includes a <RARE> token in dictionary and assigns it id 1 if incl_maskid was True, and id 0 otherwise
        :param bias: (optional) use bias. Default initial bias value is random (bias disabled when fixed=True).
        """
        assert ("worddic" not in kw)
        path = self._get_path(dim, path=path)
        value, wdic = self.loadvalue(path, dim, indim=vocabsize, maskid=incl_maskid, rareid=incl_rareid)
        self.allwords = wdic.keys()
        bias = bias and not fixed
        super(PretrainedWordLinout, self).__init__(dim, weight=value,
                                                   worddic=wdic, fixed=fixed, bias=bias,
                                                   **kw)


class AdaptedWordLinout(WordLinoutBase):
    def __init__(self, wordlinout, wdic, **kw):
        D = wordlinout.D
        # assert (self.raretoken in D)  # must have rareid in D to map extra words to it
        # assert(wordlinout.raretoken in wdic)
        super(AdaptedWordLinout, self).__init__(wdic, **kw)
        self.inner = wordlinout

        rareid_new2old = D[wordlinout.raretoken] if wordlinout.raretoken in D else 0
        rareid_old2new = wdic[self.raretoken] if self.raretoken in wdic else 0

        self.new_to_old_d = {v: D[k] if k in D else rareid_new2old
                   for k, v in wdic.items()}
        # mapping from new indexes (wdic) to old indexes (wordlinout)
        self.old_to_new_d = {v: wdic[k] if k in wdic else rareid_old2new
                             for k, v in D.items()}
        # mapping from old indexes (wordlinout) to new indexes (wdic)

        numnew = max(self.new_to_old_d.keys()) + 1
        numold = max(self.old_to_new_d.keys()) + 1

        new_to_old = np.zeros((numnew,), dtype="int64")
        for i in range(new_to_old.shape[0]):
            j = self.new_to_old_d[i] if i in self.new_to_old_d else rareid_new2old
            new_to_old[i] = j
        self.new_to_old = q.val(new_to_old).v  # for every new dic word id, contains old dic id
        # index in new dic contains idx value of old dic
        # --> used to slice from matrix in old idxs to get matrix in new idxs

        old_to_new = np.zeros((numold,), dtype="int64")
        for i in range(old_to_new.shape[0]):
            j = self.old_to_new_d[i] if i in self.old_to_new_d else rareid_old2new
            old_to_new[i] = j
        self.old_to_new = q.val(old_to_new).v  # for every old dic word id, contains new dic id

    def _getvector(self, wordid):
        wordid = self.new_to_old[wordid]
        return self.inner.lin.weight[wordid]

    def forward(self, x, mask=None):       # (batsize, indim), (batsize, outdim)
        innermask = mask.index_select(1, self.old_to_new) if mask is not None else None
        baseout = self.inner(x, mask=innermask)     # (batsize, outdim) --> need to permute columns
        out = baseout.index_select(1, self.new_to_old)
        return out#, mask?


class OverriddenWordLinout(OverriddenWordVecBase, WordLinoutBase):
    def forward(self, x, mask=None):    # (batsize, indim), (batsize, outdim)
        baseres = self.base(x, mask=mask)
        overres = self.over(x, mask=mask)
        res = self.overridemask.unsqueeze(0) * overres \
              + (1 - self.overridemask.unsqueeze(0)) * baseres
        if mask is not None:
            res = res * mask
        return res#, mask


class MergedWordLinout(MergedWordVecBase, WordLinoutBase):
    def forward(self, x, mask=None):
        baseres = self.base(x, mask=mask)
        mergres = self.merg(x, mask=mask)
        if self.mode == "sum":
            res = baseres + mergres
        elif self.mode == "cat":
            res = torch.cat([baseres, mergres], 1)
        else:
            raise q.SumTingWongException()
        return res