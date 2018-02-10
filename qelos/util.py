from __future__ import print_function
from __future__ import print_function
from __future__ import print_function
from __future__ import print_function
from __future__ import print_function
import argparse
import collections
import inspect
import re
import os
import signal
import sys
from datetime import datetime as dt
import dill as pickle

import nltk
import numpy as np
import unidecode
from IPython import embed


# torch-independent utils


class ticktock(object):
    def __init__(self, prefix="-", verbose=True):
        self.prefix = prefix
        self.verbose = verbose
        self.state = None
        self.perc = None
        self.prevperc = None
        self._tick()

    def tick(self, state=None):
        if self.verbose and state is not None:
            print("%s: %s" % (self.prefix, state))
        self._tick()

    def _tick(self):
        self.ticktime = dt.now()

    def _tock(self):
        return (dt.now() - self.ticktime).total_seconds()

    def progress(self, x, of, action="", live=False):
        if self.verbose:
            self.perc = int(round(100. * x / of))
            if self.perc != self.prevperc:
                if action != "":
                    action = " " + action + " -"
                topr = "%s:%s %d" % (self.prefix, action, self.perc) + "%"
                if live:
                    self._live(topr)
                else:
                    print(topr)
                self.prevperc = self.perc

    def tock(self, action=None, prefix=None):
        duration = self._tock()
        if self.verbose:
            prefix = prefix if prefix is not None else self.prefix
            action = action if action is not None else self.state
            print("%s: %s in %s" % (prefix, action, self._getdurationstr(duration)))
        return self

    def msg(self, action=None, prefix=None):
        if self.verbose:
            prefix = prefix if prefix is not None else self.prefix
            action = action if action is not None else self.state
            print("%s: %s" % (prefix, action))
        return self

    def _getdurationstr(self, duration):
        if duration >= 60:
            duration = int(round(duration))
            seconds = duration % 60
            minutes = (duration // 60) % 60
            hours = (duration // 3600) % 24
            days = duration // (3600*24)
            acc = ""
            if seconds > 0:
                acc = ("%d second" % seconds) + ("s" if seconds > 1 else "")
            if minutes > 0:
                acc = ("%d minute" % minutes) + ("s" if minutes > 1 else "") + (", " + acc if len(acc) > 0 else "")
            if hours > 0:
                acc = ("%d hour" % hours) + ("s" if hours > 1 else "") + (", " + acc if len(acc) > 0 else "")
            if days > 0:
                acc = ("%d day" % days) + ("s" if days > 1 else "") + (", " + acc if len(acc) > 0 else "")
            return acc
        else:
            return ("%.3f second" % duration) + ("s" if duration > 1 else "")

    def _live(self, x, right=None):
        if right:
            try:
                #ttyw = int(os.popen("stty size", "r").read().split()[1])
                raise Exception("qsdf")
            except Exception:
                ttyw = None
            if ttyw is not None:
                sys.stdout.write(x)
                sys.stdout.write(right.rjust(ttyw - len(x) - 2) + "\r")
            else:
                sys.stdout.write(x + "\t" + right + "\r")
        else:
            sys.stdout.write(x + "\r")
        sys.stdout.flush()

    def live(self, x):
        if self.verbose:
            self._live(self.prefix + ": " + x, "T: %s" % self._getdurationstr(self._tock()))

    def stoplive(self):
        if self.verbose:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()


def argparsify(f, test=None):
    args, _, _, defaults = inspect.getargspec(f)
    assert(len(args) == len(defaults))
    parser = argparse.ArgumentParser()
    i = 0
    for arg in args:
        argtype = type(defaults[i])
        if argtype == bool:     # convert to action
            if defaults[i] == False:
                action="store_true"
            else:
                action="store_false"
            parser.add_argument("-%s" % arg, "--%s" % arg, action=action, default=defaults[i])
        else:
            parser.add_argument("-%s"%arg, "--%s"%arg, type=type(defaults[i]))
        i += 1
    if test is not None:
        par = parser.parse_args([test])
    else:
        par = parser.parse_args()
    kwargs = {}
    for arg in args:
        if getattr(par, arg) is not None:
            kwargs[arg] = getattr(par, arg)
    return kwargs


def argprun(f, sigint_shell=True, **kwargs):   # command line overrides kwargs
    def handler(sig, frame):
        # find the frame right under the argprun
        print("custom handler called")
        original_frame = frame
        current_frame = original_frame
        previous_frame = None
        stop = False
        while not stop and current_frame.f_back is not None:
            previous_frame = current_frame
            current_frame = current_frame.f_back
            if "_FRAME_LEVEL" in current_frame.f_locals \
                and current_frame.f_locals["_FRAME_LEVEL"] == "ARGPRUN":
                stop = True
        if stop:    # argprun frame found
            __toexposelocals = previous_frame.f_locals     # f-level frame locals
            class L(object):
                pass
            l = L()
            for k, v in __toexposelocals.items():
                setattr(l, k, v)
            stopprompt = False
            while not stopprompt:
                whattodo = raw_input("(s)hell, (k)ill\n>>")
                if whattodo == "s":
                    embed()
                elif whattodo == "k":
                    "Killing"
                    sys.exit()
                else:
                    stopprompt = True

    if sigint_shell:
        _FRAME_LEVEL="ARGPRUN"
        prevhandler = signal.signal(signal.SIGINT, handler)
    try:
        f_args = argparsify(f)
        for k, v in kwargs.items():
            if k not in f_args:
                f_args[k] = v
        f(**f_args)

    except KeyboardInterrupt:
        print("Interrupted by Keyboard")


def inp():
    return raw_input("Press ENTER to continue:\n>>> ")


def issequence(x):
    return isinstance(x, collections.Sequence) and not isinstance(x, basestring)


def iscollection(x):
    return issequence(x) or isinstance(x, set)


def isnumber(x):
    return isinstance(x, float) or isinstance(x, int)


def isstring(x):
    return isinstance(x, basestring)


def iscallable(x):
    return hasattr(x, "__call__")


def isfunction(x):
    return iscallable(x)


def getnumargs(f):
    return len(inspect.getargspec(f).args)


class StringMatrix():
    protectedwords = ["<MASK>", "<RARE>", "<START>", "<END>"]

    def __init__(self, maxlen=None, freqcutoff=0, topnwords=None, indicate_start_end=False, indicate_start=False, indicate_end=False):
        self._strings = []
        self._wordcounts_original = dict(zip(self.protectedwords, [0] * len(self.protectedwords)))
        self._dictionary = dict(zip(self.protectedwords, range(len(self.protectedwords))))
        self._dictionary_external = False
        self._rd = None
        self._next_available_id = len(self._dictionary)
        self._maxlen = 0
        self._matrix = None
        self._max_allowable_length = maxlen
        self._rarefreq = freqcutoff
        self._topnwords = topnwords
        self._indic_e, self._indic_s = False, False
        if indicate_start_end:
            self._indic_s, self._indic_e = True, True
        if indicate_start:
            self._indic_s = indicate_start
        if indicate_end:
            self._indic_e = indicate_end
        self._rarewords = set()
        self.tokenize = tokenize
        self._cache_p = None

    def __len__(self):
        if self._matrix is None:
            return len(self._strings)
        else:
            return self.matrix.shape[0]

    def cached(self, p):
        self._cache_p = p
        if os.path.isfile(p):
            pickle.load()

    def __getitem__(self, item, *args):
        if self._matrix is None:
            return self._strings[item]
        else:
            ret = self.matrix[item]
            if len(args) == 1:
                ret = ret[args[0]]
            ret = self.pp(ret)
            return ret

    @property
    def numwords(self):
        return len(self._dictionary)

    @property
    def numrare(self):
        return len(self._rarewords)

    @property
    def matrix(self):
        if self._matrix is None:
            raise Exception("finalize first")
        return self._matrix

    @property
    def D(self):
        return self._dictionary

    def set_dictionary(self, d):
        """ dictionary set in this way is not allowed to grow,
        tokens missing from provided dictionary will be replaced with <RARE>
        provided dictionary must contain <RARE> if missing tokens are to be supported"""
        print("setting dictionary")
        self._dictionary_external = True
        self._dictionary = {}
        self._dictionary.update(d)
        self._next_available_id = max(self._dictionary.values()) + 1
        self._wordcounts_original = dict(zip(list(self._dictionary.keys()), [0]*len(self._dictionary)))
        self._rd = {v: k for k, v in self._dictionary.items()}

    @property
    def RD(self):
        return self._rd

    def d(self, x):
        return self._dictionary[x]

    def rd(self, x):
        return self._rd[x]

    def pp(self, matorvec):
        def pp_vec(vec):
            return " ".join([self.rd(x) if x in self._rd else "<UNK>" for x in vec if x != self.d("<MASK>")])
        ret = []
        if matorvec.ndim == 2:
            for vec in matorvec:
                ret.append(pp_vec(vec))
        else:
            return pp_vec(matorvec)
        return ret

    def add(self, x):
        tokens = self.tokenize(x)
        tokens = tokens[:self._max_allowable_length]
        if self._indic_s is not False and self._indic_s is not None:
            indic_s_sym = "<START>" if not isstring(self._indic_s) else self._indic_s
            tokens = [indic_s_sym] + tokens
        if self._indic_e is not False and self._indic_e is not None:
            indic_e_sym = "<END>" if not isstring(self._indic_e) else self._indic_e
            tokens = tokens + [indic_e_sym]
        self._maxlen = max(self._maxlen, len(tokens))
        tokenidxs = []
        for token in tokens:
            if token not in self._dictionary:
                if not self._dictionary_external:
                    self._dictionary[token] = self._next_available_id
                    self._next_available_id += 1
                    self._wordcounts_original[token] = 0
                else:
                    assert("<RARE>" in self._dictionary)
                    token = "<RARE>"    # replace tokens missing from external D with <RARE>
            self._wordcounts_original[token] += 1
            tokenidxs.append(self._dictionary[token])
        self._strings.append(tokenidxs)
        return len(self._strings)-1

    def finalize(self):
        ret = np.zeros((len(self._strings), self._maxlen), dtype="int64")
        for i, string in enumerate(self._strings):
            ret[i, :len(string)] = string
        self._matrix = ret
        self._do_rare_sorted()
        self._rd = {v: k for k, v in self._dictionary.items()}
        self._strings = None

    def _do_rare_sorted(self):
        """ if dictionary is not external, sorts dictionary by counts and applies rare frequency and dictionary is changed """
        if not self._dictionary_external:
            sortedwordidxs = [self.d(x) for x in self.protectedwords] + \
                             ([self.d(x) for x, y
                              in sorted(self._wordcounts_original.items(), key=lambda (x, y): y, reverse=True)
                              if y >= self._rarefreq and x not in self.protectedwords][:self._topnwords])
            transdic = zip(sortedwordidxs, range(len(sortedwordidxs)))
            transdic = dict(transdic)
            self._rarewords = {x for x in self._dictionary.keys() if self.d(x) not in transdic}
            rarewords = {self.d(x) for x in self._rarewords}
            self._numrare = len(rarewords)
            transdic.update(dict(zip(rarewords, [self.d("<RARE>")]*len(rarewords))))
            # translate matrix
            self._matrix = np.vectorize(lambda x: transdic[x])(self._matrix)
            # change dictionary
            self._dictionary = {k: transdic[v] for k, v in self._dictionary.items() if self.d(k) in sortedwordidxs}

    def save(self, p):
        pickle.dump(self, open(p, "w"))

    @staticmethod
    def load(p):
        if os.path.isfile(p):
            return pickle.load(open(p))
        else:
            return None


def tokenize(s, preserve_patterns=None, extrasubs=True):
    if not isinstance(s, unicode):
        s = s.decode("utf-8")
    s = unidecode.unidecode(s)
    repldic = None
    if preserve_patterns is not None:
        repldic = {}
        def _tokenize_preserve_repl(x):
            id = max(repldic.keys() + [-1]) + 1
            repl = "replreplrepl{}".format(id)
            assert(repl not in s)
            assert(id not in repldic)
            repldic[id] = x.group(0)
            return repl
        for preserve_pattern in preserve_patterns:
            s = re.sub(preserve_pattern, _tokenize_preserve_repl, s)
    if extrasubs:
        s = re.sub("[-_\{\}/]", " ", s)
    s = s.lower()
    tokens = nltk.word_tokenize(s)
    if repldic is not None:
        repldic = {"replreplrepl{}".format(k): v for k, v in repldic.items()}
        tokens = [repldic[token] if token in repldic else token for token in tokens]
    s = re.sub("`", "'", s)
    return tokens


class DictObj(object):
    def __init__(self, d):
        for ditem in d:
            self.__dict__[ditem] = d[ditem]


def dtoo(d):
    return DictObj(d)


class EmitStorage(object):
    storage = {}


def emit(name, dic):
    EmitStorage.storage[name] = dic


def get_emitted(name):
    return EmitStorage.storage[name]


import numpy as np


def wordmatfromdic(worddic, maxwordlen=30):
    maskid = -1
    rwd = sorted(worddic.items(), key=lambda (x, y): y)
    realmaxlen = 0
    wordmat = np.ones((rwd[-1][1]+1, maxwordlen), dtype="int32") * maskid
    for i in range(len(rwd)):
        rwdichars, rwdiidx = rwd[i]
        realmaxlen = max(realmaxlen, len(rwdichars))
        wordmat[rwdiidx, :min(len(rwdichars), maxwordlen)] \
            = [ord(c) for c in rwdichars[:min(len(rwdichars), maxwordlen)]]
    allchars = set(list(np.unique(wordmat))).difference({maskid})
    chardic = {maskid: maskid}
    chardic.update(dict(zip(allchars, range(len(allchars)))))
    wordmat = np.vectorize(lambda x: chardic[x])(wordmat)
    del chardic[maskid]
    chardic = {chr(k): v for k, v in chardic.items()}
    return wordmat, chardic


def wordmat2wordchartensor(wordmat, worddic=None, rwd=None, maxchars=30, maskid=-1):
    chartensor = wordmat2chartensor(wordmat, worddic=worddic, rwd=rwd, maxchars=maxchars, maskid=maskid)
    out = np.concatenate([wordmat[:, :, np.newaxis], chartensor], axis=2)
    #embed()
    return out


def wordmat2chartensor(wordmat, worddic=None, rwd=None, maxchars=30, maskid=-1):
    assert(worddic is not None or rwd is not None)
    assert(not(worddic is not None and rwd is not None))
    if rwd is None:
        rwd = {v: k for k, v in worddic.items()}
    wordcharmat = maskid * np.ones((max(rwd.keys())+1, maxchars), dtype="int32")
    realmaxlen = 0
    for i in rwd.keys():
        word = rwd[i]
        word = word[:min(maxchars, len(word))]
        realmaxlen = max(realmaxlen, len(word))
        wordcharmat[i, :len(word)] = [ord(ch) for ch in word]
    chartensor = wordcharmat[wordmat, :]
    chartensor[wordmat == -1] = -1
    if realmaxlen < maxchars:
        chartensor = chartensor[:, :, :realmaxlen]
    return chartensor


def wordmat2charmat(wordmat, worddic=None, rwd=None, maxlen=100, raretoken="<RARE>", maskid=-1):
    assert(worddic is not None or rwd is not None)
    assert(not(worddic is not None and rwd is not None))
    tt = ticktock("wordmat2charmat")
    tt.tick("transforming word mat to char mat")
    toolong = 0
    charmat = maskid * np.ones((wordmat.shape[0], maxlen), dtype="int32")
    if rwd is None:
        rwd = {v: (k if k != raretoken else " ")
               for k, v in worddic.items()}
    else:
        rwd = dict([(k, (v if v != raretoken else " "))
                   for k, v in rwd.items()])
    realmaxlen = 0
    for i in range(wordmat.shape[0]):
        s = wordids2string(wordmat[i], rwd, maskid=maskid)
        s = s[:min(len(s), maxlen)]
        realmaxlen = max(len(s), realmaxlen)
        if len(s) > maxlen:
            toolong += 1
        charmat[i, :len(s)] = [ord(ch) for ch in s]
        tt.progress(i, wordmat.shape[0], live=True)
    if realmaxlen < maxlen:
        charmat = charmat[:, :realmaxlen]
    if toolong > 0:
        print("{} too long".format(toolong))
    tt.tock("transformed")
    return charmat


def wordids2string(inp, rwd, maskid=-1, reverse=False):
    ret = [rwd[x] if x in rwd else "<???>" for x in inp if x != maskid]
    if reverse:
        ret.reverse()
    ret = " ".join(ret)
    return ret


def charids2string(inp, rcd=None, maskid=-1):
    if rcd is not None:
        ret = "".join([rcd[ch] if ch in rcd else "<???>"
                       for ch in inp if ch != maskid])
    else:
        ret = "".join([chr(ch) if ch != maskid else "" for ch in inp])
    return ret


def wordcharmat2string(inp, rcd=None, maskid=-1):
    if rcd is not None:
        tochar = np.vectorize(lambda x: rcd[x] if x != maskid else "" if x in rcd else "<???>")
    else:
        tochar = np.vectorize(lambda x: chr(x) if x != maskid else "")
    x = tochar(inp)
    acc = []
    for i in range(x.shape[0]):
        w = "".join(list(x[i]))
        acc.append(w)
    ret = " ".join([w for w in acc if len(w) > 0])
    return ret


def getmatrixvaluecounts(*x):
    x = np.concatenate([xe.flatten() for xe in x], axis=0)
    from pandas import Series
    pdx = Series(x)
    ret = pdx.value_counts()
    return dict(zip(ret.index, ret.values))


def _bad_sync_wordseq_flatcharseq(wordmat, charmat, wordstop=1, wordmaskid=0, charmaskid=0):     # (numex, seqlen)
    numwords = wordmat.shape[1]
    numwordsperrow = (wordmat != wordmaskid).sum(axis=1)
    stopsperrow = (charmat == wordstop).sum(axis=1)
    assert(np.allclose(numwordsperrow, stopsperrow))
    moststopsperrow = stopsperrow.max()
    addrange = np.arange(0, numwords+1)[np.newaxis, :].repeat(len(charmat), axis=0)
    toadd = addrange > stopsperrow[:, np.newaxis]
    toadd *= (wordstop-charmaskid)
    toadd += charmaskid
    charmat_x = np.concatenate([charmat, toadd], axis=1)
    (charmat_x == wordstop).sum(axis=1)


def slicer_from_flatcharseq(charmat, wordstop=1, numwords=None):
    stopsperrow = (charmat == wordstop).sum(axis=1)
    maxstopsperrow = stopsperrow.max()
    numwords = maxstopsperrow if numwords is None else numwords
    slicer = np.zeros((len(charmat), numwords), dtype="int32")
    slicermask = np.zeros_like(slicer).astype("int8")
    wordsperrow = np.zeros((len(charmat)), dtype="int32")
    for i in range(charmat.shape[0]):
        slicerpos = 0
        for j in range(charmat.shape[1]):
            if charmat[i, j] == wordstop:
                slicer[i, slicerpos:] = j
                slicermask[i, slicerpos] = 1
                slicerpos += 1
    wordsperrow = (slicermask == 1).sum(axis=1)
    return slicer, slicermask, wordsperrow


def split(npmats, splits=(80, 20), random=True):
    splits = np.round(len(npmats[0]) * np.cumsum(splits) / sum(splits)).astype("int32")

    whatsplit = np.zeros((len(npmats[0]),), dtype="int64")
    for i in range(1, len(splits)):
        a, b = splits[i-1], splits[i]
        whatsplit[a:b] = i

    if random is not False and random is not None:
        if isinstance(random, int):
            np.random.seed(random)
            random = True

        if random is True:
            np.random.shuffle(whatsplit)

    ret = []
    for i in range(0, len(splits)):
        splitmats = [npmat[whatsplit == i] for npmat in npmats]
        ret.append(splitmats)
    return ret


def log(p, mode="a", name="", body={}):
    indentlevel = 0
    acc = ""
    acc += "{}\n".format(name)
    acc += _rec_write(body, indentlevel=indentlevel+1)
    acc += "\n"
    if p is None:
        print(acc)
        return acc
    else:
        with open(p, mode) as f:
            f.write(acc)


def _rec_write(d, indentlevel=0):
    ret = ""
    for k, v in d.items():
        indent = "".join(["\t"]*indentlevel)
        if isinstance(v, dict):
            ret += "{}{}:\n".format(indent, k)
            ret += _rec_write(v, indentlevel=indentlevel+1)
        else:
            ret += "{}{}: {}\n".format(indent, k, str(v))
    return ret


class kw2dict(object):
    def __init__(self, **kw):
        super(kw2dict, self).__init__()
        self._kw = kw

    def __call__(self, **kw):
        self._kw.update(kw)

    @property
    def v(self):
        return self._kw


def save_sparse_tensor(x, f):
    shape = np.asarray(x.shape, dtype="int64")
    coords = np.argwhere(x)
    indexers = [coords[:, i] for i in range(coords.shape[1])]
    values = x[indexers]
    np.savez(f, coords, values, shape)


def load_sparse_tensor(f):
    npzl = np.load(f)
    coords, values, shape = npzl["arr_0"], npzl["arr_1"], npzl["arr_2"]
    x = np.zeros(tuple(shape), dtype=values.dtype)
    indexers = [coords[:, i] for i in range(coords.shape[1])]
    x[indexers] = values
    return x


def makeiter(dl, unwrap=True):
    def inner():
        for i in dl:
            yield i

    dli = inner()
    while True:
        try:
            ret = next(dli)
            if unwrap:
                yield ret[0]
            else:
                yield ret
        except StopIteration as e:
            dli = inner()


def getkw(kw, name, default=None, nodefault=False, remove=True):
    if name in kw:
        ret = kw[name]
        if remove:
            del kw[name]
    else:
        if nodefault:
            raise Exception("kwarg {} must be specified (no default)".format(name))
        ret = default
    return ret


