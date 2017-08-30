import argparse
import collections
import inspect
import re
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
            print "%s: %s" % (self.prefix, state)
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
            print "%s: %s in %s" % (prefix, action, self._getdurationstr(duration))
        return self

    def msg(self, action=None, prefix=None):
        if self.verbose:
            prefix = prefix if prefix is not None else self.prefix
            action = action if action is not None else self.state
            print "%s: %s" % (prefix, action)
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
        print "custom handler called"
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

    def __init__(self, maxlen=None, freqcutoff=0, topnwords=None, indicate_start_end=False):
        self._strings = []
        self._wordcounts_original = dict(zip(self.protectedwords, [0] * len(self.protectedwords)))
        self._dictionary = dict(zip(self.protectedwords, range(len(self.protectedwords))))
        self._rd = None
        self._next_available_id = len(self._dictionary)
        self._maxlen = 0
        self._matrix = None
        self._max_allowable_length = maxlen
        self._rarefreq = freqcutoff
        self._topnwords = topnwords
        self._indic_s_e = indicate_start_end
        self._rarewords = set()
        self.tokenize = tokenize

    def __len__(self):
        if self._matrix is None:
            return len(self._strings)
        else:
            return self.matrix.shape[0]

    def __getitem__(self, item):
        if self._matrix is None:
            return self._strings[item]
        else:
            return self.pp(self.matrix[item])

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
        if self._indic_s_e:
            tokens = ["<START>"] + tokens + ["<END>"]
        self._maxlen = max(self._maxlen, len(tokens))
        tokenidxs = []
        for token in tokens:
            if token not in self._dictionary:
                self._dictionary[token] = self._next_available_id
                self._next_available_id += 1
                self._wordcounts_original[token] = 0
            self._wordcounts_original[token] += 1
            tokenidxs.append(self._dictionary[token])
        self._strings.append(tokenidxs)
        return len(self._strings)-1

    def finalize(self):
        ret = np.zeros((len(self._strings), self._maxlen), dtype="int32")
        for i, string in enumerate(self._strings):
            ret[i, :len(string)] = string
        self._matrix = ret
        self._do_rare()
        self._rd = {v: k for k, v in self._dictionary.items()}
        self._strings = None

    def _do_rare(self):
        sortedwordidxs = [self.d(x) for x in self.protectedwords] + \
                         ([self.d(x) for x, y
                          in sorted(self._wordcounts_original.items(), key=lambda (x,y): y, reverse=True)
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
        return pickle.load(open(p))


def tokenize(s, preserve_patterns=None):
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
    s = re.sub("[-_\{\}/]", " ", s)
    s = s.lower()
    tokens = nltk.word_tokenize(s)
    if repldic is not None:
        repldic = {"replreplrepl{}".format(k): v for k, v in repldic.items()}
        tokens = [repldic[token] if token in repldic else token for token in tokens]
    s = re.sub("`", "'", s)
    return tokens