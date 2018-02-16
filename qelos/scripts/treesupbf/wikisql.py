# -*- coding: utf-8 -*-
import codecs
import json
import pickle as pkl
import re
from collections import OrderedDict

import nltk
import numpy as np
import torch

import qelos as q
from qelos.scripts.treesupbf.trees import Node, NodeTrackerDF
from qelos.scripts.treesupbf.pasdecode import TreeAccuracy
import random
from unidecode import unidecode


_opt_test = True


# region DATA
# region PREPARING DATA
def prepare_data(p="../../../datasets/wikisql/"):
    # load all jsons
    tt = q.ticktock("data preparer")
    tt.tick("loading jsons")
    traindata = read_jsonl(p+"train.jsonl")
    print("{} training questions".format(len(traindata)))
    traindb = read_jsonl(p+"train.tables.jsonl")
    print("{} training tables".format(len(traindb)))
    devdata = read_jsonl(p + "dev.jsonl")
    print("{} dev questions".format(len(devdata)))
    devdb = read_jsonl(p+"dev.tables.jsonl")
    print("{} dev tables".format(len(devdb)))
    testdata = read_jsonl(p + "test.jsonl")
    print("{} test questions".format(len(testdata)))
    testdb = read_jsonl(p + "test.tables.jsonl")
    print("{} test tables".format(len(testdb)))

    alltables = {}
    for table in traindb + devdb + testdb:
        alltables[table["id"]] = table

    print("{} all tables".format(len(alltables)))

    tt.tock("jsons loaded")

    tt.tick("generating examples")

    # make text lines for each example
    # format: <question words> \t answer \t <column names>
    tt.msg("train examples")
    trainexamples = []
    for line in traindata:
        try:
            trainexamples.append(make_example(line, alltables))
        except Exception as e:
            print("FAILED: {}".format(line))

    tt.msg("dev examples")
    devexamples = []
    for line in devdata:
        try:
            devexamples.append(make_example(line, alltables))
        except Exception as e:
            print("FAILED: {}".format(line))
    tt.msg("test examples")

    testexamples = []
    for line in testdata:
        try:
            testexamples.append(make_example(line, alltables))
        except Exception as e:
            print("FAILED: {}".format(line))

    tt.tock("examples generated")

    print("\n".join(trainexamples[:10]))

    with codecs.open(p + "train.lines", "w", encoding="utf-8") as f:
        for example in trainexamples:
            f.write(u"{}\n".format(example))
    with codecs.open(p+"dev.lines", "w", encoding="utf-8") as f:
        for example in devexamples:
            f.write(u"{}\n".format(example))
    with codecs.open(p+"test.lines", "w", encoding="utf-8") as f:
        for example in testexamples:
            f.write(u"{}\n".format(example))

    return trainexamples, devexamples, testexamples


def read_jsonl(p):
    lines = []
    with open(p) as f:
        for line in f:
            example = json.loads(line)
            lines.append(example)
    return lines


def make_example(line, alltables):
    column_names = alltables[line["table_id"]]["header"]
    question = u" ".join(nltk.word_tokenize(line["question"])).lower()
    # question = unidecode.unidecode(question)
    question = question.replace(u"`", u"'")
    question = question.replace(u"''", u'"')
    question = question.replace(u'\u20ac', u' \u20ac ')
    question = question.replace(u'\uffe5', u' \uffe5 ')
    question = question.replace(u'\u00a3', u' \u00a3 ')
    question = re.sub(u'/$', u' /', question)
    question = question.replace(u"'", u" ' ")
    question = re.sub(u'(\d+[,\.]\d+)(k?m)', u'\g<1> \g<2>', question)

    question = re.sub(u'\s+', u' ', question)
    # question = " ".join(q.tokenize(question, extrasubs=False))
    # question = question.replace("=", " = ")
    sql_select = u"AGG{} COL{}".format(line["sql"]["agg"], line["sql"]["sel"])
    sql_wheres = []
    for cond in line["sql"]["conds"]:
        if isinstance(cond[2], float):
            condval = unicode(cond[2].__repr__())
        else:
            condval = unicode(cond[2]).lower()
        # condval = unidecode.unidecode(condval)
        condval = condval.replace(u"`", u"'")
        condval = condval.replace(u"''", u'"')
        _condval = condval.replace(u" ", u"")
        condval = None
        if _condval == u'southaustralia':
            pass
        _condvalacc = []
        __condval = []
        questionsplit = question.split()

        for i, qword in enumerate(questionsplit):
            if qword in _condval:
                found = False
                for j in range(i+1, len(questionsplit)+1):
                    if u"".join(questionsplit[i:j]) in _condval:
                        if u"".join(questionsplit[i:j]) == _condval:
                            condval = u" ".join(questionsplit[i:j])
                            found = True
                            break
                    else:
                        break
                if found:
                    break
        assert(condval in question)

        condl = u"<COND> COL{} OP{} <VAL> {} <ENDVAL>".format(cond[0], cond[1], condval)
        sql_wheres.append(condl)
    if len(sql_wheres) > 0:
        sql = u"<QUERY> <SELECT> {} <WHERE> {}".format(sql_select, u" ".join(sql_wheres))
    else:
        sql = u"<QUERY> <SELECT> {}".format(sql_select)
    ret = u"{}\t{}\t{}".format(question, sql, u"\t".join(column_names))
    return ret
# endregion


# region LOADING DATA
def create_mats(p="../../../datasets/wikisql/"):
    # loading lines
    tt = q.ticktock("data loader")
    tt.tick("loading lines")
    trainlines = load_lines(p+"train.lines")
    print("{} train lines".format(len(trainlines)))
    devlines = load_lines(p+"dev.lines")
    print("{} dev lines".format(len(devlines)))
    testlines = load_lines(p+"test.lines")
    print("{} test lines".format(len(testlines)))
    tt.tock("lines loaded")

    # preparing matrices
    tt.tick("preparing matrices")
    i = 0
    devstart, teststart = 0, 0

    # region gather original dictionary
    ism = q.StringMatrix()
    ism.tokenize = lambda x: x.split()
    numberunique = 0
    for question, query, columns in trainlines:
        numberunique = max(numberunique, len(set(question.split())))
        ism.add(question)
        i += 1
    devstart = i
    for question, query, columns in devlines:
        numberunique = max(numberunique, len(set(question.split())))
        ism.add(question)
        i += 1
    teststart = i
    for question, query, columns in testlines:
        numberunique = max(numberunique, len(set(question.split())))
        ism.add(question)
        i += 1
    ism.finalize()
    print("max number unique words in a question: {}".format(numberunique))
    # endregion

    pedics = np.ones((ism.matrix.shape[0], numberunique+3), dtype="int64")      # per-example dictionary, mapping UWID position to GWID
                                                                                # , mapping unused UWIDs to <RARE> GWID
    pedics = pedics * ism.D["<RARE>"]
    pedics[:, 0] = ism.D["<MASK>"]
    uwidmat = np.zeros_like(ism.matrix)                         # UWIDs for questions

    rD = {v: k for k, v in ism.D.items()}

    gtoudics = []

    for i in range(len(ism.matrix)):
        row = ism.matrix[i]
        gwidtouwid = {"<MASK>": 0}
        for j in range(len(row)):
            k = row[j]
            if rD[k] not in gwidtouwid:
                gwidtouwid[rD[k]] = len(gwidtouwid)
                pedics[i, gwidtouwid[rD[k]]] = k
            uwidmat[i, j] = gwidtouwid[rD[k]]
        gtoudics.append(gwidtouwid)

    uwidD = dict(zip(["<MASK>"] + ["UWID{}".format(i+1) for i in range(pedics.shape[1]-1)], range(pedics.shape[1])))

    osm = q.StringMatrix(indicate_start=True, indicate_end=True)
    osm.tokenize = lambda x: x.split()
    i = 0
    for _, query, columns in trainlines+devlines+testlines:
        splits = query.split()
        gwidtouwid = gtoudics[i]
        splits = ["UWID{}".format(gwidtouwid[e]) if e in gwidtouwid else e for e in splits]
        _q = " ".join(splits)
        osm.add(_q)
        i += 1
    osm.finalize()

    e2cn = np.zeros((len(osm), 44), dtype="int64")
    uniquecolnames = OrderedDict({"nonecolumnnonecolumnnonecolumn": 0})
    i = 0
    for _, query, columns in trainlines+devlines+testlines:
        j = 0
        for column in columns:
            if column not in uniquecolnames:
                uniquecolnames[column] = len(uniquecolnames)
            e2cn[i, j] = uniquecolnames[column]
            j += 1
        i += 1

    csm = q.StringMatrix(indicate_start=False, indicate_end=False)
    for i, columnname in enumerate(uniquecolnames.keys()):
        if columnname == u'№':
            columnname = u'number'
        csm.add(columnname)
    csm.finalize()
    # idx 3986 is zero because it's u'№' and unidecode makes it empty string, has 30+ occurrences !!! --> REPLACE

    assert(len(np.argwhere(csm.matrix[:, 0] == 0)) == 0)

    with open(p+"matcache.mats", "w") as f:
        np.savez(f, ism=uwidmat, osm=osm.matrix, csm=csm.matrix, pedics=pedics, e2cn=e2cn)
    with open(p+"matcache.dics", "w") as f:
        dics = {"ism": uwidD, "osm": osm.D, "csm": csm.D, "pedics": ism.D, "sizes": (devstart, teststart)}
        pkl.dump(dics, f, protocol=pkl.HIGHEST_PROTOCOL)

    # print("question dic size: {}".format(len(ism.D)))
    # print("question matrix size: {}".format(ism.matrix.shape))
    # print("query dic size: {}".format(len(osm.D)))
    # print("query matrix size: {}".format(osm.matrix.shape))
    tt.tock("matrices prepared")
    pass


def load_lines(p):
    retlines = []
    with codecs.open(p, encoding="utf-8") as f:
        for line in f:
            linesplits = line.strip().split("\t")
            assert(len(linesplits) > 2)
            retlines.append((linesplits[0], linesplits[1], linesplits[2:]))
    return retlines


def load_matrices(p="../../../datasets/wikisql/"):
    tt = q.ticktock("matrix loader")
    tt.tick("loading matrices")
    with open(p+"matcache.mats") as f:
        mats = np.load(f)
        ismmat, osmmat, csmmat, pedics, e2cn \
            = mats["ism"], mats["osm"], mats["csm"], mats["pedics"], mats["e2cn"]
    tt.tock("matrices loaded")
    print(ismmat.shape)
    tt.tick("loading dics")
    with open(p+"matcache.dics") as f:
        dics = pkl.load(f)
        ismD, osmD, csmD, pedicsD, splits = dics["ism"], dics["osm"], dics["csm"], dics["pedics"], dics["sizes"]
    tt.tock("dics loaded")
    print(len(ismD))
    ism = q.StringMatrix()
    ism.set_dictionary(ismD)
    ism._matrix = ismmat
    osm = q.StringMatrix()
    osm.set_dictionary(osmD)
    osm._matrix = osmmat
    csm = q.StringMatrix()
    csm.set_dictionary(csmD)
    csm._matrix = csmmat
    psm = q.StringMatrix()
    psm.set_dictionary(pedicsD)
    psm._matrix = pedics
    # q.embed()
    return ism, osm, csm, psm, splits, e2cn


def ppq(i, ism, psm):
    return psm.pp(psm.matrix[i][ism.matrix[i]])

# endregion
# endregion


# region SQL TREES
def order_adder_wikisql(parse):
    # add order:
    def order_adder_rec(y):
        for i, ychild in enumerate(y.children):
            if y.name == "<VAL>":
                ychild.order = i
            order_adder_rec(ychild)

    order_adder_rec(parse)
    return parse


class SqlNode(Node):
    name2ctrl = {
        "<SELECT>": "A",
        "<WHERE>":  "A",
        "<COND>":   "A",
        "COL\d+":   "NC",
        "AGG\d+":   "NC",
        "OP\d+":    "NC",
        "<VAL>":    "A",
        "<ENDVAL>": "NC",
        "UWID\d+":  "NC",
    }

    def __init__(self, name, order=None, children=tuple(), **kw):
        super(SqlNode, self).__init__(name, order=order, children=children, **kw)

    def __eq__(self, other):
        return super(SqlNode, self).__eq__(other)

    @classmethod
    def parse_df(cls, inp, _toprec=True):
        if _toprec:
            parse = super(SqlNode, cls).parse_df(inp, _toprec=_toprec)
            order_adder_wikisql(parse)
            return parse
        else:
            return super(SqlNode, cls).parse_df(inp, _toprec)

    @classmethod
    def parse_sql(cls, inp, _rec_arg=None, _toprec=True, _ret_remainder=False):
        """ ONLY FOR ORIGINAL LIN, FOR NORMAL DF WITH ANNOTATIONS USE .parse_df()
            * Automatically assigns order to children of <VAL> !!! """
        if len(inp) == 0:
            return []
        tokens = inp
        parent = _rec_arg
        if _toprec:
            tokens = tokens.replace("  ", " ").strip().split()
        head = tokens[0]
        tail = tokens[1:]

        siblings = []
        jumpers = {"<SELECT>": {"<QUERY>"},
                   "<WHERE>": {"<QUERY>"},
                   "<COND>": {"<WHERE>"},
                   "<VAL>": {"<COND>"},}
        while True:
            head, islast, isleaf = head, None, None

            headsplits = head.split(SqlNode.suffix_sep)
            if len(headsplits) > 1:
                head, islast, isleaf = headsplits[0], SqlNode.last_suffix in headsplits, SqlNode.leaf_suffix in headsplits

            if head == "<QUERY>":
                assert(isleaf is None or isleaf is False)
                assert(islast is None or islast is True)
                children, tail = cls.parse_sql(tail, _rec_arg=head, _toprec=False)
                ret = SqlNode(head, children=children)
                break
            elif head == "<END>":
                ret = siblings, []
                break
            elif head in jumpers:
                assert(isleaf is None or isleaf is False)
                if _rec_arg in jumpers[head]:
                    children, tail = cls.parse_sql(tail, _rec_arg=head, _toprec=False)
                    if head == "<VAL>":
                        for i, child in enumerate(children):
                            child.order = i
                    node = SqlNode(head, children=children)
                    siblings.append(node)
                    if len(tail) > 0:
                        head, tail = tail[0], tail[1:]
                    else:
                        ret = siblings, tail
                        break
                else:
                    ret = siblings, [head] + tail
                    break
            else:
                assert(isleaf is None or isleaf is True)
                node = SqlNode(head)
                siblings.append(node)
                if head == "<ENDVAL>" or len(tail) == 0:
                    ret = siblings, tail
                    break
                else:
                    head, tail = tail[0], tail[1:]
        if isinstance(ret, tuple):
            if _toprec:
                raise q.SumTingWongException("didn't parse SQL in .parse_sql()")
            else:
                return ret
        else:
            return ret

    @classmethod
    def parse(cls, inp, _rec_arg=None, _toprec=True, _ret_remainder=False):
        tokens = inp
        if _toprec:
            tokens = tokens.replace("  ", " ").strip().split()
            for i in range(len(tokens)):
                splits = tokens[i].split("*")
                if len(splits) == 1:
                    token, suffix = splits[0], ""
                else:
                    token, suffix = splits[0], "*"+splits[1]
                if token not in "<QUERY> <SELECT> <WHERE> <COND> <VAL>".split():
                    suffix = "*NC" + suffix
                tokens[i] = token + suffix
            ret = super(SqlNode, cls).parse(" ".join(tokens), _rec_arg=None, _toprec=True)
            return ret
        else:
            return super(SqlNode, cls).parse(tokens, _rec_arg=_rec_arg, _toprec=_toprec)

    # def symbol(self, with_label=True, with_annotation=True, with_order=True):
    #     s = super(SqlNode, self).symbol(with_order=with_order, with_label=with_label, with_annotation=with_annotation)
    #     if isinstance(s, unicode):
    #         s = unidecode(s)
    #     return s


def make_tracker(osm):
    tt = q.ticktock("tree tracker maker")
    tt.tick("flushing trees")
    trees = []
    for i in range(len(osm.matrix)):
        tree = SqlNode.parse_sql(osm[i])
        trees.append(tree)
    xD = {}
    xD.update(osm.D)
    xD.update(dict([(k+"*LS", v+len(osm.D)) for k, v in osm.D.items()]))
    tracker = SqlGroupTracker(trees, osm.D, xD)

    if True:    # TEST
        accs = set()
        for j in range(200):
            acc = u""
            tracker.reset()
            for i in range(25):
                vnt = tracker.get_valid_next(0)
                sel = random.choice(vnt)
                acc += u" " + tracker.rD[sel]
                tracker.update(0, sel)
            accs.add(acc)
        print("number of unique linearizations for example 0", len(accs))
        for acc in accs:
            pacc = SqlNode.parse(unidecode(acc))
            print(pacc.pptree())

    tt.tock("trees flushed")
    return tracker


class SqlGroupTracker(object):
    def __init__(self, trackables, coreD, outD):
        super(SqlGroupTracker, self).__init__()
        self.trackables = trackables
        self.D = outD
        self.coreD = coreD
        self.rD = {v: k for k, v in self.D.items()}
        self.trackers = []
        for xe in self.trackables:
            tracker = xe.track()
            self.trackers.append(tracker)
        self._dirty_ids = set()

    def get_valid_next(self, eid):
        tracker = self.trackers[eid]
        nvt = tracker._nvt
        if len(nvt) == 0:
            # nvt = {u"<RARE>"}
            nvt = {u'<MASK>'}           # <-- why was rare? loss handles -inf on mask now
        _nvt = set()
        for x in nvt:
            x = x.replace(u"*NC", u"")
            _nvt.add(x)
        nvt = map(lambda x: self.D[x], _nvt)
        return nvt

    def update(self, eid, x, altx=None):
        tracker = self.trackers[eid]
        self._dirty_ids.add(eid)
        nvt = tracker._nvt
        if len(nvt) == 0:
            pass
        else:
            x = self.rD[x]
            xsplits = x.split(u"*")
            core, suffix = xsplits[0], u""
            if len(xsplits) > 1:
                suffix = u"*" + xsplits[1]
            if core not in u"<QUERY> <SELECT> <WHERE> <COND> <VAL>".split():
                suffix = u"*NC" + suffix
            x = core + suffix
            tracker.nxt(x)

    def is_terminated(self, eid):
        return self.trackers[eid].is_terminated()

    def reset(self, *which, **kw):
        force = q.getkw(kw, "force", default=False)
        if len(which) > 0:
            for w in which:
                self.trackers[w].reset()
        else:
            if not force and len(self._dirty_ids) > 0:
                self.reset(*list(self._dirty_ids))
            else:
                for tracker in self.trackers:
                    tracker.reset()


def make_tracker_df(osm):
    tt = q.ticktock("tree tracker maker")
    tt.tick("making trees")
    trees = []
    for i in range(len(osm.matrix)):
        tree = SqlNode.parse_sql(osm[i])
        trees.append(tree)
    tracker = SqlGroupTrackerDF(trees, osm.D)
    tt.tock("trees made")

    if False:
        for k in range(100):
            accs = set()
            for j in range(500):
                acc = u""
                tracker.reset()
                while True:
                    if tracker.is_terminated(k):
                        break
                    vnt = tracker.get_valid_next(k)
                    sel = random.choice(vnt)
                    acc += u" " + tracker.rD[sel]
                    tracker.update(k, sel)
                accs.add(acc)
                if not SqlNode.parse_sql(unidecode(acc)).equals(tracker.trackables[k]):
                    print(tracker.trackables[k].pptree())
                    print(SqlNode.parse_sql(unidecode(acc)).pptree())
                    raise q.SumTingWongException("trees not equal")
            print("number of unique linearizations for example {}: {}".format(k, len(accs)))
    return tracker


class SqlGroupTrackerDF(object):
    def __init__(self, trackables, coreD):
        super(SqlGroupTrackerDF, self).__init__()
        self.trackables = trackables
        self.D = coreD
        self.rD = {v: k for k, v in self.D.items()}
        self.trackers = []
        for xe in self.trackables:
            tracker = NodeTrackerDF(xe)
            self.trackers.append(tracker)
        self._dirty_ids = set()
        self._did_the_end = [False] * len(self.trackers)

    def reset(self, *which, **kw):
        force = q.getkw(kw, "force", default=False)
        if len(which) > 0:
            for w in which:
                self.trackers[w].reset()
                self._did_the_end[w] = False
        else:
            if not force and len(self._dirty_ids) > 0:
                self.reset(*list(self._dirty_ids))
            else:
                for tracker in self.trackers:
                    tracker.reset()
                self._did_the_end = [False] * len(self.trackers)

    def get_valid_next(self, eid):
        tracker = self.trackers[eid]
        nvt = tracker._nvt      # with structure annotation
        if len(nvt) == 0:
            if self._did_the_end[eid] is True:
                # nvt = {u'<RARE>'}
                nvt = {u'<MASK>'}           # <-- why was rare? loss handles -inf on mask now
            else:
                nvt = {u"<END>"}
                self._did_the_end[eid] = True
                self._dirty_ids.add(eid)
        _nvt = set()
        for x in nvt:
            x = x.replace(u'*NC', u'').replace(u'*LS', u'')
            _nvt.add(x)
        nvt = map(lambda x: self.D[x], _nvt)
        return nvt

    def update(self, eid, x, alt_x=None):
        tracker = self.trackers[eid]
        self._dirty_ids.add(eid)
        nvt = tracker._nvt
        if len(nvt) == 0:
            pass
        else:
            core = self.rD[x]
            suffix = u''
            if core not in u"<QUERY> <SELECT> <WHERE> <COND> <VAL>".split():
                suffix += u'*NC'
            if core in u'<ENDVAL> <END> <QUERY>'.split():
                suffix += u'*LS'
            else:       # check previous _nvt, if it occurred as *LS there, do *LS
                if core + suffix in nvt and not (core + suffix + u'*LS' in nvt):
                    suffix += u''
                elif core + suffix + u'*LS' in nvt and not (core + suffix in nvt):
                    suffix += u'*LS'
                else:
                    raise q.SumTingWongException("some ting wong in sql tracker df")
            x = core + suffix
            tracker.nxt(x)

    def is_terminated(self, eid):
        return self.trackers[eid].is_terminated() and self._did_the_end[eid] is True


def make_struct_linout(lindim, baselinout, coreD, outD, tie_weights=False, chained=False, ttt=None):
    ttt = q.ticktock("linout test") if ttt is None else ttt

    for k, v in sorted(outD.items(), key=lambda (x, y): y):
        ksplits = k.split("*")
        if len(ksplits) == 2:
            assert(ksplits[1] == "LS")
            assert(coreD[ksplits[0]] == (v - len(coreD)))
        else:
            assert(coreD[ksplits[0]] == v)
    assert(baselinout.D == coreD)

    class StrucSMO(torch.nn.Module):
        def __init__(self, indim, corelinout, worddic=None, chained=True, **kw):
            super(StrucSMO, self).__init__(**kw)
            self.dim = indim
            self.D = worddic
            self.chained = chained
            self.corelinout = corelinout

            annD = {"A": 0, "LS": 1}
            self.annout = q.WordLinout(self.dim + 2, worddic=annD, bias=False)

            self.coreemb = q.WordEmb(self.dim, worddic=coreD)
            self.coreemb.embedding.weight.data.fill_(0)

            appendix = torch.zeros(1, len(coreD), 2)
            if self.chained:
                appendix[:, :, 1] = 1.    # LS
            self.appendix = q.val(appendix).v

        def forward(self, x):
            # predict core:
            coreprobs = self.corelinout(x)      # (batsize, coreD)
            coreprobs = torch.nn.LogSoftmax(1)(coreprobs).unsqueeze(2)

            # prepare predict LS
            appendix = self.appendix.repeat(x.size(0), 1, 1)
            addition = self.coreemb.embedding.weight.unsqueeze(0)
            if not self.chained:
                addition = q.var(torch.zeros(addition.size())).cuda(addition).v
            predvec = torch.cat([appendix, x.unsqueeze(1) + addition], 2)

            # predict LS
            ctrlprobs = self.annout(predvec)
            ctrlprobs = torch.nn.LogSoftmax(2)(ctrlprobs)

            # merge and flatten
            allprobs = coreprobs + ctrlprobs
            outprobs = torch.cat([allprobs[:, :, 0], allprobs[:, :, 1]], 1)
            return outprobs

    ret = StrucSMO(lindim, baselinout, worddic=outD, chained=chained)
    # TODO TEST
    return ret


def make_struct_emb(embdim, baseemb, coreD, outD, ttt=None):
    ttt = q.ticktock("struct emb test") if ttt is None else ttt

    for k, v in sorted(outD.items(), key=lambda (x, y): y):
        ksplits = k.split("*")
        if len(ksplits) == 2:
            assert(ksplits[1] == "LS")
            assert(coreD[ksplits[0]] == (v - len(coreD)))
        else:
            assert(coreD[ksplits[0]] == v)
    assert(baseemb.D == coreD)

    to_cores = np.arange((len(coreD),), dtype="int64")
    to_cores = np.stack([to_cores, to_cores], 0)
    to_ls = np.zeros((len(outD),), dtype="int64")
    to_ls[len(coreD):] = 1

    class StructEmb(torch.nn.Module):
        def __init__(self, indim, coreemb, worddic=None, **kw):
            super(StructEmb, self).__init__(**kw)
            self.dim = indim
            self.D = worddic
            self.base = coreemb

            self.to_cores = q.val(to_cores).v
            self.to_ls = q.val(to_ls).v

            annD = {"A": 0, "LS": 1}
            self.annemb = q.WordEmb(self.dim, worddic=annD)
            self.annemb.embedding.weight.fill_(0)

        def forward(self, x):   # input is indexes in outD
            x_cores = self.to_cores[x]
            x_ls = self.to_ls[x]
            coreembs, mask = self.base(x_cores)
            lsembs = self.annemb(x_ls)
            # additive ls emb
            embs = coreembs + lsembs
            return embs, mask

    ret = StructEmb(embdim, baseemb, worddic=outD)
    # TODO TEST
    return ret
# endregion


# region DYNAMIC VECTORS
from qelos.word import WordEmbBase, WordLinoutBase


class DynamicVecComputer(torch.nn.Module):
    pass


class DynamicVecPreparer(torch.nn.Module):
    pass


class DynamicWordEmb(WordEmbBase):
    """ Dynamic Word Emb dynamically computes word embeddings on a per-example basis
        based on the given batch of word ids and batch of data.
        Basically a dynamic-data ComputedWordEmb. """
    def __init__(self, computer=None, worddic=None, **kw):
        super(DynamicWordEmb, self).__init__(worddic=worddic)
        self.computer = computer
        maskid = worddic[self.masktoken] if self.masktoken in worddic else None
        self.maskid = maskid
        self.indim = max(worddic.values()) + 1
        self.saved_data = None

    def prepare(self, *xdata):
        if isinstance(self.computer, DynamicVecPreparer):
            ret, _ = self.computer.prepare(*xdata)
            self.saved_data = ret
        else:
            self.saved_data = xdata

    def forward(self, x):
        mask = None
        if self.maskid is not None:
            mask = x != self.maskid
        emb = self._forward(x)
        return emb, mask

    def _forward(self, x):
        if isinstance(self.computer, DynamicVecComputer):
            return self.computer(x, *self.saved_data)
        else:       # default implementation
            assert(isinstance(self.computer, DynamicVecPreparer))
            vecs = self.saved_data
            xdim = x.dim()
            if xdim == 1:
                x = x.unsqueeze(1)
            ret = vecs.gather(1, x.clone().unsqueeze(2).repeat(1, 1, vecs.size(2)))
            if xdim == 1:
                ret = ret.squeeze(1)
            return ret


class DynamicWordLinout(WordLinoutBase):
    def __init__(self, computer=None, worddic=None, selfsm=True):
        super(DynamicWordLinout, self).__init__(worddic)
        self.computer = computer
        maskid = worddic[self.masktoken] if self.masktoken in worddic else None
        self.maskid = maskid
        self.outdim = max(worddic.values()) + 1
        self.saved_data = None
        self.sm = q.LogSoftmax() if selfsm else None

    def prepare(self, *xdata):
        assert(isinstance(self.computer, DynamicVecPreparer))
        ret = self.computer.prepare(*xdata)
        self.saved_data = ret

    def forward(self, x, mask=None, _no_mask_log=False, **kw):
        ret, rmask = self._forward(x)
        if rmask is not None:
            if mask is None:
                mask = rmask
            else:
                mask = mask * rmask
        if mask is not None:
            if _no_mask_log is False:
                ret = ret + torch.log(mask.float())
            else:
                ret = ret * mask.float()
        if self.sm is not None:
            ret = self.sm(ret)
        return ret

    def _forward(self, x):
        assert (isinstance(self.computer, DynamicVecPreparer))
        vecs = self.saved_data
        rmask = None
        if len(vecs) == 2:
            vecs, rmask = vecs
        xshape = x.size()
        if len(xshape) == 2:
            x = x.unsqueeze(1)
        else:   # 3D -> unsqueeze rmask, will be repeated over seq dim of x
            if rmask is not None:
                rmask = rmask.unsqueeze(1)
        ret = torch.bmm(x, vecs.transpose(2, 1))
        if len(xshape) == 2:
            ret = ret.squeeze(1)
        return ret, rmask


class BFOL(DynamicWordLinout):
    def __init__(self, computer=None, worddic=None, ismD=None, inp_trans=None):
        super(BFOL, self).__init__(computer=computer, worddic=worddic)
        self.inppos2uwid = None
        self.inpenc = None
        self.ismD = ismD
        self.inp_trans = inp_trans

    def prepare(self, inpseq, inpenc, *xdata):
        super(BFOL, self).prepare(*xdata)
        inppos2uwid = q.var(torch.zeros(inpseq.size(0), inpseq.size(1), len(self.ismD))).cuda(inpseq).v
        inppos2uwid.data.scatter_(2, inpseq.unsqueeze(2).data, 1)
        inppos2uwid.data[:, :, 0] = 0
        # inppos2uwid = torch.log(inppos2uwid)+1
        self.inppos2uwid = inppos2uwid
        self.inpenc = inpenc

    def _forward(self, x):
        ret, rmask = super(BFOL, self)._forward(x)

        xshape = x.size()
        if len(xshape) == 2:
            x = x.unsqueeze(1)
        scores = torch.bmm(x, self.inpenc.transpose(2, 1))
        inppos2uwid = self.inppos2uwid[:, :scores.size(-1), :]      # in case input matrix shrank because of seq packing
        offset = (torch.min(scores) - 1000).data[0]
        umask = (inppos2uwid == 0).float()
        uwid_scores = scores.transpose(2, 1) * inppos2uwid
        uwid_scores = uwid_scores + offset * umask
        uwid_scores, _ = torch.max(uwid_scores, 1)
        uwid_scores_mask = (inppos2uwid.sum(1) > 0).float()        # (batsize, #uwid)
        sel_uwid_scores = uwid_scores.index_select(1, self.inp_trans)
        sel_uwid_scores_mask = uwid_scores_mask.index_select(1, self.inp_trans)
        # the zeros in seluwid mask for those uwids should already be there in rmask
        rret = ret * (1 - sel_uwid_scores_mask) + sel_uwid_scores_mask * sel_uwid_scores
        if len(xshape) == 2:
            rret = rret.squeeze(1)
        # assert(((rret != 0.).float() - rmask.float()).norm().cpu().data[0] == 0)
        return rret, rmask


def make_inp_emb(dim, ism, psm, useglove=True, gdim=None, gfrac=0.1):
    embdim = gdim if gdim is not None else dim
    baseemb = q.WordEmb(dim=embdim, worddic=psm.D)
    if useglove:
        baseemb = q.PartiallyPretrainedWordEmb(dim=embdim, worddic=psm.D, gradfracs=(1., gfrac))
    # if useglove:
    #     gloveemb = q.PretrainedWordEmb(dim=embdim, worddic=psm.D)
    #     baseemb = baseemb.override(gloveemb)

    class Computer(DynamicVecComputer):
        def __init__(self):
            super(Computer, self).__init__()
            self.baseemb = baseemb
            self.trans = torch.nn.Linear(embdim, dim, bias=False) if embdim != dim else None

        def forward(self, x, data):
            transids = torch.gather(data, 1, x)
            _pp = psm.pp(transids[:5].cpu().data.numpy())
            _embs, mask = self.baseemb(transids)
            if self.trans is not None:
                _embs = self.trans(_embs)
            return _embs

    emb = DynamicWordEmb(computer=Computer(), worddic=ism.D)
    return emb, baseemb


class ColnameEncoder(torch.nn.Module):
    def __init__(self, dim, colbaseemb, nocolid=None):
        super(ColnameEncoder, self).__init__()
        self.emb = colbaseemb
        self.embdim = colbaseemb.vecdim
        self.dim = dim
        self.enc = torch.nn.LSTM(self.embdim, self.dim, 1, batch_first=True)
        self.nocolid = nocolid

    def forward(self, x):
        rmask = None
        if self.nocolid is not None:
            rmask = x[:, :, 0] != self.nocolid
        xshape = x.size()
        flatx = x.contiguous().view(-1, x.size(-1))
        embx, mask = self.emb(flatx)
        c_0 = q.var(torch.zeros(1, flatx.size(0), self.dim)).cuda(x).v
        y_0 = c_0
        packedx, order = q.seq_pack(embx, mask)
        _y_t, (y_T, c_T) = self.enc(packedx, (y_0, c_0))
        y_T = y_T[0][order]
        y_t, umask = q.seq_unpack(_y_t, order)
        ret = y_T.contiguous().view(x.size(0), x.size(1), y_T.size(-1))
        return ret, rmask


class OutVecComputer(DynamicVecPreparer):
    def __init__(self, syn_emb, syn_trans, inpbaseemb, inp_trans,
                 colencoder, col_trans, worddic, colzero_to_inf=False):
        super(OutVecComputer, self).__init__()
        self.syn_emb = syn_emb
        self.syn_trans = syn_trans
        self.inp_emb = inpbaseemb
        self.inp_trans = inp_trans
        self.col_enc = colencoder
        self.col_trans = col_trans
        self.D = worddic
        self.colzero_to_inf = colzero_to_inf
        self.inpemb_trans = torch.nn.Linear(self.inp_emb.vecdim, self.syn_emb.vecdim, bias=False) \
            if self.inp_emb.vecdim != self.syn_emb.vecdim else None

    def prepare(self, inpmaps, colnames):
        x = q.var(torch.arange(0, len(self.D))).cuda(inpmaps).v.long()
        batsize = inpmaps.size(0)

        _syn_ids = self.syn_trans[x]
        _syn_embs, _syn_mask = self.syn_emb(_syn_ids.unsqueeze(0).repeat(batsize, 1))

        _inp_ids = self.inp_trans[x]
        transids = torch.gather(inpmaps, 1, _inp_ids.unsqueeze(0).repeat(batsize, 1))
        _inp_embs, _inp_mask = self.inp_emb(transids)
        if self.inpemb_trans is not None:
            _inp_embs = self.inpemb_trans(_inp_embs)

        _colencs, _col_mask = self.col_enc(colnames)
        _col_ids = self.col_trans[x]
        _col_ids = _col_ids.unsqueeze(0).repeat(batsize, 1)
        _col_trans_mask = (_col_ids > -1).long()
        _col_ids += (1 - _col_trans_mask)
        _col_mask = torch.gather(_col_mask, 1, _col_ids)
        _col_ids = _col_ids.unsqueeze(2).repeat(1, 1, _colencs.size(2))
        _col_embs = torch.gather(_colencs, 1, _col_ids)
        _col_mask = _col_mask.float() * _col_trans_mask.float()

        # _col_mask = _col_mask.float() - _inp_mask.float() - _syn_mask.float()

        _totalmask = _syn_mask.float() + _inp_mask.float() + _col_mask.float()

        assert (np.all(_totalmask.cpu().data.numpy() < 2))

        # _col_mask = (x > 0).float() - _inp_mask.float() - _syn_mask.float()

        ret =   _syn_embs * _syn_mask.float().unsqueeze(2) \
              + _inp_embs * _inp_mask.float().unsqueeze(2) \
              + _col_embs * _col_mask.float().unsqueeze(2)

        # _pp = osm.pp(x.cpu().data.numpy())
        return ret, _totalmask


def build_subdics(osm):
    # split dictionary for SQL syntax, col names and input tokens
    synD = {"<MASK>": 0}
    colD = {}
    inpD = {"<MASK>": 0}
    syn_trans = q.val(np.zeros((len(osm.D),), dtype="int64")).v
    inp_trans = q.val(np.zeros((len(osm.D),), dtype="int64")).v
    col_trans = q.val(np.zeros((len(osm.D),), dtype="int64")).v
    col_trans.data.fill_(-1)

    for k, v in osm.D.items():
        m = re.match('(UWID|COL)(\d+)', k)
        if m:
            if m.group(1) == "UWID":
                if k not in inpD:
                    inpD[k] = int(m.group(2))
                    inp_trans.data[v] = inpD[k]
            elif m.group(1) == "COL":
                if k not in colD:
                    colD[k] = int(m.group(2))
                    col_trans.data[v] = colD[k]
        else:
            if k not in synD:
                synD[k] = len(synD)
                syn_trans.data[v] = synD[k]
    return synD, inpD, colD, syn_trans, inp_trans, col_trans


def make_out_vec_computer(dim, osm, psm, csm, inpbaseemb=None, colbaseemb=None,
                          useglove=True, gdim=None, gfrac=0.1):
    # base embedder for input tokens        # TODO might want to think about reusing encoding
    embdim = gdim if gdim is not None else dim
    if inpbaseemb is None:
        inpbaseemb = q.WordEmb(dim=embdim, worddic=psm.D)
        if useglove:
            inpbaseemb = q.PartiallyPretrainedWordEmb(dim=embdim, worddic=psm.D, gradfracs=(1., gfrac))
            # gloveemb = q.PretrainedWordEmb(dim=embdim, worddic=psm.D)
            # inpbaseemb = inpbaseemb.override(gloveemb)

    # base embedder for column names
    if colbaseemb is None:
        colbaseemb = q.WordEmb(embdim, worddic=csm.D)
        if useglove:
            colbaseemb = q.PartiallyPretrainedWordEmb(dim=embdim, worddic=csm.D, gradfracs=(1., gfrac))
            # gloveemb = q.PretrainedWordEmb(embdim, worddic=csm.D)
            # colbaseemb = colbaseemb.override(gloveemb)

    synD, inpD, colD, syn_trans, inp_trans, col_trans = build_subdics(osm)

    syn_emb = q.WordEmb(dim, worddic=synD)

    colencoder = ColnameEncoder(dim, colbaseemb, nocolid=csm.D["nonecolumnnonecolumnnonecolumn"])
    computer = OutVecComputer(syn_emb, syn_trans, inpbaseemb, inp_trans, colencoder, col_trans, osm.D)
    return computer


def make_out_emb(dim, osm, psm, csm, inpbaseemb=None, colbaseemb=None, useglove=True, gdim=None, gfrac=0.1):
    comp = make_out_vec_computer(dim, osm, psm, csm, inpbaseemb=inpbaseemb, colbaseemb=colbaseemb,
                                 useglove=useglove, gdim=gdim, gfrac=gfrac)
    return DynamicWordEmb(computer=comp, worddic=osm.D)


def make_out_lin(dim, ism, osm, psm, csm, inpbaseemb=None, colbaseemb=None, useglove=True, gdim=None, gfrac=0.1):
    comp = make_out_vec_computer(dim, osm, psm, csm, inpbaseemb=inpbaseemb, colbaseemb=colbaseemb,
                                 useglove=useglove, gdim=gdim, gfrac=gfrac)
    inp_trans = comp.inp_trans      # to index
    out = BFOL(computer=comp, worddic=osm.D, ismD=ism.D, inp_trans=inp_trans)
    return out

# endregion


def make_oracle_df(tracker, mode=None):
    ttt = q.ticktock("oracle maker")
    print("oracle mode: {}".format(mode))
    oracle = q.DynamicOracleRunner(tracker=tracker,
                                   mode=mode,
                                   explore=0.)

    if _opt_test:
        print("TODO: oracle tests")
    return oracle


def run_seq2seq_tf(lr=0.001, batsize=100, epochs=100,
                   inpembdim=50, outembdim=50, innerdim=100, numlayers=1, dim=-1, gdim=-1,
                   dropout=0.2, rdropout=0.1, edropout=0., idropout=0., irdropout=0.,
                   wreg=0.00000000001, gradnorm=5., useglove=True, gfrac=0.01,
                   cuda=False, gpu=0, tag="none", test=False):
    settings = locals().copy()
    logger = q.Logger(prefix="wikisql_s2s_tf")
    logger.save_settings(**settings)
    logger.update_settings(completed=False)
    logger.update_settings(version="1.1")

    if gdim < 0:
        gdim = None

    if dim > 0:
        innerdim = dim
        inpembdim = dim // 2
        outembdim = inpembdim

    print("Seq2Seq + TF")
    if cuda:    torch.cuda.set_device(gpu)
    tt = q.ticktock("script")
    ism, osm, csm, psm, splits, e2cn = load_matrices()
    psm._matrix = psm.matrix * (psm.matrix != psm.D["<RARE>"])      # ASSUMES there are no real <RARE> words in psm

    # tracker = make_tracker(osm)     # TODO: only for bf

    devstart, teststart = splits
    eids = np.arange(0, len(ism), dtype="int64")

    outlindim = innerdim * 2        # (because we're using attention and cat-ing encoder and decoder)

    # TODO: might want to revert to overridden wordembs because gfrac with new wordemb seems to work worse
    inpemb, inpbaseemb = make_inp_emb(inpembdim, ism, psm, useglove=useglove, gdim=gdim, gfrac=gfrac)
    outemb = make_out_emb(outembdim, osm, psm, csm, inpbaseemb=inpbaseemb, useglove=useglove, gdim=gdim, gfrac=gfrac)
    outlin = make_out_lin(outlindim, ism, osm, psm, csm, useglove=useglove, gdim=gdim, gfrac=gfrac)

    # TODO: BEWARE OF VIEWS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # has to be compatible with outlin dim
    encoder = q.FastestLSTMEncoder(inpembdim, *([outlindim//2]*numlayers),
                                   dropout_in=idropout, dropout_rec=irdropout, bidir=True)   # TODO: dropouts ?! weight dropouts

    # TODO: below is for bf-based scripts, move
    # mlinout = make_struct_linout(outlindim, outlin, tracker.coreD, tracker.D, chained=True)

    class EncDec(torch.nn.Module):
        def __init__(self, dec):
            super(EncDec, self).__init__()
            self.inpemb = inpemb
            self.outemb = outemb
            self.outlin = outlin
            self.encoder = encoder
            self.decoder = dec

        def forward(self, inpseq, outseq, inpseqmaps, colnames):
            self.inpemb.prepare(inpseqmaps)

            _inpembs, _inpmask = self.inpemb(inpseq)
            _inpenc = self.encoder(_inpembs, mask=_inpmask)
            inpmask = _inpmask[:, :_inpenc.size(1)]
            inpenc = q.intercat(_inpenc.chunk(2, -1), -1)   # to blend fwd and rev dirs of bidir lstm
            ctx = inpenc.chunk(2, -1)[0]        # only half is used for attention addr and summary

            self.outemb.prepare(inpseqmaps, colnames)
            self.outlin.prepare(inpseq, inpenc, inpseqmaps, colnames)

            decoding = self.decoder(outseq,
                                    ctx=ctx,
                                    ctx_0=ctx[:, -1, :],        # TODO: ctx_0 not used if ctx2out is False (currently holds)
                                    ctxmask=inpmask)
            return decoding

    # region decoders and models
    # train decoder with teacher forcing, valid decoder with freerunning
    decdims = [outembdim] + [innerdim] * numlayers
    layers = [q.LSTMCell(decdims[i-1], decdims[i], dropout_in=dropout, dropout_rec=rdropout) for i in range(1, len(decdims))]
    decoder_top = q.AttentionContextDecoderTop(q.Attention().dot_gen(),
                                               q.Dropout(edropout),
                                               outlin, ctx2out=False)

    decoder_core = q.DecoderCore(outemb, *layers)
    decoder_cell = q.ModularDecoderCell(decoder_core, decoder_top)
    decoder_cell.set_runner(q.TeacherForcer())
    decoder = decoder_cell.to_decoder()

    m = EncDec(decoder)

    valid_decoder_cell = q.ModularDecoderCell(decoder_core, decoder_top)
    valid_decoder_cell.set_runner(q.FreeRunner())
    valid_decoder = valid_decoder_cell.to_decoder()

    valid_m = EncDec(valid_decoder)
    # endregion

    if test:
        devstart = 50
        teststart = 100

    # region data splits
    traindata = [ism.matrix[:devstart], osm.matrix[:devstart], psm.matrix[:devstart], e2cn[:devstart]]
    validdata = [ism.matrix[devstart:teststart], osm.matrix[devstart:teststart], psm.matrix[devstart:teststart], e2cn[devstart:teststart]]
    testdata = [ism.matrix[teststart:], osm.matrix[teststart:], psm.matrix[teststart:], e2cn[teststart:]]
    trainloader = q.dataload(*traindata, batch_size=batsize, shuffle=True)
    validloader = q.dataload(*validdata, batch_size=batsize, shuffle=False)
    testloader = q.dataload(*testdata, batch_size=batsize, shuffle=False)
    # endregion

    losses = q.lossarray(q.SeqCrossEntropyLoss(ignore_index=0),
                         q.SeqAccuracy(ignore_index=0),
                         #TreeAccuracy(ignore_index=0, treeparser=lambda x: SqlNode.parse_sql(osm.pp(x)))
                         )

    validlosses = q.lossarray(q.SeqAccuracy(ignore_index=0),
                              TreeAccuracy(ignore_index=0,
                                           treeparser=lambda x: order_adder_wikisql(SqlNode.parse_sql(osm.pp(x)))))

    logger.update_settings(optimizer="adam")
    optim = torch.optim.Adam(q.paramgroups_of(m), lr=lr)

    def inp_bt(a, b, c, colnameids):
        colnames = csm.matrix[colnameids.cpu().data.numpy()]
        colnames = q.var(colnames).cuda(colnameids).v
        return a, b[:, :-1], c, colnames, b[:, 1:]

    q.train(m).train_on(trainloader, losses)\
        .optimizer(optim)\
        .clip_grad_norm(gradnorm)\
        .set_batch_transformer(inp_bt)\
        .valid_with(valid_m).valid_on(validloader, validlosses)\
        .cuda(cuda)\
        .hook(logger)\
        .train(epochs)

    logger.update_settings(completed=True)

    # region final numbers
    finalvalidlosses = q.lossarray(q.SeqAccuracy(ignore_index=0),
                              TreeAccuracy(ignore_index=0,
                                           treeparser=lambda x: order_adder_wikisql(SqlNode.parse_sql(osm.pp(x)))))

    validresults = q.test(valid_m) \
        .on(validloader, finalvalidlosses) \
        .set_batch_transformer(inp_bt) \
        .cuda(cuda) \
        .run()

    print("DEV RESULTS:")
    print(validresults)

    testlosses = q.lossarray(q.SeqAccuracy(ignore_index=0),
                              TreeAccuracy(ignore_index=0,
                                           treeparser=lambda x: order_adder_wikisql(SqlNode.parse_sql(osm.pp(x)))))

    testresults = q.test(valid_m) \
        .on(testloader, testlosses) \
        .set_batch_transformer(inp_bt) \
        .cuda(cuda) \
        .run()

    logger.update_settings(valid_seq_acc=validresults[0], valid_tree_acc=validresults[1])
    logger.update_settings(test_seq_acc=testresults[0], test_tree_acc=testresults[1])

    print("TEST RESULTS:")
    print(testresults)
    # endregion


def run_seq2seq_oracle_df(lr=0.001, batsize=100, epochs=100,
                   inpembdim=50, outembdim=50, innerdim=100, numlayers=1, dim=-1, gdim=-1,
                   dropout=0.2, rdropout=0.1, edropout=0., idropout=0., irdropout=0.,
                   wreg=0.00000000001, gradnorm=5., useglove=True, gfrac=0.01,
                   cuda=False, gpu=0, tag="none", test=False, oraclemode="argmax-uniform"):
    settings = locals().copy()
    logger = q.Logger(prefix="wikisql_s2s_oracle_df")
    logger.save_settings(**settings)
    logger.update_settings(completed=False)
    logger.update_settings(version="0.9")

    if gdim < 0:
        gdim = None

    if dim > 0:
        innerdim = dim
        inpembdim = dim // 2
        outembdim = inpembdim

    print("Seq2Seq + ORACLE-DF")
    if cuda:    torch.cuda.set_device(gpu)
    tt = q.ticktock("script")
    ism, osm, csm, psm, splits, e2cn = load_matrices()

    psm._matrix = psm.matrix * (psm.matrix != psm.D["<RARE>"])      # ASSUMES there are no real <RARE> words in psm

    tracker = make_tracker_df(osm)     # TODO: only for bf
    oracle = make_oracle_df(tracker, mode=oraclemode)

    devstart, teststart = splits
    eids = np.arange(0, len(ism), dtype="int64")

    outlindim = innerdim * 2        # (because we're using attention and cat-ing encoder and decoder)

    # TODO: might want to revert to overridden wordembs because gfrac with new wordemb seems to work worse
    inpemb, inpbaseemb = make_inp_emb(inpembdim, ism, psm, useglove=useglove, gdim=gdim, gfrac=gfrac)
    outemb = make_out_emb(outembdim, osm, psm, csm, inpbaseemb=inpbaseemb, useglove=useglove, gdim=gdim, gfrac=gfrac)
    outlin = make_out_lin(outlindim, ism, osm, psm, csm, useglove=useglove, gdim=gdim, gfrac=gfrac)

    # TODO: BEWARE OF VIEWS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # has to be compatible with outlin dim
    encoder = q.FastestLSTMEncoder(inpembdim, *([outlindim//2]*numlayers),
                                   dropout_in=idropout, dropout_rec=irdropout, bidir=True)   # TODO: dropouts ?! weight dropouts

    class EncDec(torch.nn.Module):
        def __init__(self, dec, maxtime=None):
            super(EncDec, self).__init__()
            self.inpemb = inpemb
            self.outemb = outemb
            self.outlin = outlin
            self.encoder = encoder
            self.decoder = dec
            self.maxtime = maxtime

        def forward(self, inpseq, outseq_starts, inpseqmaps, colnames,
                    eids=None, maxtime=None):
            maxtime = self.maxtime if maxtime is None else maxtime
            self.inpemb.prepare(inpseqmaps)

            _inpembs, _inpmask = self.inpemb(inpseq)
            _inpenc = self.encoder(_inpembs, mask=_inpmask)
            inpmask = _inpmask[:, :_inpenc.size(1)]
            inpenc = q.intercat(_inpenc.chunk(2, -1), -1)   # to blend fwd and rev dirs of bidir lstm
            ctx = inpenc.chunk(2, -1)[0]        # only half is used for attention addr and summary

            self.outemb.prepare(inpseqmaps, colnames)
            self.outlin.prepare(inpseq, inpenc, inpseqmaps, colnames)

            decoding = self.decoder(outseq_starts,
                                    ctx=ctx,
                                    ctx_0=ctx[:, -1, :],        # TODO: ctx_0 not used if ctx2out is False (currently holds)
                                    ctxmask=inpmask,
                                    eids=eids,
                                    maxtime=maxtime)
            return decoding

    # region decoders and models
    # train decoder with teacher forcing, valid decoder with freerunning
    decdims = [outembdim] + [innerdim] * numlayers
    layers = [q.LSTMCell(decdims[i-1], decdims[i], dropout_in=dropout, dropout_rec=rdropout) for i in range(1, len(decdims))]
    decoder_top = q.AttentionContextDecoderTop(q.Attention().dot_gen(),
                                               q.Dropout(edropout),
                                               outlin, ctx2out=False)

    decoder_core = q.DecoderCore(outemb, *layers)
    decoder_cell = q.ModularDecoderCell(decoder_core, decoder_top)
    decoder_cell.set_runner(oracle)
    decoder = decoder_cell.to_decoder()

    m = EncDec(decoder)

    valid_decoder_cell = q.ModularDecoderCell(decoder_core, decoder_top)
    valid_decoder_cell.set_runner(q.FreeRunner())
    valid_decoder = valid_decoder_cell.to_decoder()

    valid_m = EncDec(valid_decoder, maxtime=osm.matrix.shape[1])      # TODO: check maxtime
    # endregion

    if test:
        print("BEWARE: TEST MODE MINI SPLITS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        devstart = 50
        teststart = 100

    # TODO: check wtf is going on downstairs
    # region DATA LOADING
    starts = osm.matrix[:, 0]
    alldata = [ism.matrix, starts, psm.matrix, e2cn, eids, osm.matrix, eids]
    traindata = [amat[:devstart] for amat in alldata]
    validdata = [amat[devstart:teststart] for amat in alldata]
    testdata =  [amat[teststart:] for amat in alldata]
    # traindata = [ism.matrix[:devstart], starts[:devstart], psm.matrix[:devstart], e2cn[:devstart], eids[:start]]
    # validdata = [ism.matrix[devstart:teststart], starts[devstart:teststart], psm.matrix[devstart:teststart], e2cn[devstart:teststart]]
    # testdata = [ism.matrix[teststart:], starts[teststart:], psm.matrix[teststart:], e2cn[teststart:]]
    trainloader = q.dataload(*[traindata[i] for i in [0, 1, 2, 3, 4, 6]], batch_size=batsize, shuffle=True)
    validloader = q.dataload(*[validdata[i] for i in [0, 1, 2, 3, 5]], batch_size=batsize, shuffle=False)
    testloader =  q.dataload(*[testdata[i]  for i in [0, 1, 2, 3, 5]], batch_size=batsize, shuffle=False)
    # endregion

    losses = q.lossarray(q.SeqNLLLoss(ignore_index=0),
                         q.SeqAccuracy(ignore_index=0),
                         #TreeAccuracy(ignore_index=0, treeparser=lambda x: SqlNode.parse_sql(osm.pp(x)))
                         )

    validlosses = q.lossarray(q.SeqAccuracy(ignore_index=0),
                              TreeAccuracy(ignore_index=0,
                                           treeparser=lambda x: SqlNode.parse_sql(osm.pp(x))))

    logger.update_settings(optimizer="adam")
    optim = torch.optim.Adam(q.paramgroups_of(m), lr=lr)

    def inp_bt(a, b, c, colnameids, d, e):      # e is gold, is eids
        colnames = csm.matrix[colnameids.cpu().data.numpy()]
        colnames = q.var(colnames).cuda(colnameids).v
        return a, b, c, colnames, d, e

    def valid_inp_bt(a, b, c, colnameids, d):   # d is gold, is sequences of ids
        colnames = csm.matrix[colnameids.cpu().data.numpy()]
        colnames = q.var(colnames).cuda(colnameids).v
        return a, b, c, colnames, d

    def out_btf(_out):
        return _out[:, :-1, :]      # TODO: why? --> because oracle terminates decoding
                                    # when all trackers terminate and the last output doesn't correspond to anything in goldacc

    def gold_btf(_eids):
        # q.embed()
        return torch.stack(oracle.goldacc, 1)

    def valid_gold_btf(x):
        return x[:, 1:]

    q.train(m).train_on(trainloader, losses)\
        .optimizer(optim)\
        .clip_grad_norm(gradnorm)\
        .set_batch_transformer(inp_bt, out_btf, gold_btf)\
        .valid_with(valid_m).valid_on(validloader, validlosses)\
        .set_valid_batch_transformer(valid_inp_bt, out_btf, valid_gold_btf) \
        .cuda(cuda)\
        .hook(logger)\
        .train(epochs)

    logger.update_settings(completed=True)

    # region final numbers
    finalvalidlosses = q.lossarray(q.SeqAccuracy(ignore_index=0),
                              TreeAccuracy(ignore_index=0,
                                           treeparser=lambda x: SqlNode.parse_sql(osm.pp(x))))

    validresults = q.test(valid_m) \
        .on(validloader, finalvalidlosses) \
        .set_batch_transformer(valid_inp_bt, out_btf, valid_gold_btf) \
        .cuda(cuda) \
        .run()

    print("DEV RESULTS:")
    print(validresults)

    testlosses = q.lossarray(q.SeqAccuracy(ignore_index=0),
                              TreeAccuracy(ignore_index=0,
                                           treeparser=lambda x: SqlNode.parse_sql(osm.pp(x))))

    testresults = q.test(valid_m) \
        .on(testloader, testlosses) \
        .set_batch_transformer(valid_inp_bt, out_btf, valid_gold_btf) \
        .cuda(cuda) \
        .run()

    logger.update_settings(valid_seq_acc=validresults[0], valid_tree_acc=validresults[1])
    logger.update_settings(test_seq_acc=testresults[0], test_tree_acc=testresults[1])

    print("TEST RESULTS:")
    print(testresults)
    # endregion





# TODO: same as df-tf but should be using BF-lin from trees and the struct linout
def run_seq2seq_tf_bf(lr=0.001, batsize=100, epochs=100,
                   inpembdim=50, outembdim=50, innerdim=100, numlayers=1, dim=-1, gdim=-1,
                   dropout=0.2, rdropout=0.1, edropout=0., idropout=0., irdropout=0.,
                   wreg=0.00000000001, gradnorm=5., useglove=True, gfrac=0.01,
                   cuda=False, gpu=0, tag="none"):
    settings = locals().copy()
    logger = q.Logger(prefix="wikisql_s2s_tf_bf")
    logger.save_settings(**settings)
    logger.update_settings(completed=False)
    logger.update_settings(version="0.9")

    if gdim < 0:
        gdim = None

    if dim > 0:
        innerdim = dim
        inpembdim = dim // 2
        outembdim = inpembdim

    print("Seq2Seq + TF")
    if cuda:    torch.cuda.set_device(gpu)
    tt = q.ticktock("script")
    ism, osm, csm, psm, splits, e2cn = load_matrices()
    psm._matrix = psm.matrix * (psm.matrix != psm.D["<RARE>"])      # ASSUMES there are no real <RARE> words in psm

    # tracker = make_tracker(osm)     # TODO: only for bf

    devstart, teststart = splits
    eids = np.arange(0, len(ism), dtype="int64")

    outlindim = innerdim * 2        # (because we're using attention and cat-ing encoder and decoder)

    # TODO: might want to revert to overridden wordembs because gfrac with new wordemb seems to work worse
    inpemb, inpbaseemb = make_inp_emb(inpembdim, ism, psm, useglove=useglove, gdim=gdim, gfrac=gfrac)
    outemb = make_out_emb(outembdim, osm, psm, csm, inpbaseemb=inpbaseemb, useglove=useglove, gdim=gdim, gfrac=gfrac)
    outlin = make_out_lin(outlindim, ism, osm, psm, csm, useglove=useglove, gdim=gdim, gfrac=gfrac)

    # TODO: BEWARE OF VIEWS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # has to be compatible with outlin dim
    encoder = q.FastestLSTMEncoder(inpembdim, *([outlindim//2]*numlayers),
                                   dropout_in=idropout, dropout_rec=irdropout, bidir=True)   # TODO: dropouts ?! weight dropouts

    # TODO: below is for bf-based scripts, move
    # mlinout = make_struct_linout(outlindim, outlin, tracker.coreD, tracker.D, chained=True)

    class EncDec(torch.nn.Module):
        def __init__(self, dec):
            super(EncDec, self).__init__()
            self.inpemb = inpemb
            self.outemb = outemb
            self.outlin = outlin
            self.encoder = encoder
            self.decoder = dec

        def forward(self, inpseq, outseq, inpseqmaps, colnames):
            self.inpemb.prepare(inpseqmaps)

            _inpembs, _inpmask = self.inpemb(inpseq)
            _inpenc = self.encoder(_inpembs, mask=_inpmask)
            inpmask = _inpmask[:, :_inpenc.size(1)]
            inpenc = q.intercat(_inpenc.chunk(2, -1), -1)   # to blend fwd and rev dirs of bidir lstm
            ctx = inpenc.chunk(2, -1)[0]        # only half is used for attention addr and summary

            self.outemb.prepare(inpseqmaps, colnames)
            self.outlin.prepare(inpseq, inpenc, inpseqmaps, colnames)

            decoding = self.decoder(outseq,
                                    ctx=ctx,
                                    ctx_0=ctx[:, -1, :],        # TODO: ctx_0 not used if ctx2out is False (currently holds)
                                    ctxmask=inpmask)
            return decoding

    # region decoders and models
    # train decoder with teacher forcing, valid decoder with freerunning
    decdims = [outembdim] + [innerdim] * numlayers
    layers = [q.LSTMCell(decdims[i-1], decdims[i], dropout_in=dropout, dropout_rec=rdropout) for i in range(1, len(decdims))]
    decoder_top = q.AttentionContextDecoderTop(q.Attention().dot_gen(),
                                               q.Dropout(edropout),
                                               outlin, ctx2out=False)

    decoder_core = q.DecoderCore(outemb, *layers)
    decoder_cell = q.ModularDecoderCell(decoder_core, decoder_top)
    decoder_cell.set_runner(q.TeacherForcer())
    decoder = decoder_cell.to_decoder()

    m = EncDec(decoder)

    valid_decoder_cell = q.ModularDecoderCell(decoder_core, decoder_top)
    valid_decoder_cell.set_runner(q.FreeRunner())
    valid_decoder = valid_decoder_cell.to_decoder()

    valid_m = EncDec(valid_decoder)
    # endregion

    # devstart = 50
    # teststart = 100

    # region data splits
    traindata = [ism.matrix[:devstart], osm.matrix[:devstart], psm.matrix[:devstart], e2cn[:devstart]]
    validdata = [ism.matrix[devstart:teststart], osm.matrix[devstart:teststart], psm.matrix[devstart:teststart], e2cn[devstart:teststart]]
    testdata = [ism.matrix[teststart:], osm.matrix[teststart:], psm.matrix[teststart:], e2cn[teststart:]]
    trainloader = q.dataload(*traindata, batch_size=batsize, shuffle=True)
    validloader = q.dataload(*validdata, batch_size=batsize, shuffle=False)
    testloader = q.dataload(*testdata, batch_size=batsize, shuffle=False)
    # endregion

    losses = q.lossarray(q.SeqNLLLoss(ignore_index=0),
                         q.SeqAccuracy(ignore_index=0),
                         #TreeAccuracy(ignore_index=0, treeparser=lambda x: SqlNode.parse_sql(osm.pp(x)))
                         )

    validlosses = q.lossarray(q.SeqAccuracy(ignore_index=0),
                              TreeAccuracy(ignore_index=0, treeparser=lambda x: SqlNode.parse_sql(osm.pp(x))))

    logger.update_settings(optimizer="adam")
    optim = torch.optim.Adam(q.paramgroups_of(m), lr=lr)

    def inp_bt(a, b, c, colnameids):
        colnames = csm.matrix[colnameids.cpu().data.numpy()]
        colnames = q.var(colnames).cuda(colnameids).v
        return a, b[:, :-1], c, colnames, b[:, 1:]

    q.train(m).train_on(trainloader, losses)\
        .optimizer(optim)\
        .clip_grad_norm(gradnorm)\
        .set_batch_transformer(inp_bt)\
        .valid_with(valid_m).valid_on(validloader, validlosses)\
        .cuda(cuda)\
        .hook(logger)\
        .train(epochs)

    logger.update_settings(completed=True)

    # region final numbers
    finalvalidlosses = q.lossarray(q.SeqAccuracy(ignore_index=0),
                              TreeAccuracy(ignore_index=0, treeparser=lambda x: SqlNode.parse_sql(osm.pp(x))))

    validresults = q.test(valid_m) \
        .on(validloader, finalvalidlosses) \
        .set_batch_transformer(inp_bt) \
        .cuda(cuda) \
        .run()

    print("DEV RESULTS:")
    print(validresults)

    testlosses = q.lossarray(q.SeqAccuracy(ignore_index=0),
                              TreeAccuracy(ignore_index=0, treeparser=lambda x: SqlNode.parse_sql(osm.pp(x))))

    testresults = q.test(valid_m) \
        .on(testloader, testlosses) \
        .set_batch_transformer(inp_bt) \
        .cuda(cuda) \
        .run()

    logger.update_settings(valid_seq_acc=validresults[0], valid_tree_acc=validresults[1])
    logger.update_settings(test_seq_acc=testresults[0], test_tree_acc=testresults[1])

    print("TEST RESULTS:")
    print(testresults)
    # endregion




def run_seq2seq_oracle():       # TODO: uses BF-lin and struct linout like previous but uses oracle instead of teacherforcer
    pass        # TODO


def test_df_lins(tree):
    tracker = NodeTrackerDF(tree)
    accs = set()
    for i in range(1000):
        acc = u""
        tracker.reset()
        while True:
            vnt = tracker._nvt
            if len(vnt) == 0:
                break
            sel = random.choice(list(vnt))
            acc += u" " + sel
            tracker.nxt(sel)
        accs.add(acc)
        pacc = SqlNode.parse_df((unidecode(acc)))
        assert(pacc.equals(tree))
    print("number of unique linearizations for toy example: ", len(accs))
        # print(pacc.pptree())



if __name__ == "__main__":
    # q.argprun(prepare_data)
    # create_mats()
    # q.argprun(load_matrices)
    # q.argprun(run_seq2seq_tf)
    q.argprun(run_seq2seq_oracle_df)
    # tree = SqlNode.parse_sql("<QUERY> <SELECT> AGG0 COL5 <WHERE> <COND> COL3 OP0 <VAL> UWID4 UWID5 <ENDVAL> <COND> COL1 OP1 <VAL> UWID1 UWID2 UWID3 <ENDVAL> <END> <select> <END>")
    # test_df_lins(tree)
    # print(tree.pptree())
    # treestr = tree.pp()
    # treestr = treestr.replace("*NC", "")
    # print(treestr)
    # retree = SqlNode.parse(treestr)
    # print(retree)
