# -*- coding: utf-8 -*-
import codecs
import json
import os
import pickle as pkl
import re
from collections import OrderedDict, Counter

import nltk
import numpy as np
import requests
import torch

import qelos as q
from qelos.scripts.treesupbf.trees import Node, NodeTrackerDF
from qelos.scripts.treesupbf.pasdecode import TreeAccuracy
import random
from unidecode import unidecode
from qelos.train import BestSaver
from tqdm import tqdm


_opt_test = True


# region DATA
# region GENERATING .LINES
def read_jsonl(p):
    """ takes a path and returns objects from json lines """
    lines = []
    with open(p) as f:
        for line in f:
            example = json.loads(line)
            lines.append(example)
    return lines


def jsonls_to_lines(p="../../../datasets/wikisql_clean/"):
    """ loads all jsons, converts them to .lines, saves and returns """
    # region load all jsons
    tt = q.ticktock("data preparer")
    tt.tick("loading jsons")

    # load examples
    traindata = read_jsonl(p+"train.jsonl")
    print("{} training questions".format(len(traindata)))
    traindb = read_jsonl(p+"train.tables.jsonl")
    print("{} training tables".format(len(traindb)))
    devdata = read_jsonl(p + "dev.jsonl")

    # load tables
    print("{} dev questions".format(len(devdata)))
    devdb = read_jsonl(p+"dev.tables.jsonl")
    print("{} dev tables".format(len(devdb)))
    testdata = read_jsonl(p + "test.jsonl")
    print("{} test questions".format(len(testdata)))
    testdb = read_jsonl(p + "test.tables.jsonl")
    print("{} test tables".format(len(testdb)))

    # join all tables in one
    alltables = {}
    for table in traindb + devdb + testdb:
        alltables[table["id"]] = table

    print("total number of tables: {} ".format(len(alltables)))
    tt.tock("jsons loaded")
    # endregion

    # region generating examples
    tt.tick("generating examples")
    tt.msg("train examples")
    trainexamples = []
    for line in traindata:
        try:
            trainexamples.append(jsonl_to_line(line, alltables))
        except Exception as e:
            print("FAILED: {}".format(line))

    # IMPORTANT: try not to omit any examples in dev and test
    tt.msg("dev examples")
    devexamples = []
    for line in devdata:
        try:
            devexamples.append(jsonl_to_line(line, alltables))
        except Exception as e:
            print("FAILED: {}".format(line))

    tt.msg("test examples")
    testexamples = []
    for line in testdata:
        try:
            testexamples.append(jsonl_to_line(line, alltables))
        except Exception as e:
            print("FAILED: {}".format(line))

    tt.tock("examples generated")
    # endregion

    # region save lines
    tt.msg("saving lines")
    print("\n".join(trainexamples[:10]))

    with codecs.open(p + "train.lines", "w", encoding="utf-8") as f:
        for example in trainexamples:
            f.write(u"{}\n".format(example))
    with codecs.open(p + "dev.lines", "w", encoding="utf-8") as f:
        for example in devexamples:
            f.write(u"{}\n".format(example))
    with codecs.open(p + "test.lines", "w", encoding="utf-8") as f:
        for example in testexamples:
            f.write(u"{}\n".format(example))
    # endregion

    return trainexamples, devexamples, testexamples


def jsonl_to_line(line, alltables):
    """ takes object from .jsonl, creates line"""
    column_names = alltables[line["table_id"]]["header"]
    question = u" ".join(nltk.word_tokenize(line["question"])).lower()

    # region replacements on top of tokenization:
    question = question.replace(u"\u2009", u" ")    # unicode long space to normal space
    question = question.replace(u"\u2013", u"-")    # unicode long dash to normal dash
    question = question.replace(u"`", u"'")         # unicode backward apostrophe to normal
    question = question.replace(u"\u00b4", u"'")    # unicode forward apostrophe to normal
    question = question.replace(u"''", u'"')    # double apostrophe to quote
    # add spaces around some special characters because nltk tokenizer doesn't:
    question = question.replace(u'\u20ac', u' \u20ac ')     # euro sign
    question = question.replace(u'\uffe5', u' \uffe5 ')     # yen sign
    question = question.replace(u'\u00a3', u' \u00a3 ')     # pound sign
    question = question.replace(u"'", u" ' ")
    question = re.sub(u'/$', u' /', question)
    # split up meters and kilometers because nltk tokenizer doesn't
    question = re.sub(u'(\d+[,\.]\d+)(k?m)', u'\g<1> \g<2>', question)

    question = re.sub(u'\s+', u' ', question)
    # endregion

    # MANUAL FIXES FOR TYPOS OR FAILED TOKENIZATION (done in original jsonl's) - train fixes not included:
    # dev, line 6277, "25,000." to "25,000 ,"
    # dev, line 7784, "No- Gold Coast" to "No Gold Coast"
    # test, line 6910, "Difference of- 17" to "Difference of - 17" (added space)
    # test, line 2338, "baccalaureate colleges" to "baccalaureate college" (condition value contains latter)
    # test, line 5440, "44+" to "44 +"
    # test, line 7159, "a frequency of 1600MHz and voltage under 1.35V" to "a frequency of 1600 MHz and voltage under 1.35 V" (added two spaces)
    # test, line 8042, "under 20.6bil" to "under 20.6 bil" (added space)
    # test, line 13290, "35.666sqmi" to "35.666 sqmi" (added space)
    # BAD TEST CHANGES (left in to ensure consistency of line numbers)
    # test, line 6077, changed first condition from "\u2013" to "no"

    # region construct query
    # select clause
    sql_select = u"AGG{} COL{}".format(line["sql"]["agg"], line["sql"]["sel"])
    # where clause
    sql_wheres = []
    for cond in line["sql"]["conds"]:
        if isinstance(cond[2], float):
            condval = unicode(cond[2].__repr__())       # printing float in original precision
        else:
            condval = unicode(cond[2]).lower()

        # replacements in condval, equivalent to question replacements
        condval = condval.replace(u"\u2009", u" ")    # unicode long space to normal space
        condval = condval.replace(u"\u2013", u"-")    # unicode long dash to normal dash
        condval = condval.replace(u"`", u"'")
        condval = condval.replace(u"\u00b4", u"'")    # unicode forward apostrophe to normal
        condval = condval.replace(u"''", u'"')
        _condval = condval.replace(u" ", u"")

        # region rephrase condval in terms of span of question
        condval = None
        questionsplit = question.split()

        for i, qword in enumerate(questionsplit):
            for qwordd in _condval:
                found = False
                for j in range(i+1, len(questionsplit) + 1):
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
        # endregion

        condl = u"<COND> COL{} OP{} <VAL> {} <ENDVAL>".format(cond[0], cond[1], condval)
        sql_wheres.append(condl)

    # create full query:
    if len(sql_wheres) > 0:
        sql = u"<QUERY> <SELECT> {} <WHERE> {}".format(sql_select, u" ".join(sql_wheres))
    else:
        sql = u"<QUERY> <SELECT> {}".format(sql_select)
    ret = u"{}\t{}\t{}".format(question, sql, u"\t".join(column_names))

    return ret
# endregion


# region GENERATING .MATS
def load_lines(p):
    retlines = []
    with codecs.open(p, encoding="utf-8") as f:
        for line in f:
            linesplits = line.strip().split("\t")
            assert(len(linesplits) > 2)
            retlines.append((linesplits[0], linesplits[1], linesplits[2:]))
    return retlines


def create_mats(p="../../../datasets/wikisql_clean/"):
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

    gwids = np.ones((ism.matrix.shape[0], numberunique + 3),
                     dtype="int64")  # per-example dictionary, mapping UWID position to GWID
                                     # , mapping unused UWIDs to <RARE> GWID

    # ism.D contains dictionary over all question words
    gwids = gwids * ism.D["<RARE>"]  # set default to gwid <RARE>
    gwids[:, 0] = ism.D["<MASK>"]  # set UWID0 to <MASK> for all examples

    # pedics matrix is used as follows:
    # suppose question has "UWID1 UWID2", then map to ints ([1, 2]),
    # select [gwids[example_id, 1], gwids[example_id, 2]] to get the actual words

    uwids = np.zeros_like(ism.matrix)                         # questions in terms of uwids

    rD = {v: k for k, v in ism.D.items()}                     # gwid reverse dictionary

    gwid2uwid_dics = []           # list of dictionaries mapping gwids to uwids for every example

    for i in range(len(ism.matrix)):    # for every example
        row = ism.matrix[i]             # get row
        gwid2uwid = {"<MASK>": 0}      # initialize gwid2uwid dictionary (for this example)
        for j in range(len(row)):       # for position in row
            k = row[j]                  # get gwid for word at that position
            if rD[k] not in gwid2uwid:                  # word for that gwid is not in gwid2uwid
                gwid2uwid[rD[k]] = len(gwid2uwid)       # add uwid for the word for that gwid to gwid2wid
                gwids[i, gwid2uwid[rD[k]]] = k          # add mapping from new uwid to gwid
            uwids[i, j] = gwid2uwid[rD[k]]              # put uwid in uwid mat
        gwid2uwid_dics.append(gwid2uwid)                # add gwid2uwid dic to list

    # create dictionary from uwid words to ids
    uwidD = dict(
        zip(["<MASK>"] + ["UWID{}".format(i + 1) for i in range(gwids.shape[1] - 1)],
            range(gwids.shape[1])))

    # region target sequences matrix
    osm = q.StringMatrix(indicate_start=True, indicate_end=True)
    osm.tokenize = lambda x: x.split()
    i = 0
    for _, query, columns in trainlines + devlines + testlines:
        query_tokens = query.split()
        gwid2uwid = gwid2uwid_dics[i]
        query_tokens = ["UWID{}".format(gwid2uwid[e])       # map target words to UWIDs where possible
                        if e in gwid2uwid else e
                        for e in query_tokens]
        _q = " ".join(query_tokens)
        osm.add(_q)
        i += 1
    osm.finalize()
    # endregion

    # region column names
    example2columnnames = np.zeros((len(osm), 44), dtype="int64")
                # each row contains sequence of column name ids available for that example
    uniquecolnames = OrderedDict({"nonecolumnnonecolumnnonecolumn": 0})     # unique column names
    i = 0
    for _, query, columns in trainlines + devlines + testlines:
        j = 0
        for column in columns:
            if column not in uniquecolnames:
                uniquecolnames[column] = len(uniquecolnames)
            example2columnnames[i, j] = uniquecolnames[column]
            j += 1
        i += 1

    # create matrix with for every unique column name (row), the sequence of words describing it
    csm = q.StringMatrix(indicate_start=False, indicate_end=False)
    for i, columnname in enumerate(uniquecolnames.keys()):
        if columnname == u'№':
            columnname = u'number'
        csm.add(columnname)
    csm.finalize()
    # idx 3986 is zero because it's u'№' and unidecode makes it empty string, has 30+ occurrences
    assert(len(np.argwhere(csm.matrix[:, 0] == 0)) == 0)
    # endregion

    # region save
    with open(p + "matcache.mats", "w") as f:
        np.savez(f, ism=uwids, osm=osm.matrix, csm=csm.matrix, pedics=gwids, e2cn=example2columnnames)
    with open(p + "matcache.dics", "w") as f:
        dics = {"ism": uwidD, "osm": osm.D, "csm": csm.D, "pedics": ism.D, "sizes": (devstart, teststart)}
        pkl.dump(dics, f, protocol=pkl.HIGHEST_PROTOCOL)

    # print("question dic size: {}".format(len(ism.D)))
    # print("question matrix size: {}".format(ism.matrix.shape))
    # print("query dic size: {}".format(len(osm.D)))
    # print("query matrix size: {}".format(osm.matrix.shape))
    tt.tock("matrices prepared")
    # endregion


def load_matrices(p="../../../datasets/wikisql_clean/"):
    """ loads matrices generated before.
        Returns:    * ism: input questions - in uwids
                    * osm: target sequences - use uwids for words
                    * csm: column names for unique column names
                    * gwids: for every uwid in ism/osm, mapping to gwids by position
                    * splits: indexes where train ends and dev ends
                    * e2cn: example ids to column names mapping (matrix)

        """
    tt = q.ticktock("matrix loader")
    tt.tick("loading matrices")
    with open(p+"matcache.mats") as f:
        mats = np.load(f)
        ismmat, osmmat, csmmat, gwids, e2cn \
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
    gwids = q.StringMatrix()
    gwids.set_dictionary(pedicsD)
    gwids._matrix = gwids
    # q.embed()
    return ism, osm, csm, gwids, splits, e2cn
# endregion


def ppq(i, ism, gwids):
    return gwids.pp(gwids.matrix[i][ism.matrix[i]])

# endregion

# region SQL TREES
# region Node and Order
class SqlNode(Node):
    mode = "limited"
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
    def parse_sql(cls, inp, _rec_arg=None, _toprec=True, _ret_remainder=False):
        """ ONLY FOR ORIGINAL LIN
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
                head, islast, isleaf = headsplits[
                                           0], SqlNode.last_suffix in headsplits, SqlNode.leaf_suffix in headsplits

            if head == "<QUERY>":
                assert (isleaf is None or isleaf is False)
                assert (islast is None or islast is True)
                children, tail = cls.parse_sql(tail, _rec_arg=head, _toprec=False)
                ret = SqlNode(head, children=children)
                break
            elif head == "<END>":
                ret = siblings, []
                break
            elif head in jumpers:
                assert (isleaf is None or isleaf is False)
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
                assert (isleaf is None or isleaf is True)
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
            if cls.mode == "limited":
                order_adder_wikisql_limited(ret)
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
                    token, suffix = splits[0], "*" + splits[1]
                if token not in "<QUERY> <SELECT> <WHERE> <COND> <VAL>".split():
                    suffix = "*NC" + suffix
                tokens[i] = token + suffix
            ret = super(SqlNode, cls).parse(" ".join(tokens), _rec_arg=None, _toprec=True)
            if cls.mode == "limited":
                order_adder_wikisql_limited(ret)
            else:
                order_adder_wikisql(ret)
            return ret
        else:
            return super(SqlNode, cls).parse(tokens, _rec_arg=_rec_arg, _toprec=_toprec)

    @classmethod
    def parse_df(cls, inp, _toprec=True):
        if _toprec:
            parse = super(SqlNode, cls).parse_df(inp, _toprec=_toprec)
            if cls.mode == "limited":
                order_adder_wikisql_limited(parse)
            else:
                order_adder_wikisql(parse)
            return parse
        else:
            return super(SqlNode, cls).parse_df(inp, _toprec)


def order_adder_wikisql(parse):
    # add order to children of VAL
    def order_adder_rec(y):
        for i, ychild in enumerate(y.children):
            if y.name == "<VAL>":
                ychild.order = i
            order_adder_rec(ychild)

    order_adder_rec(parse)
    return parse


def order_adder_wikisql_limited(parse):
    # add order everywhere except children of WHERE
    def order_adder_rec(y):
        for i, ychild in enumerate(y.children):
            if y.name != "<WHERE>":
                ychild.order = i
            order_adder_rec(ychild)

    order_adder_rec(parse)
    return parse


# region test
def test_sqlnode():
    # TODO: test parsing from different linearizations
    # TODO: test __eq__
    # TODO: test order while parsing and __eq__
    pass
# endregion

# endregion

# region Trackers
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
                    suffix += u''
                    print("sum ting wong in sql tracker df !!!!!!!!!!!")
            x = core + suffix
            tracker.nxt(x)

    def is_terminated(self, eid):
        return self.trackers[eid].is_terminated() and self._did_the_end[eid] is True


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
        for k in range(5000, 5100):  # TODO: 5034 throws error (nvts wrong) <-- should not affect tf script
            accs = set()
            for j in range(100):
                acc = u""
                tracker.reset()
                if k == 5033:
                    pass
                while True:
                    if tracker.is_terminated(k):
                        break
                    vnt = tracker.get_valid_next(k)
                    sel = random.choice(vnt)
                    acc += u" " + tracker.rD[sel]
                    tracker.update(k, sel)
                accs.add(acc)
                if not SqlNode.parse_sql(unidecode(acc)).equals(tracker.trackables[k]):
                    print(acc)
                    print(tracker.trackables[k].pptree())
                    print(SqlNode.parse_sql(unidecode(acc)).pptree())
                    raise q.SumTingWongException("trees not equal")
            print("number of unique linearizations for example {}: {}".format(k, len(accs)))
    return tracker

# region test
def test_grouptracker():
    # TODO: define what to test
    pass
# endregion

# endregion

# endregion SQL TREES

# region DYNAMIC VECTORS
# region intro
from qelos.word import WordEmbBase, WordLinoutBase

class DynamicVecComputer(torch.nn.Module):  pass
class DynamicVecPreparer(torch.nn.Module):  pass
# endregion

# region dynamic word emb and word linout in general
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


class DynamicWordLinout(WordLinoutBase):        # removed the logsoftmax in here
    def __init__(self, computer=None, worddic=None):
        super(DynamicWordLinout, self).__init__(worddic)
        self.computer = computer
        maskid = worddic[self.masktoken] if self.masktoken in worddic else None
        self.maskid = maskid
        self.outdim = max(worddic.values()) + 1
        self.saved_data = None

    def prepare(self, *xdata):
        assert (isinstance(self.computer, DynamicVecPreparer))
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
        else:  # 3D -> unsqueeze rmask, will be repeated over seq dim of x
            if rmask is not None:
                rmask = rmask.unsqueeze(1)
        ret = torch.bmm(x, vecs.transpose(2, 1))
        if len(xshape) == 2:
            ret = ret.squeeze(1)
        return ret, rmask
# endregion

# region dynamic vectors modules
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
        if self.inp_emb.vecdim != self.syn_emb.vecdim:
            print("USING LIN ADAPTER in OUT")
            self.inpemb_trans = torch.nn.Linear(self.inp_emb.vecdim, self.syn_emb.vecdim, bias=False)
        else:
            self.inpemb_trans = None

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


class BFOL(DynamicWordLinout):
    def __init__(self, computer=None, worddic=None, ismD=None, inp_trans=None, nocopy=False):
        super(BFOL, self).__init__(computer=computer, worddic=worddic)
        self.inppos2uwid = None
        self.inpenc = None
        self.ismD = ismD
        self.inp_trans = inp_trans
        self.nocopy = nocopy

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
        if self.nocopy is True:
            return ret, rmask

        xshape = x.size()
        if len(xshape) == 2:
            x = x.unsqueeze(1)
        # x will be twice the size of inpenc -> which part of x to take???
        compx = x[:, :, :x.size(2) // 2]  # slice out first half, which is y_t (not ctx_t)
        scores = torch.bmm(compx, self.inpenc.transpose(2, 1))

        inppos2uwid = self.inppos2uwid[:, :scores.size(-1), :]  # in case input matrix shrank because of seq packing
        offset = (torch.min(scores) - 1000).data[0]
        umask = (inppos2uwid == 0).float()
        uwid_scores = scores.transpose(2, 1) * inppos2uwid
        uwid_scores = uwid_scores + offset * umask
        uwid_scores, _ = torch.max(uwid_scores, 1)
        uwid_scores_mask = (inppos2uwid.sum(1) > 0).float()  # (batsize, #uwid)
        sel_uwid_scores = uwid_scores.index_select(1, self.inp_trans)
        sel_uwid_scores_mask = uwid_scores_mask.index_select(1, self.inp_trans)
        # the zeros in seluwid mask for those uwids should already be there in rmask
        rret = ret * (1 - sel_uwid_scores_mask) + sel_uwid_scores_mask * sel_uwid_scores
        if len(xshape) == 2:
            rret = rret.squeeze(1)
        # assert(((rret != 0.).float() - rmask.float()).norm().cpu().data[0] == 0)
        return rret, rmask
# endregion

# region dynamic vector module creation functions
# region dynamic vector module creation helper functions
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


def make_out_vec_computer(dim, osm, psm, csm, inpbaseemb=None, colbaseemb=None, colenc=None,
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

    if colenc is None:
        colenc = ColnameEncoder(dim, colbaseemb, nocolid=csm.D["nonecolumnnonecolumnnonecolumn"])

    computer = OutVecComputer(syn_emb, syn_trans, inpbaseemb, inp_trans, colenc, col_trans, osm.D)
    return computer, inpbaseemb, colbaseemb, colenc
# endregion


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
            if embdim != dim:
                print("USING LIN ADAPTER")
                self.trans = torch.nn.Linear(embdim, dim, bias=False)
            else:
                self.trans = None

        def forward(self, x, data):
            transids = torch.gather(data, 1, x)
            _pp = psm.pp(transids[:5].cpu().data.numpy())
            _embs, mask = self.baseemb(transids)
            if self.trans is not None:
                _embs = self.trans(_embs)
            return _embs

    emb = DynamicWordEmb(computer=Computer(), worddic=ism.D)
    return emb, baseemb


def make_out_emb(dim, osm, psm, csm, inpbaseemb=None, colbaseemb=None,
                 useglove=True, gdim=None, gfrac=0.1, colenc=None):
    print("MAKING OUT EMB")
    comp, inpbaseemb, colbaseemb, colenc \
        = make_out_vec_computer(dim, osm, psm, csm, inpbaseemb=inpbaseemb, colbaseemb=colbaseemb,
                                colenc=colenc, useglove=useglove, gdim=gdim, gfrac=gfrac)
    return DynamicWordEmb(computer=comp, worddic=osm.D), inpbaseemb, colbaseemb, colenc


def make_out_lin(dim, ism, osm, psm, csm, inpbaseemb=None, colbaseemb=None,
                 useglove=True, gdim=None, gfrac=0.1, colenc=None, nocopy=False):
    print("MAKING OUT LIN")
    comp, inpbaseemb, colbaseemb, colenc \
        = make_out_vec_computer(dim, osm, psm, csm, inpbaseemb=inpbaseemb, colbaseemb=colbaseemb,
                                colenc=colenc, useglove=useglove, gdim=gdim, gfrac=gfrac)
    inp_trans = comp.inp_trans  # to index
    out = BFOL(computer=comp, worddic=osm.D, ismD=ism.D, inp_trans=inp_trans, nocopy=nocopy)
    return out, inpbaseemb, colbaseemb, colenc

# endregion


# endregion


# region MAIN SCRIPTS
# region main scripts helper functions
def make_oracle_df(tracker, mode=None):
    ttt = q.ticktock("oracle maker")
    print("oracle mode: {}".format(mode))
    oracle = q.DynamicOracleRunner(tracker=tracker,
                                   mode=mode,
                                   explore=0.)

    if _opt_test:
        print("TODO: oracle tests")
    return oracle
# endregion

# region main scripts

# endregion


# region eval helper functions

# endregion
# endregion


# region TEST
def test_data(x=0):
    # jsonls_to_lines()
    create_mats()

# endregion


if __name__ == "__main__":
    q.argprun(test_data)