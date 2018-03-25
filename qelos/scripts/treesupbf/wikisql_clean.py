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


# TODO: !!!!!!!!!!!!! DON'T FORGET THAT VALID BT'S OVERRIDE NORMAL BT'S --> HAVE TO SET TO NONE TO CANCEL TRAIN TIME BT
#       --> TODO: CHECK IN OLD SCRIPT AND NEW SCRIPT THAT NOTHING WRONG HAPPENS !!!!!!!!!

# REMARK:   changed set_(valid)_batch_transform() to accept None's as they are
#           before, might have "sticky" bt's, but shouldn't have affected old wikisql TODO: CHECK !!!

# TODO: make sure test and dev splits are correct
# TODO: MAKE SURE vanilla embeddings are changed after training

# TODO: rare words in col names
# DONE: UNIQUE RARES  --> make_inp_emb and OutVecComputer use rare-X embeddings based on rare_gwids_not_in_glove
#                         (implemented by a hack replacement with rare_vec)


_opt_test = True
DATA_PATH = "../../../datasets/wikisql_clean/"


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


def jsonls_to_lines(p=DATA_PATH):
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
    # question = question.replace(u"\u2009", u" ")    # unicode long space to normal space
    # question = question.replace(u"\u2013", u"-")    # unicode long dash to normal dash
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
    # test, line 7159, "a frequency of 1600MHz and voltage under 1.35V" to "a frequency of 1600 MHz and voltage under 1.35 V" (added two spaces) and changed condition "1600mhz" to "1600 mhz"
    # test, line 8042, "under 20.6bil" to "under 20.6 bil" (added space)
    # test, line 8863, replaced long spaces "\u2009" with normal space
    # test, line 8866, replaced long spaces "\u2009" with normal space
    # test, line 8867, replaced long spaces "\u2009" with normal space
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
        # condval = condval.replace(u"\u2009", u" ")    # unicode long space to normal space
        # condval = condval.replace(u"\u2013", u"-")    # unicode long dash to normal dash
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


def create_mats(p=DATA_PATH):
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


def load_matrices(p=DATA_PATH):
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
        ismmat, osmmat, csmmat, gwidsmat, e2cn \
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
    gwids._matrix = gwidsmat
    # q.embed()
    return ism, osm, csm, gwids, splits, e2cn


def reconstruct_question(uwids, gwids, rgd):
    words = gwids[uwids]
    question = " ".join([rgd[wordid] for wordid in words])
    question = question.replace("<MASK>", " ")
    question = re.sub("\s+", " ", question)
    question = question.strip()
    return question


def reconstruct_query(osmrow, gwidrow, rod, rgd):
    query = u" ".join([rod[wordid] for wordid in osmrow])
    query = query.replace(u"<MASK>", u" ")
    query = re.sub(u"\s+", u" ", query)
    query = query.strip()
    query = re.sub(u"UWID\d+", lambda x: rgd[gwidrow[int(x.group(0)[4:])]], query)
    return query


# region test
def test_matrices(p=DATA_PATH):
    ism, osm, csm, gwids, splits, e2cn = load_matrices()
    devlines = load_lines(p+"dev.lines")
    print("{} dev lines".format(len(devlines)))
    testlines = load_lines(p+"test.lines")
    print("{} test lines".format(len(testlines)))
    devstart, teststart = splits

    dev_ism, dev_gwids, dev_osm, dev_e2cn = ism.matrix[devstart:teststart], gwids.matrix[devstart:teststart], \
                                            osm.matrix[devstart:teststart], e2cn[devstart:teststart]
    test_ism, test_gwids, test_osm, test_e2cn = ism.matrix[teststart:], gwids.matrix[teststart:], \
                                                osm.matrix[teststart:], e2cn[teststart:]
    rgd = {v: k for k, v in gwids.D.items()}
    rod = {v: k for k, v in osm.D.items()}

    # test question reconstruction
    for i in range(len(devlines)):
        orig_question = devlines[i][0].strip()
        reco_question = reconstruct_question(dev_ism[i], dev_gwids[i], rgd)
        assert(orig_question == reco_question)
    print("dev questions reconstruction matches")
    for i in range(len(testlines)):
        orig_question = testlines[i][0].strip()
        reco_question = reconstruct_question(test_ism[i], test_gwids[i], rgd)
        assert(orig_question == reco_question)
    print("test questions reconstruction matches")

    # test query reconstruction
    for i in range(len(devlines)):
        orig_query = devlines[i][1].strip()
        reco_query = reconstruct_query(dev_osm[i], dev_gwids[i], rod, rgd).replace("<START>", "").replace("<END>", "").strip()
        try:
            assert (orig_query == reco_query)
        except Exception as e:
            print(u"FAILED: {} \n - {}".format(orig_query, reco_query))
    print("dev queries reconstruction matches")
    for i in range(len(testlines)):
        orig_query = testlines[i][1].strip()
        reco_query = reconstruct_query(test_osm[i], test_gwids[i], rod, rgd).replace("<START>", "").replace("<END>", "").strip()
        assert (orig_query == reco_query)
    print("test queries reconstruction matches")
# endregion
# endregion

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

            # TODO: might want to disable this
            headsplits = head.split(SqlNode.suffix_sep)
            if len(headsplits) in (2, 3):
                if headsplits[1] in (SqlNode.leaf_suffix, SqlNode.last_suffix) \
                    and (len(headsplits) == 1 or (headsplits[2] in (SqlNode.leaf_suffix, SqlNode.last_suffix))):
                        head, islast, isleaf = headsplits[0], SqlNode.last_suffix in headsplits, SqlNode.leaf_suffix in headsplits

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
            else:
                order_adder_wikisql(ret)
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


def get_children_by_name(node, cre):
    for child in node.children:
        if re.match(cre, child.name):
            yield child


def querylin2json(qlin, origquestion):
    try:
        parsedtree = SqlNode.parse_sql(qlin)
        assert(parsedtree.name == "<QUERY>")            # root must by <query>
        # get select and where subtrees
        selectnode = list(get_children_by_name(parsedtree, "<SELECT>"))
        assert(len(selectnode) == 1)
        selectnode = selectnode[0]
        wherenode = list(get_children_by_name(parsedtree, "<WHERE>"))
        assert(len(wherenode) <= 1)
        if len(wherenode) == 0:
            wherenode = None
        else:
            wherenode = wherenode[0]
        assert(selectnode.name == "<SELECT>")
        assert(wherenode is None or wherenode.name == "<WHERE>")
        # get select arguments
        assert(len(selectnode.children) == 2)
        select_col = list(get_children_by_name(selectnode, "COL\d{1,2}"))
        assert(len(select_col) == 1)
        select_col = int(select_col[0].name[3:])
        select_agg = list(get_children_by_name(selectnode, "AGG\d"))
        assert(len(select_agg) == 1)
        select_agg = int(select_agg[0].name[3:])
        # get where conditions
        conds = []
        if wherenode is not None:
            for child in wherenode.children:
                assert(child.name == "<COND>")
                assert(len(child.children) == 3)
                cond_col = list(get_children_by_name(child, "COL\d{1,2}"))
                assert(len(cond_col) == 1)
                cond_col = int(cond_col[0].name[3:])
                cond_op = list(get_children_by_name(child, "OP\d"))
                assert(len(cond_op) == 1)
                cond_op = int(cond_op[0].name[2:])
                val_node = list(get_children_by_name(child, "<VAL>"))
                assert(len(val_node) == 1)
                val_node = val_node[0]
                val_nodes = val_node.children
                if val_nodes[-1].name == "<ENDVAL>":        # all should end with endval but if not, accept
                    val_nodes = val_nodes[:-1]
                valstring = u" ".join(map(lambda x: x.name, val_nodes))
                valsearch = re.escape(valstring.lower()).replace("\\ ", "\s?")
                found = re.findall(valsearch, origquestion.lower())
                if len(found) > 0:
                    found = found[0]
                    conds.append([cond_col, cond_op, found])
        return {"sel": select_col, "agg": select_agg, "conds": conds}
    except Exception as e:
        return {"agg": 10, "sel": 1345, "conds": [[5, 0, "https://www.youtube.com/watch?v=oHg5SJYRHA0"]]}


def same_sql_json(x, y):
    same = True
    same &= x["sel"] == y["sel"]
    same &= x["agg"] == y["agg"]
    same &= len(x["conds"]) == len(y["conds"])
    xconds = x["conds"] + []
    yconds = y["conds"] + []
    for xcond in xconds:
        found = False
        for j in range(len(yconds)):
            xcondval = xcond[2]
            if isinstance(xcondval, float):
                xcondval = xcondval.__repr__()
            ycondval = yconds[j][2]
            if isinstance(ycondval, float):
                ycondval = ycondval.__repr__()
            xcondval, ycondval = unicode(xcondval), unicode(ycondval)
            if xcond[0] == yconds[j][0] \
                    and xcond[1] == yconds[j][1] \
                    and xcondval.lower() == ycondval.lower():
                found = True
                del yconds[j]
                break
        same &= found
    return same


def load_jsonls(p, questionsonly=False, sqlsonly=False):
    ret = []
    with open(p) as f:
        for line in f:
            jsonl = json.loads(line)
            if questionsonly:
                question = jsonl["question"]
                ret.append(question)
            elif sqlsonly:
                sql = jsonl["sql"]
                ret.append(sql)
            else:
                ret.append(jsonl)
    return ret


def test_querylin2json():
    qlin = u"<QUERY> <SELECT> AGG0 COL3 <WHERE> <COND> COL5 OP0 <VAL> butler cc ( ks ) <ENDVAL>"
    jsonl = u"""{"phase": 1, "table_id": "1-10015132-11", "question": "What position does the player who played for butler cc (ks) play?",
                "sql": {"sel": 3, "conds": [[5, 0, "Butler CC (KS)"]], "agg": 0}}"""
    jsonl = json.loads(jsonl)
    origquestion = jsonl["question"]
    orig_sql = jsonl["sql"]
    recon_sql = querylin2json(qlin, origquestion)
    assert(same_sql_json(recon_sql, orig_sql))

    # test dev lines
    p = DATA_PATH
    devlines = load_lines(p + "dev.lines")
    failures = 0
    with open(p + "dev.jsonl") as f:
        i = 0
        for l in f:
            jsonl = json.loads(l)
            origquestion, orig_sql = jsonl["question"], jsonl["sql"]
            recon_sql = querylin2json(devlines[i][1], origquestion)
            try:
                assert (same_sql_json(recon_sql, orig_sql))
            except Exception as e:
                failures += 1
                print("FAILED: {}: {}\n-{}".format(i, orig_sql, recon_sql))
            i += 1
    if failures == 0:
        print("dev querylin2json passed")
    else:
        print("dev querylin2json: FAILED")

    # test test lines
    p = DATA_PATH
    devlines = load_lines(p+"test.lines")
    failures = 0
    with open(p+"test.jsonl") as f:
        i = 0
        for l in f:
            jsonl = json.loads(l)
            origquestion, orig_sql = jsonl["question"], jsonl["sql"]
            recon_sql = querylin2json(devlines[i][1], origquestion)
            try:
                assert(same_sql_json(recon_sql, orig_sql))
            except Exception as e:
                failures += 1
                print("FAILED: {}: {}\n-{}".format(i, orig_sql, recon_sql))
            i += 1
    if failures == 0:
        print("test querylin2json passed")
    else:
        print("test querylin2json: FAILED - expected 1 failure")
    # !!! example 15485 in test fails because wrong gold constructed in lines (wrong occurence of 2-0 is taken)


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
def test_sqlnode_and_sqls(x=0):
    orig_question = "which city of license has a frequency mhz smaller than 100.9 , and a erp w larger than 100 ?"
    orig_line = "<QUERY> <SELECT> AGG0 COL2 " \
                "<WHERE> <COND> COL1 OP2 <VAL> 100.9 <ENDVAL> <COND> COL3 OP1 <VAL> 100 <ENDVAL>"
    orig_tree = SqlNode.parse_sql(orig_line)
    orig_sql = querylin2json(orig_line, orig_question)

    print(orig_tree.pptree())

    swapped_conds_line = "<QUERY> <SELECT> AGG0 COL2 " \
                "<WHERE> <COND> COL3 OP1 <VAL> 100 <ENDVAL> <COND> COL1 OP2 <VAL> 100.9 <ENDVAL>"

    swapped_conds_tree = SqlNode.parse_sql(swapped_conds_line)
    swapped_conds_sql = querylin2json(swapped_conds_line, orig_question)

    print("swapped conds testing:")
    assert(orig_tree == swapped_conds_tree)
    print("trees same")
    assert(same_sql_json(orig_sql, swapped_conds_sql))
    print("sqls same")

    swapped_select_args_line = "<QUERY> <SELECT> COL2 AGG0 " \
                "<WHERE> <COND> COL1 OP2 <VAL> 100.9 <ENDVAL> <COND> COL3 OP1 <VAL> 100 <ENDVAL>"
    swapped_select_args_tree = SqlNode.parse_sql(swapped_select_args_line)
    swapped_select_args_sql = querylin2json(swapped_select_args_line, orig_question)
    print("swapped select args testing:")
    assert(orig_tree != swapped_select_args_tree)
    print("trees NOT same")
    assert(same_sql_json(orig_sql, swapped_select_args_sql))
    print("sqls same")

    wrong_line = "<QUERY> <SELECT> AGG0 COL2 " \
                "<WHERE> <COND> COL1 OP2 <VAL> 100.92 <ENDVAL> <COND> COL3 OP1 <VAL> 100 <ENDVAL>"
    wrong_tree = SqlNode.parse_sql(wrong_line)
    wrong_sql = querylin2json(wrong_line, orig_question)
    print("wrong query testing:")
    assert(orig_tree != wrong_tree)
    print("trees NOT same")
    assert(not same_sql_json(orig_sql, wrong_sql))
    print("sqls NOT same")

    bad_line = "<QUERY> <SELECT> AGG0 COL2 " \
                "<WHERE> <COND> COL1 OP2 <VAL> 100.9 <ENDVAL> <COND> COL3 OP1 <VAL> 100 "
    bad_tree = SqlNode.parse_sql(bad_line)
    print("bad correct line: tree parsed")
    bad_sql = querylin2json(bad_line, orig_question)
    assert(same_sql_json(orig_sql, bad_sql))
    print("sqls same")

    # test with broken lines
    linesplits = orig_line.split()
    for i in range(len(linesplits)-1):      # every except last one
        for j in range(i+1, len(linesplits)):
            broken_line = linesplits[:i] + linesplits[j:]
            broken_line = " ".join(broken_line)
            try:
                broken_tree = SqlNode.parse_sql(broken_line)
                broken_sql = querylin2json(broken_line, orig_question)
                assert(broken_tree != orig_tree)
                if " ".join(linesplits[i:j]) not in ("<ENDVAL>"):
                    assert(not same_sql_json(broken_sql, orig_sql))
            except q.SumTingWongException as e:
                # didn't parse
                pass
    print("all brokens passed")



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
    return tracker


# region test
def test_grouptracker():
    # TODO: test that every possible tree is correct tree and leads to correct sql
    ism, osm, csm, psm, splits, e2cn = load_matrices()
    devstart, teststart = splits

    tracker = make_tracker_df(osm)

    devstart = 74500

    for i in range(devstart, len(osm.matrix)):
        accs = set()
        numconds = len(re.findall("<COND>", tracker.trackables[i].pp()))
        numsamples = {0: 3, 1: 3, 2: 5, 3: 10, 4: 100, 5: 1000}[numconds]
        for j in range(numsamples):
            acc = u""
            tracker.reset()
            while True:
                if tracker.is_terminated(i):
                    break
                vnt = tracker.get_valid_next(i)
                sel = random.choice(vnt)
                acc += u" " + tracker.rD[sel]
                tracker.update(i, sel)
            accs.add(acc)
            assert(SqlNode.parse_sql(acc).equals(tracker.trackables[i]))
            if not SqlNode.parse_sql(unidecode(acc)).equals(tracker.trackables[i]):
                print(acc)
                print(tracker.trackables[i].pptree())
                print(SqlNode.parse_sql(unidecode(acc)).pptree())
                raise q.SumTingWongException("trees not equal")
        assert(len(accs) > 0)
        print("number of unique linearizations for example {}: {} - {}".format(i, len(accs), numconds))

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

        Computer can be a DynamicVecPreparer or a DynamicVecComputer.
        The .prepare() method must be called at the beginning of every batch
            with the per-example data batch used to compute the vectors.

        If a Preparer is used, vectors are prepared for every example in the batch.
        The Preparer receives the data, per example and must compute different sets of vectors for different examples.
        During forward, the prepared vectors are sliced from, on a per-example basis.

        If a Computer is used, .prepare() only stores the given data,
            which are passed to the computer during forward, together with the actual input.
        """

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
    """ Must be used with a DynamicVecPreparer (see DynamicWordEmb doc).
        As with DynamicWordEmb, the vectors used in this layer are different for every example.
        .prepare() must be called with the per-example data batch at the beginning of the batch.
    """
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
    """ Encoder for column names.
        Uses one LSTM layer.
    """
    def __init__(self, dim, colbaseemb, nocolid=None):
        super(ColnameEncoder, self).__init__()
        self.emb = colbaseemb
        self.embdim = colbaseemb.vecdim
        self.dim = dim
        self.enc = torch.nn.LSTM(self.embdim, self.dim, 1, batch_first=True)
        self.nocolid = nocolid

    def forward(self, x):
        """ input is (batsize, numcols, colnamelen)
            out is (batsize, numcols, dim)
        """
        # TODO: test
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
        # y_t, umask = q.seq_unpack(_y_t, order)
        ret = y_T.contiguous().view(x.size(0), x.size(1), y_T.size(-1))
        return ret, rmask


class OutVecComputer(DynamicVecPreparer):
    """ This is a DynamicVecPreparer used for both output embeddings and output layer.
        To be created, needs:
         * syn_emb:         normal syntax embedding
         * syn_trans:       normal syntax trans - if sliced with osm.D ids, return ids to use with syn_emb in .prepare()
         * inpbaseemb:      embedder for input words
         * inp_trans:       input words trans - if sliced with osm.D ids, return ids to use with inpbaseemb and inpmaps in .prepare()
         * colencoder:      encoder for column names
         * col_trans:       column names trans
         * worddic:         dictionary of output symbols (synids, colids, uwids) = osm.D
    """
    def __init__(self, syn_emb, syn_trans, inpbaseemb, inp_trans,
                 colencoder, col_trans, worddic, colzero_to_inf=False,
                 rare_gwids=None):
        super(OutVecComputer, self).__init__()
        self.syn_emb = syn_emb
        self.syn_trans = syn_trans
        self.inp_emb = inpbaseemb
        self.inp_trans = inp_trans
        self.col_enc = colencoder
        self.col_trans = col_trans
        self.D = worddic
        self.colzero_to_inf = colzero_to_inf

        # initialize rare vec to rare vector from syn_emb
        self.rare_vec = torch.nn.Parameter(syn_emb.embedding.weight[syn_emb.D["<RARE>"]].data)
        self.rare_gwids = rare_gwids

        if self.inp_emb.vecdim != self.syn_emb.vecdim:
            print("USING LIN ADAPTER in OUT")
            self.inpemb_trans = torch.nn.Linear(self.inp_emb.vecdim, self.syn_emb.vecdim, bias=False)
        else:
            self.inpemb_trans = None

    def prepare(self, inpmaps, colnames):
        """ inpmaps (batsize, num_uwids) contains mapping from uwids to gwids for every example = batch from gwids matrix
            colnames (batsize, numcols, colnamelen) contains colnames for every example
        """
        x = q.var(torch.arange(0, len(self.D))).cuda(inpmaps).v.long()      # prepare for all possible input tokens (from osm.D)
        batsize = inpmaps.size(0)

        # region syntax words
        _syn_ids = self.syn_trans[x]            # maps input ids to syn ids
        # computes vectors and mask for syn ids from input
        _syn_embs, _syn_mask = self.syn_emb(_syn_ids.unsqueeze(0).repeat(batsize, 1))
        # repeat(batsize, 1) because syn embs are same across examples
        #   --> produces (batsize, vocsize) indexes that are then embedded to (batsize, vocsize, embdim)
        # endregion

        # region input words --> used in output embedding but normally overridden by BFOL for output scores
        _inp_ids = self.inp_trans[x]            # maps input ids to uwid numbers
        # gets gwids for uwids using inpmaps (=batch from gwids)
        transids = torch.gather(inpmaps, 1, _inp_ids.unsqueeze(0).repeat(batsize, 1))       # in gwids
        # repeat(batsize, 1) because mapping from input ids to uwid number is same across all examples
        #   --> produces (batsize, vocsize) matrix of uwid numbers
        #           that are then used to gather to (batsize, vocsize) matrix of gwid ids

        # embeds retrieved gwids using provided inpbaseemb
        _inp_embs, _inp_mask = self.inp_emb(transids)               # inp_emb embeds gwids
        # if expected vector size doesn't match inpemb vector size, apply linear transform to adapt
        if self.inpemb_trans is not None:
            _inp_embs = self.inpemb_trans(_inp_embs)
        _inp_embs = replace_rare_gwids_with_rare_vec(_inp_embs, transids, self.rare_gwids, self.rare_vec)
        # _inp_mask should zero out _inp_embs if there is something wrong in rare vec replacement -> TEST!
        # endregion

        # region column names
        _colencs, _col_mask = self.col_enc(colnames)        # encode given column names
        _col_ids = self.col_trans[x]                        # map input ids to col ids
        _col_ids = _col_ids.unsqueeze(0).repeat(batsize, 1) # mapped col ids are shared across all examples
        # ???
        _col_trans_mask = (_col_ids > -1).long()
        _col_ids += (1 - _col_trans_mask)
        _col_mask = torch.gather(_col_mask, 1, _col_ids)
        _col_ids = _col_ids.unsqueeze(2).repeat(1, 1, _colencs.size(2))     # because colens are (batsize, numcols, embdim)

        _col_embs = torch.gather(_colencs, 1, _col_ids)     # gather the right column ids for every example --> (batsize, vocsize, embdim)
        _col_mask = _col_mask.float() * _col_trans_mask.float()
        # endregion

        # _col_mask = _col_mask.float() - _inp_mask.float() - _syn_mask.float()

        # region combine
        _totalmask = _syn_mask.float() + _inp_mask.float() + _col_mask.float()

        assert (np.all(_totalmask.cpu().data.numpy() < 2))

        # _col_mask = (x > 0).float() - _inp_mask.float() - _syn_mask.float()

        # merge --> (batsize, vocsize, embdim)
        ret =   _syn_embs * _syn_mask.float().unsqueeze(2) \
              + _inp_embs * _inp_mask.float().unsqueeze(2) \
              + _col_embs * _col_mask.float().unsqueeze(2)

        # endregion
        # _pp = osm.pp(x.cpu().data.numpy())
        return ret, _totalmask


class BFOL(DynamicWordLinout):
    """ Implements pointer-based scores over uwids, overrides uwid scores produced by OutVecComputer() """
    def __init__(self, computer=None, worddic=None, ismD=None, inp_trans=None, nocopy=False):
        """ computer is an OutVecComputer() """
        super(BFOL, self).__init__(computer=computer, worddic=worddic)
        self.inppos2uwid = None     # maps positions in the input to uwids
        self.inpenc = None
        self.ismD = ismD
        self.inp_trans = inp_trans
        self.nocopy = nocopy

    def prepare(self, inpseq, inpenc, *xdata):
        """ inpseq: (batsize, seqlen) """
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
            # just outveccomputer results
            return ret, rmask

        xshape = x.size()
        if len(xshape) == 2:
            x = x.unsqueeze(1)

        # compute scores over input positions
        # x will be twice the size of inpenc -> which part of x to take???
        compx = x[:, :, :x.size(2) // 2]  # slice out first half, which is y_t (not ctx_t)
        scores = torch.bmm(compx, self.inpenc.transpose(2, 1))

        # translate scores over input positions to scores over uwids
        inppos2uwid = self.inppos2uwid[:, :scores.size(-1), :]  # in case input matrix shrank because of seq packing
        offset = (torch.min(scores) - 1000).data[0]
        umask = (inppos2uwid == 0).float()
        uwid_scores = scores.transpose(2, 1) * inppos2uwid
        uwid_scores = uwid_scores + offset * umask

        # take max scores
        uwid_scores, _ = torch.max(uwid_scores, 1)
        uwid_scores_mask = (inppos2uwid.sum(1) > 0).float()  # (batsize, #uwid)

        # replace old scores over uwids with the just computer pointer-based scores over uwids
        sel_uwid_scores = uwid_scores.index_select(1, self.inp_trans)
        sel_uwid_scores_mask = uwid_scores_mask.index_select(1, self.inp_trans)
        # the zeros in seluwid mask for those uwids should already be there in rmask
        rret = ret * (1 - sel_uwid_scores_mask) + sel_uwid_scores_mask * sel_uwid_scores
        if len(xshape) == 2:
            rret = rret.squeeze(1)
        # assert(((rret != 0.).float() - rmask.float()).norm().cpu().data[0] == 0)
        return rret, rmask
# endregion


# region dynamic vector modules helper functions
def replace_rare_gwids_with_rare_vec(x, ids, rare_gwids, rare_vec):
    if rare_gwids is None:
        return x
    # get mask based on where rare_gwids occur in ids
    ids_np = ids.cpu().data.numpy()
    ids_mask_np = np.vectorize(lambda x: x not in rare_gwids)(ids_np).astype("uint8")       # ids_mask is one if NOT rare
    ids_mask = q.var(ids_mask_np).cuda(x).v.float()
    # switch between vectors
    ret = rare_vec.unsqueeze(0).unsqueeze(1) * (1 - ids_mask.unsqueeze(2))
    ret = ret + x * ids_mask.unsqueeze(2)
    # TODO: test properly
    return ret


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
                          useglove=True, gdim=None, gfrac=0.1,
                          rare_gwids=None):
    # base embedder for input tokens
    embdim = gdim if gdim is not None else dim
    if inpbaseemb is None:
        inpbaseemb = q.WordEmb(dim=embdim, worddic=psm.D)
        if useglove:
            inpbaseemb = q.PartiallyPretrainedWordEmb(dim=embdim, worddic=psm.D, gradfracs=(1., gfrac))

    # base embedder for column names
    if colbaseemb is None:
        colbaseemb = q.WordEmb(embdim, worddic=csm.D)
        if useglove:
            colbaseemb = q.PartiallyPretrainedWordEmb(dim=embdim, worddic=csm.D, gradfracs=(1., gfrac))

    synD, inpD, colD, syn_trans, inp_trans, col_trans = build_subdics(osm)

    syn_emb = q.WordEmb(dim, worddic=synD)

    if colenc is None:
        colenc = ColnameEncoder(dim, colbaseemb, nocolid=csm.D["nonecolumnnonecolumnnonecolumn"])

    computer = OutVecComputer(syn_emb, syn_trans, inpbaseemb, inp_trans, colenc, col_trans, osm.D,
                              rare_gwids=rare_gwids)
    return computer, inpbaseemb, colbaseemb, colenc
# endregion


# region dynamic vector module creation functions
def make_inp_emb(dim, ism, psm, useglove=True, gdim=None, gfrac=0.1,
                 rare_gwids=None):
    embdim = gdim if gdim is not None else dim
    baseemb = q.WordEmb(dim=embdim, worddic=psm.D)
    if useglove:
        baseemb = q.PartiallyPretrainedWordEmb(dim=embdim, worddic=psm.D, gradfracs=(1., gfrac))

    class Computer(DynamicVecComputer):
        def __init__(self):
            super(Computer, self).__init__()
            self.baseemb = baseemb
            self.rare_vec = torch.nn.Parameter(baseemb.embedding.weight[baseemb.D["<RARE>"]].data)
            if embdim != dim:
                print("USING LIN ADAPTER")
                self.trans = torch.nn.Linear(embdim, dim, bias=False)
            else:
                self.trans = None

        def forward(self, x, data):
            transids = torch.gather(data, 1, x)         # transids are in gwids
            # _pp = psm.pp(transids[:5].cpu().data.numpy())
            _embs, mask = self.baseemb(transids)        # baseemb embedds gwids
            if self.trans is not None:
                _embs = self.trans(_embs)
            _embs = replace_rare_gwids_with_rare_vec(_embs, transids, rare_gwids, self.rare_vec)
            return _embs

    emb = DynamicWordEmb(computer=Computer(), worddic=ism.D)
    return emb, baseemb


def make_out_emb(dim, osm, psm, csm, inpbaseemb=None, colbaseemb=None,
                 useglove=True, gdim=None, gfrac=0.1, colenc=None,
                 rare_gwids=None):
    print("MAKING OUT EMB")
    comp, inpbaseemb, colbaseemb, colenc \
        = make_out_vec_computer(dim, osm, psm, csm, inpbaseemb=inpbaseemb, colbaseemb=colbaseemb,
                                colenc=colenc, useglove=useglove, gdim=gdim, gfrac=gfrac,
                                rare_gwids=rare_gwids)
    return DynamicWordEmb(computer=comp, worddic=osm.D), inpbaseemb, colbaseemb, colenc


def make_out_lin(dim, ism, osm, psm, csm, inpbaseemb=None, colbaseemb=None,
                 useglove=True, gdim=None, gfrac=0.1, colenc=None, nocopy=False,
                 rare_gwids=None):
    print("MAKING OUT LIN")
    comp, inpbaseemb, colbaseemb, colenc \
        = make_out_vec_computer(dim, osm, psm, csm, inpbaseemb=inpbaseemb, colbaseemb=colbaseemb,
                                colenc=colenc, useglove=useglove, gdim=gdim, gfrac=gfrac,
                                rare_gwids=rare_gwids)
    inp_trans = comp.inp_trans  # to index
    out = BFOL(computer=comp, worddic=osm.D, ismD=ism.D, inp_trans=inp_trans, nocopy=nocopy)
    return out, inpbaseemb, colbaseemb, colenc

# endregion


# endregion

# region MAIN SCRIPTS
# region main scripts helper functions
def get_rare_stats(trainism, traingwids, gwidsD, gdic, rarefreq=2):
    """  get rare gwids, missing gwids, ignoring glove (glove should be taken into account in the modules) """
    rD = {v: k for k, v in gwidsD.items()}
    # count all unique ids in traingwids
    uniquegwids, gwid_counts = np.unique(traingwids, return_counts=True)
    rare_gwids = gwid_counts <= rarefreq
    number_rare = np.sum(rare_gwids.astype("int32"))
    print("{} gwids with freq <= {} (counted in train)".format(number_rare, rarefreq))
    unique_nonrare_ids = uniquegwids * (~rare_gwids).astype("int32")
    unique_nonrare_ids = set(unique_nonrare_ids)
    unique_nonrare_words = set([rD[unrid] for unrid in unique_nonrare_ids])
    rare_words = set(gwidsD.keys()) - unique_nonrare_words - set(gdic.keys())
    rare_gwids_after_glove = set([gwidsD[rare_word] for rare_word in rare_words])
    print("{} gwids with freq <= {} (counted in train) and not in used glove".format(len(rare_gwids_after_glove), rarefreq))
    return rare_gwids_after_glove


def do_rare_in_colnames(cnsm, traincolids, gdic, rarefreq=3, replace=False):
    """ cnsm: (#cols, colnamelen) -- traincolids: (numex, #cols) """
    # print(np.sum(cnsm._matrix == cnsm.D["<RARE>"]))
    tt = q.ticktock("rare column names")
    tt.tick("computing counts")
    rD = {v: k for k, v in cnsm.D.items()}
    # print("{} words in cnsm.D".format(len(cnsm.D)))
    wordcounts = {w: 0 for w in cnsm.D.keys()}

    uniquecolids, colidcounts = np.unique(traincolids, return_counts=True)
    for i in range(len(uniquecolids)):
        for j in range(len(cnsm.matrix[uniquecolids[i]])):
            word = rD[cnsm.matrix[uniquecolids[i], j]]
            wordcounts[word] += colidcounts[i]

    rarewords = set()
    for k, v in wordcounts.items():
        if v <= rarefreq:
            rarewords.add(k)
    # print("nonecolumnnonecolumnnonecolumn" in rarewords)
    rarewords -= {"nonecolumnnonecolumnnonecolumn",}
    print("{} rare words (rarefreq {}) out of {} unique words in col names"
          .format(len(rarewords), rarefreq, len(wordcounts)))
    rarewords_notinglove = rarewords - set(gdic.keys())
    print("{} rare words (rarefreq {}) not in glove, out of {} unique words in col names"
          .format(len(rarewords_notinglove), rarefreq, len(wordcounts)))
    tt.tock("counts computed")
    if replace:
        cnsm._matrix = np.vectorize(lambda x: x if rD[x] not in rarewords_notinglove else cnsm.D["<RARE>"])(cnsm.matrix)
        tt.msg("total rare words replaced with rare id: {}".format(np.sum(cnsm.matrix == cnsm.D["<RARE>"])))
    return cnsm


def make_oracle_df(tracker, mode=None):
    ttt = q.ticktock("oracle maker")
    print("oracle mode: {}".format(mode))
    oracle = q.DynamicOracleRunner(tracker=tracker,
                                   mode=mode,
                                   explore=0.)

    if _opt_test:
        print("TODO: oracle tests")
    return oracle


def get_output(model, data, origquestions, batsize=100, inp_bt=None, cuda=False,
               rev_osm_D=None, rev_gwids_D=None):
    """ takes a model (must be freerunning !!!) """
    gwids = data[2]
    # TODO: make sure q.eval() doesn't feed anything wrong or doesn't forget to reset things
    dataloader = q.dataload(*data, batch_size=batsize, shuffle=False)
    predictions = q.eval(model).on(dataloader) \
        .set_batch_transformer(inp_bt).cuda(cuda).run()
    _, predictions = predictions.max(2)
    predictions = predictions.cpu().data.numpy()
    rawlines = []
    sqls = []
    for i in range(len(gwids)):
        rawline = reconstruct_query(predictions[i], gwids[i], rev_osm_D, rev_gwids_D)
        rawline = rawline.replace("<START>", "").strip()
        rawline = rawline.split("<END>")[0].strip()
        rawlines.append(rawline)
        sqljson = querylin2json(rawline, origquestions[i])
        sqls.append(sqljson)
    return rawlines, sqls


def evaluate_model(m, devdata, testdata, rev_osm_D, rev_gwids_D,
                   inp_bt=None, batsize=100, cuda=False, savedir=None, test=False):
    def compute_sql_acc(pred_sql, gold_sql):
        sql_acc = 0.
        sql_acc_norm = 1e-6
        for pred_sql_i, gold_sql_i in zip(pred_sql, gold_sql):
            sql_acc_norm += 1
            sql_acc += 1. if same_sql_json(pred_sql_i, gold_sql_i) else 0.
        return sql_acc / sql_acc_norm


    def save_lines(lines, fname):
        with codecs.open(savedir + '/' + fname, "w", encoding="utf-8") as f:
            for lin in lines:
                f.write(u"{}\n".format(lin))


    # dev predictions
    devquestions = load_jsonls(DATA_PATH + "dev.jsonl", questionsonly=True)
    devsqls = load_jsonls(DATA_PATH + "dev.jsonl", sqlsonly=True)

    if test:
        devquestions = load_jsonls(DATA_PATH + "train.jsonl", questionsonly=True)[200:250]
        devsqls = load_jsonls(DATA_PATH + "train.jsonl", sqlsonly=True)[200:250]

    pred_devlines, pred_devsqls = get_output(m, devdata, devquestions,
                                             batsize=batsize, inp_bt=inp_bt, cuda=cuda,
                                             rev_osm_D=rev_osm_D, rev_gwids_D=rev_gwids_D)
    if savedir is not None:
        save_lines(pred_devlines, "dev_pred.lines")
        save_lines(map(lambda x: json.dumps(x), pred_devsqls), "dev_pred.jsonl")

    dev_sql_acc = compute_sql_acc(pred_devsqls, devsqls)
    print("DEV SQL ACC: {}".format(dev_sql_acc))

    # test predictions
    testquestions = load_jsonls(DATA_PATH + "test.jsonl", questionsonly=True)
    testsqls = load_jsonls(DATA_PATH + "test.jsonl", sqlsonly=True)
    pred_testlines, pred_testsqls = get_output(m, testdata, testquestions,
                                               batsize=batsize, inp_bt=inp_bt, cuda=cuda,
                                               rev_osm_D=rev_osm_D, rev_gwids_D=rev_gwids_D)
    if savedir is not None:
        save_lines(pred_testlines, "test_pred.lines")
        save_lines(map(lambda x: json.dumps(x), pred_testsqls), "test_pred.jsonl")

    test_sql_acc = compute_sql_acc(pred_testsqls, testsqls)
    print("TEST SQL ACC: {}".format(test_sql_acc))
    return dev_sql_acc, test_sql_acc


# region test
def test_reconstruct_save_reload_and_eval():
    """ load matrices, get test set, reconstruct using get_output()'s logic, save, read, test """
    tt = q.ticktock("testsave")
    ism, osm, cnsm, gwids, splits, e2cn = load_matrices()
    rev_osm_D = {v: k for k, v in osm.D.items()}
    rev_gwids_D = {v: k for k, v in gwids.D.items()}
    _, teststart = splits
    testquestions = load_jsonls(DATA_PATH+"test.jsonl", questionsonly=True)
    testsqls = load_jsonls(DATA_PATH+"test.jsonl", sqlsonly=True)

    tt.tick("reconstructing test gold...")
    ism, osm, gwids = ism.matrix[teststart:], osm.matrix[teststart:], gwids.matrix[teststart:]
    rawlines, sqls = [], []
    for i in range(len(gwids)):
        rawline = reconstruct_query(osm[i], gwids[i], rev_osm_D, rev_gwids_D)
        rawline = rawline.replace("<START>", "").replace("<END>", "").strip()
        rawlines.append(rawline)
        sqljson = querylin2json(rawline, testquestions[i])
        sqls.append(sqljson)
    tt.tock("reconstructed")

    tt.tick("saving test gold...")
    # saving without codecs doesn't work
    with codecs.open("testsave.lines", "w", encoding="utf-8") as f:
        for line in rawlines:
            f.write(u"{}\n".format(line))

    with codecs.open("testsave.sqls", "w", encoding="utf-8") as f:
        for sql in sqls:
            f.write(u"{}\n".format(json.dumps(sql)))
    tt.tock("saved")

    tt.tick("reloading saved...")
    reloaded_lines = []
    with codecs.open("testsave.lines", encoding="utf-8") as f:
    # with open("testsave.lines") as f:
        for line in f:
            reloaded_lines.append(line.strip())

    reloaded_sqls = []
    # with codecs.open("testsave.sqls", encoding="utf-8") as f:
    with open("testsave.sqls") as f:
        for line in f:
            sql = json.loads(line)
            reloaded_sqls.append(sql)
    tt.tock("reloaded saved")

    for rawline, reloadedline in zip(rawlines, reloaded_lines):
        if not rawline == reloadedline:
            print(u"FAILED: '{}' \n - '{}'".format(rawline, reloadedline))
        assert(rawline == reloadedline)

    failures = 0
    for testsql, reloaded_sql in zip(testsqls, reloaded_sqls):
        if not same_sql_json(testsql, reloaded_sql):
            print("FAILED: {} \n - {} ".format(testsql, reloaded_sql))
            failures += 1
        # assert(same_sql_json(testsql, reloaded_sql))
    assert(failures == 1)
    print("only one failure")

    print(len(reloaded_lines))
# endregion

# endregion


def run_seq2seq_tf(lr=0.001, batsize=100, epochs=100,
                   inpembdim=50, outembdim=50, innerdim=100, numlayers=1, dim=-1, gdim=-1,
                   dropout=0.2, rdropout=0.1, edropout=0., idropout=0., irdropout=0.,
                   wreg=0.000000000001, gradnorm=5., useglove=True, gfrac=0.01,
                   cuda=False, gpu=0, tag="none", ablatecopy=False, test=False,
                   tieembeddings=False, dorare=False):
    # region init
    settings = locals().copy()
    logger = q.Logger(prefix="wikisql_s2s_clean")
    logger.save_settings(**settings)
    logger.update_settings(completed=False)
    print("LOGGER PATH: {}".format(logger.p))
    logger.update_settings(version="1")

    model_save_path = os.path.join(logger.p, "model")

    print("Seq2Seq + TF (clean)")
    print("PyTorch initial seed: {}".format(torch.initial_seed()))
    if cuda:    torch.cuda.set_device(gpu)
    tt = q.ticktock("script")
    # endregion

    # region dimensions
    gdim = None if gdim < 0 else gdim

    if dim > 0:
        innerdim = dim              # total dimension of encoding
        inpembdim = dim // 2        # dimension of input embedding
        outembdim = dim // 2        # dimension of output embedding
        if gdim is not None:
            inpembdim = gdim
            outembdim = gdim

    outlindim = innerdim * 2    # dimension of output layer - twice the encoding dim because cat of enc and state

    encdim = innerdim // 2          # half the dimension because bidirectional encoder doubles it back
    encdims = [inpembdim] + [encdim] * numlayers    # encoder's layers' dimensions
    decdims = [outembdim] + [innerdim] * numlayers  # decoder's layers' dimensions
    # endregion

    # region data
    ism, osm, cnsm, gwids, splits, e2cn = load_matrices()
    gwids._matrix = gwids.matrix * (gwids.matrix != gwids.D["<RARE>"])
    devstart, teststart = splits
    eids = np.arange(0, len(ism), dtype="int64")
    # splits
    if test:    devstart, teststart, batsize = 200, 250, 50
    datamats = [ism.matrix, osm.matrix, gwids.matrix, e2cn]
    traindata = [datamat[:devstart] for datamat in datamats]
    devdata = [datamat[devstart:teststart] for datamat in datamats]
    testdata = [datamat[teststart:] for datamat in datamats]

    rev_osm_D = {v: k for k, v in osm.D.items()}
    rev_gwids_D = {v: k for k, v in gwids.D.items()}

    gdic = q.PretrainedWordEmb(gdim).D
    rare_gwids_after_glove = get_rare_stats(traindata[0], traindata[2], gwids.D, gdic)
    print("{} doing rare".format("NOT" if not dorare else ""))
    if not dorare:
        rare_gwids_after_glove = None
    cnsm = do_rare_in_colnames(cnsm, traindata[-1], gdic, replace=dorare)
    # endregion

    # region submodules
    inpemb, inpbaseemb = make_inp_emb(inpembdim, ism, gwids, useglove=useglove, gdim=gdim, gfrac=gfrac, rare_gwids=rare_gwids_after_glove)
    outemb, inpbaseemb, colbaseemb, _ = make_out_emb(outembdim, osm, gwids, cnsm, gdim=gdim,
                                                  inpbaseemb=inpbaseemb, useglove=useglove, gfrac=gfrac, rare_gwids=rare_gwids_after_glove)
    if not tieembeddings:
        inpbaseemb, colbaseemb = None, None

    outlin, inpbaseemb, colbaseemb, colenc = make_out_lin(outlindim, ism, osm, gwids, cnsm,
                                                          useglove=useglove, gdim=gdim, gfrac=gfrac,
                                                          inpbaseemb=inpbaseemb, colbaseemb=colbaseemb,
                                                          colenc=None, nocopy=ablatecopy, rare_gwids=rare_gwids_after_glove)

    encoder = q.FastestLSTMEncoder(*encdims, dropout_in=idropout, dropout_rec=irdropout, bidir=True)
    # endregion

    # region encoder-decoder definition
    class EncDec(torch.nn.Module):
        def __init__(self, dec, maxtime=None):
            super(EncDec, self).__init__()
            self.inpemb, self.outemb, self.outlin, self.encoder, self.decoder \
                = inpemb, outemb, outlin, encoder, dec
            self.maxtime = maxtime

        def forward(self, inpseq, outseq, inpseqmaps, colnames):
            # encoding
            self.inpemb.prepare(inpseqmaps)
            _inpembs, _inpmask = self.inpemb(inpseq)
            _inpenc = self.encoder(_inpembs, mask=_inpmask)
            inpmask = _inpmask[:, :_inpenc.size(1)]
            inpenc = q.intercat(_inpenc.chunk(2, -1), -1)       # TODO: do we need intercat?
            ctx = inpenc    # old normalpointer mode

            # decoding
            self.outemb.prepare(inpseqmaps, colnames)
            self.outlin.prepare(inpseq, inpenc, inpseqmaps, colnames)

            decoding = self.decoder(outseq, ctx=ctx, ctx_0=ctx[:, -1, :],
                                    ctxmask=inpmask, maxtime=osm.matrix.shape[1]-1)
            # TODO: why -1 in maxtime?
            # --? maybe because that's max we need to do, given that gold seqs are -1 in len

            return decoding
    # endregion

    # region decoders and model, for train and test
    layers = [q.LSTMCell(decdims[i-1], decdims[i], dropout_in=dropout, dropout_rec=rdropout)
              for i in range(1, len(decdims))]
    decoder_top = q.AttentionContextDecoderTop(q.Attention().dot_gen(),
                                               q.Dropout(edropout),
                                                  outlin, ctx2out=False)
    decoder_core = q.DecoderCore(outemb, *layers)
    decoder_cell = q.ModularDecoderCell(decoder_core, decoder_top)
    decoder_cell.set_runner(q.TeacherForcer())
    decoder = decoder_cell.to_decoder()

    m = EncDec(decoder)         # ONLY USE FOR TRAINING !!!

    valid_decoder_cell = q.ModularDecoderCell(decoder_core, decoder_top)
    valid_decoder_cell.set_runner(q.FreeRunner())
    valid_decoder = valid_decoder_cell.to_decoder()

    valid_m = EncDec(valid_decoder)     # use for valid
    # TODO: verify that valid_m doesn't get something wrong !
    # endregion

    # region training preparation
    trainloader = q.dataload(*traindata, batch_size=batsize, shuffle=True)
    validloader = q.dataload(*devdata, batch_size=batsize, shuffle=False)
    testloader = q.dataload(*testdata, batch_size=batsize, shuffle=False)

    losses = q.lossarray(q.SeqCrossEntropyLoss(ignore_index=0),
                         q.SeqAccuracy(ignore_index=0))

    row2tree = lambda x: SqlNode.parse_sql(osm.pp(x))

    validlosses = q.lossarray(q.SeqAccuracy(ignore_index=0),
                              TreeAccuracy(ignore_index=0, treeparser=row2tree))

    logger.update_settings(optimizer="adam")
    optim = torch.optim.Adam(q.paramgroups_of(m), lr=lr, weight_decay=wreg)

    def inp_bt(ismbatch, osmbatch, gwidsbatch, colnameids):
        colnames = cnsm.matrix[colnameids.cpu().data.numpy()]
        colnames = q.var(colnames).cuda(colnameids).v
        return ismbatch, osmbatch[:, :-1], gwidsbatch, colnames, osmbatch[:, 1:]

    def valid_inp_bt(ismbatch, osmbatch, gwidsbatch, colnameids):
        colnames = cnsm.matrix[colnameids.cpu().data.numpy()]
        colnames = q.var(colnames).cuda(colnameids).v
        return ismbatch, osmbatch[:, 0], gwidsbatch, colnames, osmbatch[:, 1:]

    # saving best model
    best_saver = BestSaver(lambda: validlosses.get_agg_errors()[1],
                           valid_m, path=model_save_path, verbose=True)
    # endregion

    # region training
    q.train(m).train_on(trainloader, losses)\
        .optimizer(optim).clip_grad_norm(gradnorm).set_batch_transformer(inp_bt)\
        .valid_with(valid_m).valid_on(validloader, validlosses).set_valid_batch_transformer(valid_inp_bt)\
        .cuda(cuda).hook(logger).hook(best_saver)\
        .train(epochs)
    logger.update_settings(completed=True)

    # grad check inspection: only output layer inpemb and inpemb_trans have zero-norm grads,
    # because they're overridden in BFOL
    # endregion

    # region evaluation
    tt.tick("Loading best model...")
    valid_m.load_state_dict(torch.load(model_save_path))

    dev_sql_acc, test_sql_acc = evaluate_model(valid_m, devdata, testdata, rev_osm_D, rev_gwids_D,
                                               inp_bt=valid_inp_bt, batsize=batsize, cuda=cuda,
                                               savedir=logger.p, test=test)
    # endregion


def run_seq2seq_oracle_df(lr=0.001, batsize=100, epochs=100,
                          inpembdim=50, outembdim=50, innerdim=100, numlayers=1, dim=-1, gdim=-1,
                          dropout=0.2, rdropout=0.1, edropout=0., idropout=0., irdropout=0.,
                          wreg=0.0000000000001, gradnorm=5., useglove=True, gfrac=0.01,
                          cuda=False, gpu=0, tag="none", ablatecopy=False, test=False,
                          tieembeddings=False, dorare=False,
                          oraclemode="zerocost"):
    # region init
    settings = locals().copy()
    logger = q.Logger(prefix="wikisql_s2s_oracle_df_clean")
    logger.save_settings(**settings)
    logger.update_settings(completed=False)
    print("LOGGER PATH: {}".format(logger.p))
    logger.update_settings(version="1")

    model_save_path = os.path.join(logger.p, "model")

    print("Seq2Seq + ORACLE (clean)")
    print("PyTorch initial seed: {}".format(torch.initial_seed()))
    if cuda:    torch.cuda.set_device(gpu)
    tt = q.ticktock("script")
    # endregion

    # region dimensions     # exactly the same as tf script
    gdim = None if gdim < 0 else gdim

    if dim > 0:
        innerdim = dim  # total dimension of encoding
        inpembdim = dim // 2  # dimension of input embedding
        outembdim = dim // 2  # dimension of output embedding
        if gdim is not None:
            inpembdim = gdim
            outembdim = gdim

    outlindim = innerdim * 2  # dimension of output layer - twice the encoding dim because cat of enc and state

    encdim = innerdim // 2  # half the dimension because bidirectional encoder doubles it back
    encdims = [inpembdim] + [encdim] * numlayers  # encoder's layers' dimensions
    decdims = [outembdim] + [innerdim] * numlayers  # decoder's layers' dimensions
    # endregion

    # region data
    ism, osm, cnsm, gwids, splits, e2cn = load_matrices()
    gwids._matrix = gwids.matrix * (gwids.matrix != gwids.D["<RARE>"])
    devstart, teststart = splits
    eids = np.arange(0, len(ism), dtype="int64")
    # splits
    if test:    devstart, teststart, batsize = 200, 250, 50
    datamats = [ism.matrix, osm.matrix, gwids.matrix, e2cn, eids]
    traindata = [datamats[i][:devstart] for i in [0, 1, 2, 3, 4]]
    devdata = [datamats[i][devstart:teststart] for i in [0, 1, 2, 3]]       # should be same as tf script
    testdata = [datamats[i][teststart:] for i in [0, 1, 2, 3]]              # should be same as tf script

    rev_osm_D = {v: k for k, v in osm.D.items()}
    rev_gwids_D = {v: k for k, v in gwids.D.items()}

    gdic = q.PretrainedWordEmb(gdim).D
    rare_gwids_after_glove = get_rare_stats(traindata[0], traindata[2], gwids.D, gdic)
    print("{} doing rare".format("NOT" if not dorare else ""))
    if not dorare:
        rare_gwids_after_glove = None
    cnsm = do_rare_in_colnames(cnsm, traindata[-1], gdic, replace=dorare)

    # oracle:
    tracker = make_tracker_df(osm)
    oracle = make_oracle_df(tracker, mode=oraclemode)
    # endregion

    # region submodules     # exactly the same as tf script
    inpemb, inpbaseemb = make_inp_emb(inpembdim, ism, gwids, useglove=useglove, gdim=gdim, gfrac=gfrac, rare_gwids=rare_gwids_after_glove)
    outemb, inpbaseemb, colbaseemb, _ = make_out_emb(outembdim, osm, gwids, cnsm, gdim=gdim,
                                                  inpbaseemb=inpbaseemb, useglove=useglove, gfrac=gfrac, rare_gwids=rare_gwids_after_glove)
    if not tieembeddings:
        inpbaseemb, colbaseemb = None, None

    outlin, inpbaseemb, colbaseemb, colenc = make_out_lin(outlindim, ism, osm, gwids, cnsm,
                                                          useglove=useglove, gdim=gdim, gfrac=gfrac,
                                                          inpbaseemb=inpbaseemb, colbaseemb=colbaseemb,
                                                          colenc=None, nocopy=ablatecopy, rare_gwids=rare_gwids_after_glove)

    encoder = q.FastestLSTMEncoder(*encdims, dropout_in=idropout, dropout_rec=irdropout, bidir=True)
    # endregion

    # region encoder decoder definitions
    # -- changes from TF script: added maxtime on class itself, forward additionally takes eids and maxtime
    class EncDec(torch.nn.Module):
        def __init__(self, dec, maxtime=None):
            super(EncDec, self).__init__()
            self.inpemb, self.outemb, self.outlin, self.encoder, self.decoder \
                = inpemb, outemb, outlin, encoder, dec
            self.maxtime = maxtime

        def forward(self, inpseq, outseq_starts, inpseqmaps, colnames, eids=None, maxtime=None):
            maxtime = self.maxtime if maxtime is None else maxtime
            # encoding
            self.inpemb.prepare(inpseqmaps)
            _inpembs, _inpmask = self.inpemb(inpseq)
            _inpenc = self.encoder(_inpembs, mask=_inpmask)
            inpmask = _inpmask[:, :_inpenc.size(1)]
            inpenc = q.intercat(_inpenc.chunk(2, -1), -1)
            ctx = inpenc  # old normalpointer mode

            # decoding
            self.outemb.prepare(inpseqmaps, colnames)
            self.outlin.prepare(inpseq, inpenc, inpseqmaps, colnames)

            decoding = self.decoder(outseq_starts, ctx=ctx, ctx_0=ctx[:, -1, :],
                                    ctxmask=inpmask, eids=eids, maxtime=maxtime)

            return decoding
    # endregion

    # region decoders and model, for train and test
    layers = [q.LSTMCell(decdims[i - 1], decdims[i], dropout_in=dropout, dropout_rec=rdropout)
              for i in range(1, len(decdims))]
    decoder_top = q.AttentionContextDecoderTop(q.Attention().dot_gen(),
                                               q.Dropout(edropout),
                                               outlin, ctx2out=False)
    decoder_core = q.DecoderCore(outemb, *layers)
    decoder_cell = q.ModularDecoderCell(decoder_core, decoder_top)
    decoder_cell.set_runner(oracle)                 # change from TF script
    decoder = decoder_cell.to_decoder()

    m = EncDec(decoder)  # ONLY USE FOR TRAINING !!!

    valid_decoder_cell = q.ModularDecoderCell(decoder_core, decoder_top)
    valid_decoder_cell.set_runner(q.FreeRunner())
    valid_decoder = valid_decoder_cell.to_decoder()

    valid_m = EncDec(valid_decoder, maxtime=osm.matrix.shape[1]-1)  # use for valid -- change from TF script
                # change from original oracle script: added -1 to have same maxtime as TF script --> don't need out_bt
    # TODO: verify that valid_m doesn't get something wrong !
    # endregion

    # region training preparation
    trainloader = q.dataload(*traindata, batch_size=batsize, shuffle=True)
    validloader = q.dataload(*devdata, batch_size=batsize, shuffle=False)
    testloader = q.dataload(*testdata, batch_size=batsize, shuffle=False)

    losses = q.lossarray(q.SeqCrossEntropyLoss(ignore_index=0),
                         q.SeqAccuracy(ignore_index=0))

    row2tree = lambda x: SqlNode.parse_sql(osm.pp(x))

    validlosses = q.lossarray(q.SeqAccuracy(ignore_index=0),
                              TreeAccuracy(ignore_index=0, treeparser=row2tree))

    logger.update_settings(optimizer="adam")
    optim = torch.optim.Adam(q.paramgroups_of(m), lr=lr, weight_decay=wreg)

    def inp_bt(ismbatch, osmbatch, gwidsbatch, colnameids, eids):
        colnames = cnsm.matrix[colnameids.cpu().data.numpy()]
        colnames = q.var(colnames).cuda(colnameids).v
        return ismbatch, osmbatch[:, 0], gwidsbatch, colnames, eids, eids

    def valid_inp_bt(ismbatch, osmbatch, gwidsbatch, colnameids):
        colnames = cnsm.matrix[colnameids.cpu().data.numpy()]
        colnames = q.var(colnames).cuda(colnameids).v
        return ismbatch, osmbatch[:, 0], gwidsbatch, colnames, osmbatch[:, 1:]

    # old script had gold return whole seq but used valid_gold_bt to remove first element

    def out_bt(_out):
        return _out[:, :-1, :]

    def gold_bt(_eids):
        return torch.stack(oracle.goldacc, 1)

    # saving best model
    best_saver = BestSaver(lambda: validlosses.get_agg_errors()[1],
                           valid_m, path=model_save_path, verbose=True)
    # endregion

    # region training
    q.train(m).train_on(trainloader, losses) \
        .optimizer(optim).clip_grad_norm(gradnorm).set_batch_transformer(inp_bt, out_bt, gold_bt) \
        .valid_with(valid_m).valid_on(validloader, validlosses).set_valid_batch_transformer(valid_inp_bt, None, None) \
        .cuda(cuda).hook(logger).hook(best_saver) \
        .train(epochs)
    logger.update_settings(completed=True)
    # endregion

    # region evaluation
    tt.tick("Loading best model...")
    valid_m.load_state_dict(torch.load(model_save_path))

    dev_sql_acc, test_sql_acc = evaluate_model(valid_m, devdata, testdata, rev_osm_D, rev_gwids_D,
                                               inp_bt=valid_inp_bt, batsize=batsize, cuda=cuda,
                                               savedir=logger.p, test=test)
    # endregion

# endregion

# endregion


if __name__ == "__main__":
    # test_matrices()
    # test_querylin2json()
    # test_sqlnode_and_sqls()
    # test_grouptracker()
    # test_save()
    # q.argprun(run_seq2seq_tf)
    q.argprun(run_seq2seq_oracle_df)