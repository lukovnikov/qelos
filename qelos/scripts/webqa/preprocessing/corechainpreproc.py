import qelos as q
from qelos.scripts.webqa.preprocessing.getflrepdata import Querier
from qelos.scripts.webqa.preprocessing.buildvnt import RelDicAcc
import dill as pickle
from collections import OrderedDict
import numpy as np

# TAKE webqsp.webqsp.all.chains
# PRODUCE question stringmatrix and stringmatrices for relation info --> put in corechain.lexmats


def run(p="../../../../datasets/webqsp/webqsp.webqsp.all.chains",
        outp="../../../../datasets/webqsp/webqsp.chains.all"):
    tt = q.ticktock("corechainpreproc")
    data = pickle.load(open(p))
    reldic = data["reldic"]
    parses = data["parses"]

    qids = {}
    for qpid, parsedata in parses.items():
        qid = parsedata["qid"]
        qids[qid] = set()
        qids[qid].add(qpid)

    # region build question_sm
    question_sm = q.StringMatrix()
    question_sm.tokenize = lambda x: x.split()

    qid2qpid = {}

    qid2tx = {}
    qid2qsm = {}
    i = 0
    for qid, qpids in qids.items():
        # first parse only
        chosenqpid = list(qpids)[0]
        parsedata = parses[chosenqpid]
        qid2qpid[qid] = chosenqpid
        questiontext = parsedata["question"]
        topicmention = parsedata["topicmention"]
        questiontext = questiontext.replace(topicmention, "<E0>")
        question_sm.add(questiontext)
        qid2tx[qid] = parsedata["tx"]
        qid2qsm[qid] = i
        i += 1

    question_sm.finalize()
    # endregion

    # region getting relation info
    batsize = 100
    que = Querier()
    collectedrels = set(reldic.reldic.keys())
    collectedrels -= {"<NONE>"}
    print("{} collected rels".format(len(collectedrels)))
    relations = list(collectedrels)
    relations = sorted(relations, key=lambda x: reldic.reldic[x])
    relationinfo = OrderedDict()
    i = 0
    while i < len(relations):
        j = min(i+batsize, len(relations))
        batch = relations[i:j]
        names = que.get_entity_property(batch, "type.object.name", language="en")
        domains = que.get_entity_property(batch, "type.property.schema")
        domainsnames = que.get_entity_property(batch, "type.property.schema type.object.name", language="en")
        ranges = que.get_entity_property(batch, "type.property.expected_type")
        rangesnames = que.get_entity_property(batch, "type.property.expected_type type.object.name", language="en")
        for x in batch:
            x = x[1:]
            relationinfo[":"+x] = {"name": names[x] if x in names else None,
                                   "domain": domains[x] if x in domains else None,
                                   "range": ranges[x] if x in ranges else None,
                                   "domainname": domainsnames[x] if x in domainsnames else None,
                                   "rangename": rangesnames[x] if x in rangesnames else None,}
        tt.msg("batch {}-{}".format(i, j))
        i = j
    tt.tock("relation info got")

    rdic, names, domains, ranges, domainids, rangeids, \
    urlwords, urltokens \
        = build_relation_matrices(relationinfo)

    for reluri, relid in reldic.reldic.items():
        if reluri != "<NONE>":
            assert(relid == rdic[reluri] + 1)

    lexmats = {"names": names, "domains": domains, "ranges": ranges,
               "domainids": domainids, "rangeids": rangeids,
               "urlwords": urlwords, "urltokens": urltokens, "dic": rdic}
    # RDIC's ID for each relation uri is -1 the id in original reldic
    # endregion

    # region making corechain mat
    uniquechains = OrderedDict()    # maps unique chains to ids
    qpid2uchain = {}                # maps qpid to correct chain
    qpid2badchains = {}             # maps qpid to bad chains
    for qpid, parsedata in parses.items():
        correctchain = parsedata["corechain"]
        otherchains = parsedata["negchains"]
        if correctchain not in uniquechains:
            uniquechains[correctchain] = len(uniquechains)
        correctchainid = uniquechains[correctchain]
        qpid2uchain[qpid] = correctchainid
        for otherchain in otherchains:
            if otherchain not in uniquechains:
                uniquechains[otherchain] = len(uniquechains)
            otherchainid = uniquechains[otherchain]
            if qpid not in qpid2badchains:
                qpid2badchains[qpid] = set()
            qpid2badchains[qpid].add(otherchainid)

    uniquechainsmat = np.zeros((len(uniquechains), 2), dtype="int64")
    i = 0
    for uniquechain, _ in uniquechains.items():
        uniquechainsmat[i, :len(uniquechain)] = [chainrel for chainrel in uniquechain]
        i += 1

    # REL IDS in uniquechainsmat are +1 the ids from rdic (0 id is mask)
    qid2poschain = {}
    qid2negchains = {}
    for qid, qidqpid in qids.items():
        qid2poschain[qid] = qpid2uchain[qid2qpid[qid]]
        qid2negchains[qid] = qpid2badchains[qid2qpid[qid]]
    # endregion

    # region saving
    # save question_sm, lexmats, uniquechainsmat, qpid2uchain, qpid2badchains
    outdata = {"question_sm": question_sm,      # question stringmatrix
               "qid2qsm": qid2qsm,
               "lexmats": lexmats,              # lexmats (see above)
               "uniquechains": uniquechainsmat, # unique chains matrix, rel ids are +1 from rdic
               "poschain": qid2poschain,        # maps qid to row of good chain
               "negchains": qid2negchains,      # maps qid to rows of bad chains
               "txdic": qid2tx,
               "rdic": rdic,
               "reldic": reldic}

    pickle.dump(outdata, open(outp, "w"))









def build_relation_matrices(info):
    tt = q.ticktock("relation matrix builder")
    tt.tick("building")

    # build
    ids = []
    names = q.StringMatrix()
    names.tokenize = lambda x: q.tokenize(x) if x != "<RARE>" else [x]
    domains = q.StringMatrix()
    domains.tokenize = lambda x: q.tokenize(x) if x != "<RARE>" else [x]
    ranges = q.StringMatrix()
    ranges.tokenize = lambda x: q.tokenize(x) if x != "<RARE>" else [x]
    urlwords = q.StringMatrix()
    urlwords.tokenize = lambda x: q.tokenize(x, preserve_patterns=['<[A-Z]+>']) if x != "<RARE>" else [x]
    urltokens = q.StringMatrix()
    urltokens.tokenize = lambda x: x
    domainids = q.StringMatrix()
    domainids.tokenize = lambda x: x
    rangeids = q.StringMatrix()
    rangeids.tokenize = lambda x: x

    for key, val in info.items():
        ids.append(key)
        name = list(val["name"])[0] if val["name"] is not None else "<RARE>"
        names.add(name)
        domain = list(val["domainname"])[0] if val["domainname"] is not None else "<RARE>"
        domains.add(domain)
        rangename = list(val["rangename"])[0] if val["rangename"] is not None else "<RARE>"
        ranges.add(rangename)
        rangeid = list(val["range"]) if val["range"] is not None else ["<RARE>"]
        rangeids.add(rangeid)
        domainid = list(val["domain"]) if val["domain"] is not None else ["<RARE>"]
        domainids.add(domainid)
        splits = key[1:].split(".")
        if splits[0] == "user":
            splits = splits[2:]
        while len(splits) < 3:
            splits = ["default"] + splits
        url = ".".join(splits)
        urlword = " <SEP> ".join(splits)
        urlwords.add(urlword)
        urltoken = [".".join(splits[:-2]),
                    splits[-2],
                    splits[-1]]
        urltokens.add(urltoken)
    tt.tock("built")

    tt.tick("finalizing")
    names.finalize()
    domains.finalize()
    ranges.finalize()
    rangeids.finalize()
    domainids.finalize()
    urlwords.finalize()
    urltokens.finalize()
    tt.tock("finalized")

    rdic = dict(zip(ids, range(len(ids))))

    return rdic, names, domains, ranges, domainids, rangeids, urlwords, urltokens




if __name__ == "__main__":
    q.argprun(run)