import dill as pickle
import qelos as q
import numpy as np
import scipy.sparse as sparse


defaultp = "../../../../datasets/webqsp/flmats/"
defaultqp = "../../../../datasets/webqsp/webqsp"


def load_info_mats(p=defaultp):
    tt = q.ticktock("loader")
    tt.tick()
    entity_dict = pickle.load(open(p+"webqsp.entity.dic"))
    relation_dict = pickle.load(open(p+"webqsp.relation.dic"))
    entity_names_char = pickle.load(open(p+"webqsp.entity.names.char.sm"))
    entity_names_word = pickle.load(open(p+"webqsp.entity.names.sm"))
    entity_notabletypes = pickle.load(open(p+"webqsp.entity.notabletypes.sm"))
    entity_types = pickle.load(open(p+"webqsp.entity.types.sm"))
    entity_typenames = pickle.load(open(p+"webqsp.entity.typenames.sm"))
    relation_names = pickle.load(open(p+"webqsp.relation.names.sm"))
    relation_domainids = pickle.load(open(p+"webqsp.relation.domainids.sm"))
    relation_rangeids = pickle.load(open(p+"webqsp.relation.rangeids.sm"))
    relation_domains = pickle.load(open(p+"webqsp.relation.domains.sm"))
    relation_ranges = pickle.load(open(p+"webqsp.relation.ranges.sm"))
    relation_urltokens = pickle.load(open(p+"webqsp.relation.urltokens.sm"))
    relation_urlwords = pickle.load(open(p+"webqsp.relation.urlwords.sm"))
    tt.tock("loaded everything")
    return (entity_dict, relation_dict), \
           (entity_names_word, entity_names_char, entity_notabletypes, entity_types, entity_typenames), \
           (relation_names, relation_urlwords, relation_urltokens, relation_domains, relation_ranges, relation_domainids, relation_rangeids)


def load_entity_info_mats(p=defaultp):
    tt = q.ticktock("loader")
    tt.tick()
    entity_dict = pickle.load(open(p+"webqsp.entity.dic"))
    entity_names_char = pickle.load(open(p+"webqsp.entity.names.char.sm"))
    entity_names_word = pickle.load(open(p+"webqsp.entity.names.sm"))
    entity_notabletypes = pickle.load(open(p+"webqsp.entity.notabletypes.sm"))
    entity_types = pickle.load(open(p+"webqsp.entity.types.sm"))
    entity_typenames = pickle.load(open(p+"webqsp.entity.typenames.sm"))
    tt.tock("loaded entity info")
    return entity_dict, q.dtoo(
                        {"names_word": entity_names_word,
                         "names_char": entity_names_char,
                         "notabletypes_word": entity_notabletypes,
                         "types_ids": entity_types,
                         "types_word": entity_typenames})


def load_relation_info_mats(p=defaultp):
    tt = q.ticktock("loader")
    tt.tick()
    relation_dict = pickle.load(open(p+"webqsp.relation.dic"))
    relation_names = pickle.load(open(p+"webqsp.relation.names.sm"))
    relation_domainids = pickle.load(open(p+"webqsp.relation.domainids.sm"))
    relation_rangeids = pickle.load(open(p+"webqsp.relation.rangeids.sm"))
    relation_domains = pickle.load(open(p+"webqsp.relation.domains.sm"))
    relation_ranges = pickle.load(open(p+"webqsp.relation.ranges.sm"))
    relation_urltokens = pickle.load(open(p+"webqsp.relation.urltokens.sm"))
    relation_urlwords = pickle.load(open(p+"webqsp.relation.urlwords.sm"))
    tt.tock("loaded relation info")
    return relation_dict, q.dtoo({"names": relation_names,
                                  "urlwords": relation_urlwords,
                                  "urltokens": relation_urltokens,
                                  "domainwords": relation_domains,
                                  "rangewords": relation_ranges,
                                  "domainids": relation_domainids,
                                  "rangeids": relation_rangeids})


def load_questions(p=defaultqp):
    tt = q.ticktock("question loader")
    tt.tick("loading questions")
    questions, queries = q.StringMatrix(), q.StringMatrix()
    xquestions, xqueries = q.StringMatrix(), q.StringMatrix()

    queries.tokenize = lambda x: x.split()
    xqueries.tokenize = lambda x: x.split()

    with open(p+".train.butd") as f:
        for line in f:
            qid, question, query, replacements = line.split("\t")
            questions.add(question)
            queries.add(query)

    questions.finalize()
    queries.finalize()

    with open(p+".test.butd") as f:
        for line in f:
            qid, question, query, replacements = line.split("\t")
            xquestions.add(question)
            xqueries.add(query)

    xquestions.finalize()
    xqueries.finalize()
    tt.tock("loaded questions")
    return (questions, queries), (xquestions, xqueries)


def load_questions_inone(p=defaultqp):
    tt = q.ticktock("question loader")
    tt.tick("loading questions")
    questions, queries = q.StringMatrix(), q.StringMatrix()
    qids = []
    questions.tokenize = lambda x: x.split()
    queries.tokenize = lambda x: x.split()

    with open(p+".train.butd") as f:
        for line in f:
            qid, question, query, replacements = line.split("\t")
            questions.add(question)
            queries.add(query)
            qids.append(qid)

    tx_sep = len(qids)

    with open(p+".test.butd") as f:
        for line in f:
            qid, question, query, replacements = line.split("\t")
            questions.add(question)
            queries.add(query)
            qids.append(qid)

    questions.finalize()
    queries.finalize()

    return questions, queries, qids, tx_sep


def load_vnt_mats(qids=None, p=defaultqp, tgtdict=None):   # tgtdict: get from makereps.py get_all_reps() tgt_emb or tgt_lin
    tt = q.ticktock("vnt loader")
    tt.tick("loading vnts")
    trainvntp = p+".train.butd.vnt"
    testvntp = p+".test.butd.vnt"
    trainvnt = pickle.load(open(trainvntp))
    print(len(trainvnt))
    testvnt = pickle.load(open(testvntp))
    print(len(testvnt))
    tt.tock("loaded vnts")
    tt.tick("making vnt mat")
    vntmat = get_vnt_mat(qids, tgtdict, [trainvnt, testvnt])
    tt.tock("made vnt mat")
    return vntmat


def get_vnt_mat(qids, tgtdict, vnts):
    vnt = {}
    for vnte in vnts:
        vnt.update(vnte)

    maxlen = 0
    for example_vnt in vnt:
        maxlen = max(maxlen, len(vnt[example_vnt]))

    # vntmat = sparse.lil_matrix((len(qids) * (maxlen+1), max(tgtdict.values())+1), dtype="int32")
    vntmat = np.zeros((len(qids), (maxlen + 1), max(tgtdict.values()) + 1), dtype="int32")
    for i, qid in enumerate(qids):
        for j, timestep_vnt in enumerate(vnt[qid]):
            for timestep_vnt_element in timestep_vnt:
                k = tgtdict[timestep_vnt_element]
                # print(i, j, k)
                vntmat[i, j, k] = 1
                # vntmat[i * (maxlen + 1) + j, k] = 1
    # return vntmat, (len(qids), maxlen+1, vntmat.shape[1])

    return vntmat, vntmat.shape

