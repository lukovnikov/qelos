import qelos as q
import dill as pickle
import numpy as np


defaultp = "../../../../datasets/webqsp/flmats/"
defaultqp = "../../../../datasets/webqsp/webqsp"


def run(p=defaultp, qp=defaultqp):
    dicts, entity_info, relation_info = load_info_mats(p)
    train, test = load_questions(qp)
    glovewords_ = {}

    def loadglovewords(dim=300):
        if dim in glovewords_ and glovewords_[dim] is not None:
            return glovewords_[dim]
        p = "../../../../data/glove/glove.{}d.words".format(dim)
        tt = q.ticktock("glove loader")
        tt.tick("loading words {}D".format(dim))
        words = set(pickle.load(open(p)))
        tt.tock("{} words loaded".format(len(words)))
        glovewords_[dim] = words
        return words

    def print_relinfo(relid):
        print(
    """{}
    name: {}
    url_words: {}
    url_tokens: {}
    domain words: {}
    domain_token: {}
    range words: {}
    range_token: {}""".format(relid,
               relation_info[0].pp(relation_info[0].matrix[dicts[1][relid]]),
               relation_info[1].pp(relation_info[1].matrix[dicts[1][relid]]),
               relation_info[2].pp(relation_info[2].matrix[dicts[1][relid]]),
               relation_info[3].pp(relation_info[3].matrix[dicts[1][relid]]),
               relation_info[5].pp(relation_info[5].matrix[dicts[1][relid]]),
               relation_info[4].pp(relation_info[4].matrix[dicts[1][relid]]),
               relation_info[6].pp(relation_info[6].matrix[dicts[1][relid]])
               ))
        pass

    def matstats(relmatsm, glovewords=None):
        ret = {
            "maximum_length": relmatsm.matrix.shape[1],
            "vocabulary_size": len(set(relmatsm.D.keys())),
            "volume": "{}/{}".format(np.sum(relmatsm.matrix != 0), relmatsm.matrix.shape[0] * relmatsm.matrix.shape[1]),
            "glove_coverage": "{}/{}".format(0 if glovewords is None else len(glovewords & set(relmatsm.D.keys())), len(set(relmatsm.D.keys()))),
        }
        return ret

    def print_allrelmatstats(glovedim=300):
        glovewords = loadglovewords(glovedim)
        for name, relmatsm in zip("names urlwords urltokens domainwords rangewords domainids rangeids".split(),
                                  relation_info):
            stats = matstats(relmatsm, glovewords=glovewords)
            print("{}\n{}".format(name, "\n".join(["\t{}:\t{}".format(x, y) for x, y in stats.items()])))

    def print_allentmatstats(glovedim=300):
        glovewords = loadglovewords(glovedim)
        for name, entmatsm in zip("names_word names_char notable_type_names types type_names".split(),
                                  entity_info):
            stats = matstats(entmatsm, glovewords=glovewords)
            print("{}\n{}".format(name, "\n".join(["\t{}:\t{}".format(x, y) for x, y in stats.items()])))

    def print_words_uncovered(matsm, glovedim=300):
        glovewords = loadglovewords(glovedim)
        diff = set(matsm.D.keys()) - glovewords
        diff = sorted(diff, key=lambda x: matsm._wordcounts_original[x], reverse=True)
        for word in diff:
            print("\t{}:\t {}".format(word, matsm._wordcounts_original[word]))
        print("\t{}/{} words not covered by glove".format(len(diff), len(set(matsm.D.keys()))))

    def print_words_uncovered_entmats(glovedim=300):
        for name, entmatsm in zip("names_word notable_type_names type_names".split(),
                                  [entity_info[0], entity_info[2], entity_info[4]]):
            print("{}".format(name))
            print_words_uncovered(entmatsm, glovedim=glovedim)

    def print_words_uncovered_relmats(glovedim=300):
        for name, relmatsm in zip("names urlwords domainwords rangewords".split(),
                                  [relation_info[x] for x in (0, 1, 3, 4)]):
            print("{}".format(name))
            print_words_uncovered(relmatsm, glovedim=glovedim)

    def get_rows_with_uncovered_words_from(matsm, glovedim=300):
        glovewords = loadglovewords(glovedim)
        diff = set(matsm.D.keys()) - glovewords - {"<START>", "<END>", "<RARE>", "<MASK>"}
        diffids = {matsm.D[x] for x in diff}
        diffmat = np.vectorize(lambda x: 1 if x in diffids else 0)(matsm.matrix)
        diffmat = np.sum(diffmat, axis=1)
        rowswithuncoveredwords = np.argwhere(diffmat > 0)
        return rowswithuncoveredwords

    def get_used_tokens(querysm, d):
        ret = {d[x] for x in querysm.D.keys() if x in d}
        return ret

    q.embed()


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



if __name__ == "__main__":
    q.argprun(run)