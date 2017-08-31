import qelos as q
import dill as pickle
import numpy as np
from qelos.scripts.webqa.preprocessing.loader import *


defaultp = "../../../../datasets/webqsp/flmats/"
defaultqp = "../../../../datasets/webqsp/webqsp"


def run(p=defaultp, qp=defaultqp):
    dicts, entity_info, relation_info = load_info_mats(p)
    train, test = load_questions(qp)
    glovewords_ = {}

    def loadglovewords(dim=300, trylowercase=True):
        if dim in glovewords_ and glovewords_[dim] is not None:
            return glovewords_[dim]
        p = "../../../../data/glove/glove.{}d.words".format(dim)
        tt = q.ticktock("glove loader")
        tt.tick("loading words {}D".format(dim))
        words = set(pickle.load(open(p)))
        if trylowercase:
            newwords = set()
            for word in words:
                if word.lower() not in words:
                    newwords.add(word.lower())
                else:
                    newwords.add(word)
            words = newwords
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

    def get_stats_train_test(glovedim=50):
        train_nl_words = set(train[0].D.keys())
        test_nl_words = set(test[0].D.keys())
        train_fl_words = set(train[1].D.keys())
        test_fl_words = set(test[1].D.keys())
        nl_unseen_words = test_nl_words - train_nl_words
        fl_unseen_words = test_fl_words - train_fl_words
        # train frequencies for words in test that were seen during training
        nl_overlap = train_nl_words & test_nl_words
        fl_overlap = train_fl_words & test_fl_words
        print("\n OVERLAP STATS \n")
        nl_vrl_sorted = sorted(nl_overlap, key=lambda x: train[0]._wordcounts_original[x], reverse=True)
        for nl_vrl_sorted_e in nl_vrl_sorted:
            print(nl_vrl_sorted_e, train[0]._wordcounts_original[nl_vrl_sorted_e])
        print("-")
        fl_vrl_sorted = sorted(fl_overlap, key=lambda x: train[1]._wordcounts_original[x], reverse=True)
        for fl_vrl_sorted_e in fl_vrl_sorted:
            print(fl_vrl_sorted_e, train[1]._wordcounts_original[fl_vrl_sorted_e])
        print("nl overlap: {}".format(len(nl_overlap), len(test_nl_words)))
        print("fl overlap: {}".format(len(fl_overlap), len(test_fl_words)))

        print("\n UNSEEN STATS \n")
        nl_unseen_sorted = sorted(nl_unseen_words, key=lambda x: test[0]._wordcounts_original[x], reverse=True)
        for nl_vrl_sorted_e in nl_unseen_sorted:
            print(nl_vrl_sorted_e, test[0]._wordcounts_original[nl_vrl_sorted_e])
        print("-")
        fl_unseen_sorted = sorted(fl_unseen_words, key=lambda x: test[1]._wordcounts_original[x], reverse=True)
        for fl_vrl_sorted_e in fl_unseen_sorted:
            print(fl_vrl_sorted_e, test[1]._wordcounts_original[fl_vrl_sorted_e])

        print("{}/{} nl words in test not seen during training".format(len(nl_unseen_words), len(set(test[0].D.keys()))))
        print("{}/{} fl words in test not seen during training".format(len(fl_unseen_words), len(set(test[1].D.keys()))))

        print("\n GLOVE COVERAGE OF UNSEEN TEST WORDS \n")
        glovewords = loadglovewords(glovedim)
        glove_unc_test = nl_unseen_words - glovewords
        print("{}/{} unseen test words not covered by Glove".format(len(glove_unc_test), len(nl_unseen_words)))

        print("\n GLOVE COVERAGE OF ALL WORDS \n")
        glovewords = loadglovewords(glovedim)
        print("{}/{} train words not covered by Glove".format(len(train_nl_words - glovewords), len(train_nl_words)))
        print("{}/{} test words not covered by Glove".format(len(test_nl_words - glovewords), len(test_nl_words)))
        print(train_nl_words - glovewords)
        print(test_nl_words - glovewords)

        return nl_unseen_words, fl_unseen_words

    q.embed()


if __name__ == "__main__":
    q.argprun(run)