from qelos.scripts.webqa.makereps import get_all_reps
from qelos.scripts.webqa.preprocessing.loader import load_vnt_mats, load_questions_inone
import qelos as q
import os.path
import dill as pickle
import scipy.sparse as sparse
import numpy as np


defaultp = "../../../datasets/webqsp/flmats/"
defaultqp = "../../../datasets/webqsp/webqsp"
vntcachep = "wholevnt.mat.cache"


def load_full(p=defaultp, qp=defaultqp, dim=50, glovedim=50,
             merge_mode="cat", rel_which=("urlwords",)):
    tt = q.ticktock("load")
    tt.tick("loading everything")
    question_sm, query_sm, qids, tx_sep = load_questions_inone(p=qp)
    src_emb, tgt_emb, tgt_lin = get_all_reps(dim=dim, glovedim=glovedim,
                                             merge_mode=merge_mode, rel_which=rel_which,
                                             question_sm=question_sm, query_sm=query_sm)

    vnt_mat, vnt_mat_shape = load_vnt_mats(qids=qids, p=qp, tgtdict=tgt_emb.D)
    print(vnt_mat.nbytes)
    # print(np.sum(vnt_mat == 1), np.sum(vnt_mat == 0))

    tt.tock("loaded everything")
    # vnt_mat = vnt_mat.todense()
    # print(vnt_mat.shape)
    # vnt_mat = vnt_mat.reshape(vnt_mat_shape)
    print(vnt_mat.shape)
    assert(len(vnt_mat) == len(question_sm.matrix))
    print("vnt mat has same length as question mat")

    # check real next token in vnt
    tt.tick("checking loaded vnts")
    for i in range(query_sm.matrix.shape[0]):
        for j in range(query_sm.matrix.shape[1]):
            if query_sm.matrix[i, j] != query_sm.D["<MASK>"]:
                next_symbol = query_sm.matrix[i, j]
                assert(vnt_mat[i, j, next_symbol] == 1)

    tt.tock("checked loaded vnts")
    return (question_sm, query_sm, vnt_mat, tx_sep, qids), (src_emb, tgt_emb, tgt_lin)


def load_both():
    pass
    # TODO: merge core-only and full examples and their vnts
    # TODO: are all vnt for core chains in flmats?
        # --> default arguments of getflrepdata.py make it seem so --> TODO test !!!
    #


if __name__ == "__main__":
    q.argprun(load_full)