import qelos as q
from qelos.scripts.webqa.load import load_all


def run(lr=0.1,
        glovedim=50,
        merge_mode="cat",        # "cat" or "sum"
        rel_which="urlwords",     # "urlwords ... ..."
        rel_embdim=-1,
        ):
    tt = q.ticktock("script")

    # load data and reps
    tt.tick("loading everything")
    rel_which = tuple(rel_which.split())
    rel_embdim = None if rel_embdim == -1 else rel_embdim
    (question_sm, query_sm, vnt_mat, tx_sep, qids), (src_emb, tgt_emb, tgt_lin) \
        = load_all(glovedim=glovedim, merge_mode=merge_mode,
                   rel_which=rel_which, rel_embdim=rel_embdim)
    tt.tock("loaded everything")

    # make main model

    # train main model


if __name__ == "__main__":
    q.argprun(run)