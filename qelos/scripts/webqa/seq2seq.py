import qelos as q
from qelos.scripts.webqa.load import load_all
import torch


def run(lr=0.1,
        gradnorm=2.,
        epochs=100,
        wreg=1e-6,
        glovedim=50,
        merge_mode="cat",        # "cat" or "sum"
        rel_which="urlwords",     # "urlwords ... ..."
        rel_embdim=-1,
        batsize=128,
        cuda=False,
        gpu=1,
        ):
    if cuda:
        torch.cuda.set_device(gpu)

    tt = q.ticktock("script")
    # region I. data
    # load data and reps
    tt.tick("loading data and rep")
    rel_which = tuple(rel_which.split())
    rel_embdim = None if rel_embdim == -1 else rel_embdim
    (question_sm, query_sm, vnt_mat, tx_sep, qids), (src_emb, tgt_emb, tgt_lin) \
        = load_all(glovedim=glovedim, merge_mode=merge_mode,
                   rel_which=rel_which, rel_embdim=rel_embdim)
    tt.tock("loaded data and rep")
    tt.tick("making data loaders")
    # train/valid/test split:
    train_questions, test_questions = question_sm.matrix[:tx_sep], question_sm[tx_sep:]
    train_queries, test_queries = query_sm.matrix[:tx_sep], query_sm[tx_sep:]
    train_vnt, test_vnt = vnt_mat[:tx_sep], vnt_mat[tx_sep:]

    validportion = .2
    tv_sep = int(len(train_questions) * (1 - validportion))
    train_questions, valid_questions = train_questions[:tv_sep], train_questions[tv_sep:]
    train_queries, valid_queries = train_queries[:tv_sep], train_queries[tv_sep:]
    train_vnt, valid_vnt = train_vnt[:tv_sep], train_vnt[tv_sep:]

    # make data loaders
    train_dataloader = q.dataload(train_questions, train_vnt, train_queries, batch_size=batsize)
    valid_dataloader = q.dataload(valid_questions, valid_vnt, valid_queries, batch_size=batsize)
    test_dataloader = q.dataload(test_questions, test_vnt, test_queries, batch_size=batsize)
    tt.tock("made data loaders")
    # endregion

    # make main model
    m = make_model(src_emb, tgt_emb, tgt_lin, **model_options)

    # training settings
    losses = q.lossarray(q.SeqNLLLoss(), q.SeqAccuracy())
    validlosses = q.lossarray(q.SeqNLLLoss(), q.SeqAccuracy())

    optimizer = torch.optim.Adadelta(m.parameters(), lr=lr, weight_decay=wreg)

    # train
    q.train(m).cuda(cuda).train_on(train_dataloader, losses)\
        .set_batch_transformer(lambda a, b, c: (a, b, c[:, :-1], c[:, 1:]))\
        .valid_on(valid_dataloader, validlosses)\
        .optimizer(optimizer).clip_grad_norm(gradnorm)\
        .train(epochs)

    # test


if __name__ == "__main__":
    q.argprun(run)