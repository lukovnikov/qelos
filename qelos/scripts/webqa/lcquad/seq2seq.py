from __future__ import print_function
import qelos as q
from qelos.scripts.webqa.lcquad.makereps import load_all
import torch
from torch import nn
import sys
import numpy as np
from collections import OrderedDict
import sparkline
import pickle

from qelos.scripts.webqa.seq2seq_shared import make_encoder, make_decoder, \
    test_model, ErrorAnalyzer, Model, allgiven_adjust_vnt, get_corechain


def run(lr=0.1,
        gradnorm=2.,
        epochs=100,
        wreg=1e-6,
        dropout=0.1,
        glovedim=50,
        encdim=100,
        decdim=100,
        decsplit=True,
        attmode="bilin",        # "bilin" or "fwd"
        merge_mode="sum",        # "cat" or "sum"
        rel_which="urlwords",     # "urlwords ... ..."
        celltype="normal",          # "normal", "tree"
        batsize=2,
        cuda=False,
        gpu=1,
        inspectdata=False,
        log=True,
        erroranalysis=True,
        allgiven=False,          # assume all entities and relations needed are linked (and not more) --> filter vnt
        entityfilter=False,
        onlycorechain=False,
        ):
    localvars = locals()
    savesettings = "celltype allgiven entityfilter onlycorechain glovedim encdim decdim attmode gradnorm dropout merge_mode batsize epochs rel_which decsplit".split()
    savesettings = OrderedDict({k: localvars[k] for k in savesettings})
    if cuda:
        torch.cuda.set_device(gpu)

    tt = q.ticktock("script")
    # region I. data
    # load data and reps
    tt.tick("loading data and rep")

    flvecdim = 0
    if decsplit:    flvecdim = decdim // 2 + encdim // 2
    else:           flvecdim = decdim + encdim

    # LC-QuaD loading
    (qids, question_sm, query_sm, vnt_mat), (trainids, testids), (src_emb, tgt_emb, tgt_lin) \
        = load_all(dim=flvecdim, glovedim=glovedim, mergemode=merge_mode)
    vntvocsize = vnt_mat[0, 0, 1]
    # end

    vnt_mat = vnt_mat[:, :query_sm.matrix.shape[1], :]

    # test tgt_lin
    testlinx = q.var(np.random.random((3, flvecdim)).astype("float32")).v
    testlinvnt = np.zeros((3, vntvocsize), dtype="int64")
    testlinvntx = np.asarray([0, 0, 0,  1, 1,   1, 2, 2,  2,   2])
    testlinvnty = np.asarray([0, 1, 21, 0, 1, 201, 0, 1, 21, 201])
    testlinvnt[testlinvntx, testlinvnty] = 1
    testliny = tgt_lin(testlinx, q.var(testlinvnt).v)

    tt.tock("loaded data and rep")
    tt.tick("making data loaders")
    # train/test split:
    train_questions, test_questions = question_sm.matrix[trainids], question_sm.matrix[testids]
    train_queries, test_queries = query_sm.matrix[trainids], query_sm.matrix[testids]
    train_vnt, test_vnt = vnt_mat[trainids], vnt_mat[testids]
    train_qids, test_qids = [qids[trainid] for trainid in trainids], \
                            [qids[testid] for testid in testids]
    print(len(test_qids))
    # q.embed()

    if entityfilter:
        pass        # TODO

    if onlycorechain:
        train_queries, train_vnt = get_corechain(mode="webqsp")(train_queries, train_vnt, query_sm.D)
        test_queries, test_vnt = get_corechain(mode="webqsp")(test_queries, test_vnt, query_sm.D)
        pass        # TODO

    if allgiven:
        tt.msg("using allgiven")
        train_vnt = allgiven_adjust_vnt(mode="webqsp")(train_queries, train_vnt, query_sm.D)
        test_vnt = allgiven_adjust_vnt(mode="webqsp")(test_queries, test_vnt, query_sm.D)

    # train/valid split
    (train_questions, train_queries, train_vnt), (valid_questions, valid_queries, valid_vnt) \
        = q.split([train_questions, train_queries, train_vnt], splits=(80, 20), random=True)

    for k, v in query_sm.D.items():
        assert(tgt_emb.D[k] == v)
    tt.msg("tgt_emb uses the same ids as query_sm")

    if inspectdata:
        def pp(i, which="train"):
            questionsmat = train_questions if which=="train" else valid_questions if which == "valid" else test_questions
            queriesmat = train_queries if which=="train" else valid_queries if which == "valid" else test_queries
            question = question_sm.pp(questionsmat[i])
            query = query_sm.pp(queriesmat[i])
            print("{}\n{}".format(question, query))

        tgt_emb(q.var(np.asarray([[0, 1, 21, 201]])).v)

        q.embed()


    # make data loaders
    train_dataloader = q.dataload(train_questions, train_vnt, train_queries,
                                  batch_size=batsize, shuffle=True)
    valid_dataloader = q.dataload(valid_questions, valid_vnt, valid_queries,
                                  batch_size=batsize, shuffle=False)
    test_dataloader = q.dataload(test_questions, test_vnt, test_queries,
                                 batch_size=batsize, shuffle=False)
    tt.tock("made data loaders")
    # endregion

    # make main model
    src_emb_dim = glovedim
    tgt_emb_dim = flvecdim
    encoder = make_encoder(src_emb, embdim=src_emb_dim, dim=encdim, dropout=dropout)
    ctxdim = encdim
    decoder = make_decoder(mode="lcquad")(tgt_emb, tgt_lin, ctxdim=ctxdim,
                           embdim=tgt_emb_dim, dim=decdim,
                           attmode=attmode, decsplit=decsplit,
                           celltype=celltype)
    m = Model(encoder, decoder)

    # test_model(encoder, decoder, m, test_questions, test_queries, test_vnt)

    # training settings
    lt = lambda a: (a[0], {"mask": a[2]})       # need mask because cross entropy loss
    # lt = lambda a: (a[0], {})
    losses = q.lossarray((q.SeqCrossEntropyLoss(ignore_index=0), lt),
                         (q.SeqAccuracy(ignore_index=0), lt),
                         (q.SeqElemAccuracy(ignore_index=0), lt))
    validlosses = q.lossarray((q.SeqCrossEntropyLoss(ignore_index=0), lt),
                              (q.SeqAccuracy(ignore_index=0), lt),
                              (q.SeqElemAccuracy(ignore_index=0), lt))
    testlosses = q.lossarray((q.SeqCrossEntropyLoss(ignore_index=0), lt),
                             (q.SeqAccuracy(ignore_index=0), lt),
                             (q.SeqElemAccuracy(ignore_index=0), lt))

    optimizer = torch.optim.Adadelta(q.params_of(m), lr=lr, weight_decay=wreg)

    # sys.exit()

    bt = lambda a, b, c: (a, c[:, :-1], b[:, 1:], c[:, 1:])

    if erroranalysis and False:
        # TODO remove from here
        tt.msg("error analysis")
        erranal = ErrorAnalyzer(question_sm.D, tgt_emb.D)
        anal_losses = q.lossarray((erranal, lambda x: x))
        q.test(m).cuda(cuda)\
            .on(valid_dataloader, anal_losses)\
            .set_batch_transformer(bt)\
            .run()
        print(erranal.summary())
        erranal.inspect()

    # train
    q.train(m).cuda(cuda).train_on(train_dataloader, losses)\
        .set_batch_transformer(bt)\
        .valid_on(valid_dataloader, validlosses)\
        .optimizer(optimizer).clip_grad_norm(gradnorm)\
        .clip_grad_norm(gradnorm)\
        .train(epochs)

    # test
    nll, seqacc, elemacc \
        = q.test(m).cuda(cuda)\
        .on(test_dataloader, testlosses)\
        .set_batch_transformer(bt).run()

    tt.msg("Test results:")
    tt.msg("NLL:\t{}\n Seq Accuracy:\t{}\n Elem Accuracy:\t{}"
          .format(nll, seqacc, elemacc))

    # tt.msg("getting EL- and data-corrected seq accuracy")
    # welllinked = pickle.load(open("../../../datasets/webqsp/welllinked_qids.pkl"))
    # ebt = lambda a, b, c: (a, c[:, :-1], b[:, 1:])
    # argmaxer = q.Lambda(lambda x: torch.max(x[0], 2)[1])
    # predictions = q.eval(m).cuda(cuda).on(test_dataloader)\
    #     .set_batch_transformer(ebt, argmaxer).run()
    # predictions = predictions.cpu().data.numpy()
    #
    # totaltest = 1639.
    # same = predictions != test_queries[:, 1:]
    # mask = test_queries[:, 1:] != 0
    # same = same * mask
    # same = np.sum(same, axis=1)
    # acc = 0.
    # for diff_e, qid in zip(list(same), test_qids):
    #     if diff_e == 0 and qid in welllinked:
    #         acc += 1
    # tt.msg("{} = corrected seq accuracy".format(acc / totaltest))
    # correctedacc = acc / totaltest
    correctedacc = 0

    # log
    if log:
        import datetime
        trainlossscores = losses.get_agg_errors()
        validlossscores = validlosses.get_agg_errors()

        body = OrderedDict()
        body["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        body["settings"] = savesettings
        body["test_results"] = \
                   OrderedDict([("NLL", nll),
                                ("seq_acc", seqacc),
                                ("elem_acc", elemacc),
                                ("corr_seq_acc", correctedacc)])
        body["final_train_scores"] = \
                   OrderedDict([("train_NLL", trainlossscores[0]),
                                ("train_seq_acc", trainlossscores[1]),
                                ("train_elem_acc", trainlossscores[2])])
        body["final_valid_scores"] = \
                   OrderedDict([("valid_NLL", validlossscores[0]),
                                ("valid_seq_acc", validlossscores[1]),
                                ("valid_elem_acc", validlossscores[2])])

        q.log("experiments_seq2seq_lcquad.log", mode="a", name="seq2seq_run", body=body)

    # error analysis
    if erroranalysis:
        tt.msg("error analysis")
        erranal = ErrorAnalyzer(question_sm.D, tgt_emb.D, train_queries, mode="lcquad")
        anal_losses = q.lossarray((erranal, lambda x: (x, {})))
        q.test(m).cuda(cuda)\
            .on(test_dataloader, anal_losses)\
            .set_batch_transformer(bt)\
            .run()
        print(erranal.summary())
        erranal.inspect()


if __name__ == "__main__":
    q.argprun(run)
