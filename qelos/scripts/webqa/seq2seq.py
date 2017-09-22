import qelos as q
from qelos.scripts.webqa.load import load_all
import torch
from torch import nn
import sys
import numpy as np
from collections import OrderedDict


def test_model(encoder, decoder, m, questions, queries, vnt):
    batsize = 10
    questions = q.var(questions[:batsize]).v
    queries = q.var(queries[:batsize]).v
    vnt = q.var(vnt[:batsize]).v

    # region try encoder
    ctx, ctxmask, finalctx = encoder(questions)
    # print(ctx.size())
    assert(ctx.size(0) == finalctx.size(0))
    assert(ctx.size(1) == ctxmask.float().size(1))
    assert(ctx.size(2) == finalctx.size(1))
    maskedctx = ctx * ctxmask.unsqueeze(2).float()
    assert((ctx.norm(2) == maskedctx.norm(2)).data.numpy()[0])
    # print(ctx.norm(2) - maskedctx.norm(2))
    loss = finalctx.sum()
    loss.backward()
    encoder.zero_grad()
    ctx, ctxmask, finalctx = encoder(questions)
    loss = finalctx.sum()
    loss.backward()
    encparams = q.params_of(encoder)

    print("dry run of encoder didn't throw errors")
    # endregion
    # region try decoder
    # decoder.block.embedder = nn.Embedding(100000, 200, padding_idx=0)
    # decoder.block.smo = q.Stack(
    #                                      q.argsave.spec(mask={"mask"}),
    #                                      q.argmap.spec(0),
    #                                      nn.Linear(200, 11075),
    #                                      q.argmap.spec(0, mask=["mask"]),
    #                                      q.LogSoftmax(),
    #                                      q.argmap.spec(0),
    #                     )
    # decoder.block.smo = None
    # try decoder cell
    for t in range(3):
        # ctx, ctxmask, finalctx = encoder(questions)
        decoder.block.core.reset_state()        # ESSENTIAL !!! otherwise double .backward() error
        decoder.set_init_states(finalctx.detach())
        decoder.block.zero_grad()
        outmaskt=vnt[:, t]
        # outmaskt=q.var(np.ones_like(vnt[:, t].data.numpy()).astype("int64")).v
        y_t, att_t = decoder.block(queries[:, t], ctx.detach(), ctxmask=ctxmask.detach(), t=t, outmask_t=outmaskt)
        loss = torch.max(y_t)
        print(loss)
        loss.backward()
        decparams = q.params_of(decoder)
        print("backward done")
    print("dry run of decoder cell didn't throw errors")
    # endregion

    # region whole model
    m.zero_grad()
    dec = m(questions, queries[:, :-1], vnt[:, 1:])
    assert(dec.size(0) == questions.size(0))
    assert(dec.size(1) == queries.size(1) - 1)
    # assert(dec.size(2) == 11075)
    loss = dec.sum()
    loss.backward()
    allparams = q.params_of(m)
    assert(len(set(allparams) - set(decparams) - set(encparams)) == 0)
    print("dry run of whole model didn't throw errors")
    # endregion
    # q.embed()


def make_encoder(src_emb, embdim=100, dim=100, dropout=0.0, **kw):
    """ make encoder
    # concatenating bypass encoder:
    #       embedding  --> top GRU
    #                  --> 1st BiGRU
    #       1st BiGRU  --> top GRU
    #                  --> 2nd BiGRU
    #       2nd BiGRU  --> top GRU
    """
    encoder = q.RecurrentStack(
        src_emb,        # embs, masks
        q.argsave.spec(emb=0, mask=1),
        q.argmap.spec(0, mask=["mask"]),
        q.BidirGRULayer(embdim, dim),
        q.TimesharedDropout(dropout),
        q.argsave.spec(bypass=0),
        q.argmap.spec(0, mask=["mask"]),
        q.BidirGRULayer(dim * 2, dim),
        q.TimesharedDropout(dropout),
        q.argmap.spec(0, ["bypass"], ["emb"]),
        q.Lambda(lambda x, y, z: torch.cat([x, y, z], 1)),
        q.argmap.spec(0, mask=["mask"]),
        q.GRULayer(dim * 4 + embdim, dim).return_final(True),
        q.argmap.spec(1, ["mask"], 0),
    )   # returns (all_states, mask, final_state)
    return encoder


def make_decoder(emb, lin, ctxdim=100, embdim=100, dim=100,
                 attmode="bilin", decsplit=False, **kw):
    """ makes decoder
    # attention cell decoder that accepts VNT !!!
    """
    ctxdim = ctxdim if not decsplit else ctxdim // 2
    coreindim = embdim + ctxdim     # if ctx_to_decinp is True else embdim

    coretocritdim = dim if not decsplit else dim // 2
    critdim = dim + embdim          # if decinp_to_att is True else dim

    if attmode == "bilin":
        attention = q.Attention().bilinear_gen(ctxdim, critdim)
    elif attmode == "fwd":
        attention = q.Attention().forward_gen(ctxdim, critdim)
    else:
        raise q.SumTingWongException()

    attcell = q.AttentionDecoderCell(attention=attention,
                                     embedder=emb,
                                     core=q.RecStack(
                                         q.GRUCell(coreindim, dim),
                                         q.GRUCell(dim, dim),
                                     ),
                                     smo=q.Stack(
                                         q.argsave.spec(mask={"mask"}),
                                         lin,
                                         q.argmap.spec(0, mask=["mask"]),
                                         q.LogSoftmax(),
                                         q.argmap.spec(0),
                                     ),
                                     ctx_to_decinp=True,
                                     ctx_to_smo=True,
                                     state_to_smo=True,
                                     decinp_to_att=True,
                                     state_split=decsplit,
                                     return_att=True,
                                     return_out=True)
    return attcell.to_decoder()


class Model(nn.Module):
    def __init__(self, encoder, decoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, srcseq, tgtseq, outmask=None):
        ctx, ctxmask, finalstate = self.encoder(srcseq)
        self.decoder.set_init_states(finalstate)
        dec, att = self.decoder(tgtseq, ctx, ctxmask=ctxmask, outmask=outmask)
        return dec


def run(lr=0.1,
        gradnorm=2.,
        epochs=100,
        wreg=1e-6,
        dropout=0.1,
        glovedim=50,
        encdim=100,
        decdim=100,
        decsplit=False,
        attmode="bilin",        # "bilin" or "fwd"
        merge_mode="sum",        # "cat" or "sum"
        rel_which="urlwords",     # "urlwords ... ..."
        batsize=128,
        cuda=False,
        gpu=1,
        inspectdata=False,
        log=True,
        ):
    print(locals()['glovedim'])
    localvars = locals()
    savesettings = "glovedim encdim decdim attmode gradnorm dropout merge_mode batsize epochs rel_which decsplit".split()
    savesettings = OrderedDict({k: localvars[k] for k in savesettings})
    if cuda:
        torch.cuda.set_device(gpu)

    tt = q.ticktock("script")
    # region I. data
    # load data and reps
    tt.tick("loading data and rep")
    rel_which = tuple(rel_which.split())
    flvecdim = 0
    if decsplit:    flvecdim += decdim // 2
    else:           flvecdim += decdim
    flvecdim += encdim
    (question_sm, query_sm, vnt_mat, tx_sep, qids), (src_emb, tgt_emb, tgt_lin) \
        = load_all(dim=flvecdim, glovedim=glovedim, merge_mode=merge_mode,
                   rel_which=rel_which)

    # test tgt_lin
    testlinx = q.var(np.random.random((3, flvecdim)).astype("float32")).v
    testlinvnt = np.zeros((3, vnt_mat.shape[-1]), dtype="int64")
    testlinvntx = np.asarray([0, 0, 0,  1, 1,   1, 2, 2,  2,   2])
    testlinvnty = np.asarray([0, 1, 21, 0, 1, 201, 0, 1, 21, 201])
    testlinvnt[testlinvntx, testlinvnty] = 1
    testliny = tgt_lin(testlinx, q.var(testlinvnt).v)


    tt.tock("loaded data and rep")
    tt.tick("making data loaders")
    # train/test split:
    train_questions, test_questions = question_sm.matrix[:tx_sep], question_sm.matrix[tx_sep:]
    train_queries, test_queries = query_sm.matrix[:tx_sep], query_sm.matrix[tx_sep:]
    train_vnt, test_vnt = vnt_mat[:tx_sep], vnt_mat[tx_sep:]

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
    decoder = make_decoder(tgt_emb, tgt_lin, ctxdim=ctxdim,
                           embdim=tgt_emb_dim, dim=decdim,
                           attmode=attmode, decsplit=decsplit)
    m = Model(encoder, decoder)

    test_model(encoder, decoder, m, test_questions, test_queries, test_vnt)

    # training settings
    losses = q.lossarray(q.SeqNLLLoss(time_average=False), q.SeqAccuracy(), q.SeqElemAccuracy())
    validlosses = q.lossarray(q.SeqNLLLoss(time_average=False), q.SeqAccuracy(), q.SeqElemAccuracy())
    testlosses = q.lossarray(q.SeqNLLLoss(time_average=False), q.SeqAccuracy(), q.SeqElemAccuracy())

    optimizer = torch.optim.Adadelta(q.params_of(m), lr=lr, weight_decay=wreg)

    # sys.exit()

    # train
    bt = lambda a, b, c: (a, c[:, :-1], b[:, 1:], c[:, 1:])
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

    if log:
        import datetime
        trainlossscores = losses.get_agg_errors()
        validlossscores = validlosses.get_agg_errors()

        body = OrderedDict()
        body["timestamp"] = datetime.datetime.now()
        body["settings"] = savesettings
        body["test_results"] = \
                   OrderedDict([("NLL", nll),
                                ("seq_acc", seqacc),
                                ("elem_acc", elemacc)])
        body["final_train_scores"] = \
                   OrderedDict([("train_NLL", trainlossscores[0]),
                                ("train_seq_acc", trainlossscores[1]),
                                ("train_elem_acc", trainlossscores[2])])
        body["final_valid_scores"] = \
                   OrderedDict([("valid_NLL", validlossscores[0]),
                                ("valid_seq_acc", validlossscores[1]),
                                ("valid_elem_acc", validlossscores[2])])

        q.log("experiments_seq2seq.log", mode="a", name="seq2seq_run", body=body)

    # TODO test number taking into account non-perfect starting entity linking !!!


if __name__ == "__main__":
    q.argprun(run)


# 22/09 -