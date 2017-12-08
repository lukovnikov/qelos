import torch
import sys
import qelos as q
from qelos.scripts.treesup.nltrees import GroupTracker, OrderSentence
import numpy as np
from collections import OrderedDict
import re
import ujson


OPT_LR = 0.1
OPT_BATSIZE = 128
OPT_GRADNORM = 5.
OPT_EPOCHS = 50
OPT_NUMEX = 1000
OPT_INPEMBDIM = 50
OPT_OUTEMBDIM = 50
OPT_LINOUTDIM = 50
OPT_JOINT_LINOUT_MODE = "sum"
OPT_ORACLE_MODE = "sample"
OPT_EXPLORE = 0.
OPT_DROPOUT = 0.3
OPT_WREG = 1e-5
OPT_ENCDIM = 100
OPT_DECDIM = 100
OPT_USEATTENTION = False
OPT_INPLINMODE = "df"       # "df" or "bf"

_opt_test = True
_tree_gen_seed = 1234


def load_jokes(p="../../../datasets/jokes/reddit_jokes.json",
               min_rating=0, maxlen=100):
    with open(p) as f:
        f = ujson.load(f)
    # print(len(f))
    lines = []
    for joke in f:
        if min_rating <= joke["score"]:
            line = joke["title"] + " . " + joke["body"]
            if maxlen is None or len(line.split()) < maxlen:
                lines.append(line)
    # print(len(lines))
    return lines


def run_seq_teacher_forced(lr=OPT_LR, batsize=OPT_BATSIZE, epochs=OPT_EPOCHS,
                           dropout=OPT_DROPOUT, gradnorm=OPT_GRADNORM, wreg=OPT_WREG,
                           embdim=100, lindim=100, decdim=200, maxlen=100,
                           cuda=False, gpu=0, devmode=False):
    """ normal language model with single RNN """
    print("started")
    if cuda:
        torch.cuda.set_device(gpu)
    tt = q.ticktock("script")
    ttt = q.ticktock("test")
    # region load data
    tt.tick("loading data")

    lines = load_jokes()
    if devmode:
        pass
        # lines = lines[:1000]
    sm = q.StringMatrix.load(".jokes.sm.cached")
    if sm is None:
        sm = q.StringMatrix(topnwords=100000, freqcutoff=5, indicate_start_end=True, maxlen=maxlen)
        for line in lines:
            sm.add(line)
        sm.finalize()
        sm.save(".jokes.sm.cached")
    tt.msg("data matrix size: {}".format(sm.matrix.shape))
    tt.msg("size dict: {}".format(len(sm.D)))
    for i in range(10):
        print(sm[i])
    tt.tock("data loaded")
    # endregion

    # sys.exit()

    # region make model
    # normal sequence generator model with teacher forcing
    emb = q.WordEmb(embdim, worddic=sm.D)
    lin = q.WordLinout(lindim, worddic=sm.D)

    layers = (torch.nn.Dropout(dropout), q.GRUCell(embdim, decdim),
              torch.nn.Dropout(dropout), q.GRUCell(decdim, lindim),)
    decoder_core = q.DecoderCore(emb, *layers)
    decoder_top = q.DecoderTop(q.wire((0, 0)), lin)
    decoder_cell = q.ModularDecoderCell(decoder_core, decoder_top)
    decoder_cell.set_runner(q.TeacherForcer())
    decoder = decoder_cell.to_decoder()

    if _opt_test:
        ttt.tick("testing dry-run")
        test_inp = q.var(sm.matrix[:5, :-1]).v
        test_gold = q.var(sm.matrix[:5, 1:]).v
        test_out = decoder(test_inp)
        # print(torch.max(test_out, 2)[1])
        # print(test_gold)
        loss = q.SeqCrossEntropyLoss(ignore_index=0)
        lossvalue = loss(test_out, test_gold)
        ttt.msg("value of Seq CE loss in test: {}".format(lossvalue))
        decoder.zero_grad()
        lossvalue.backward()
        ttt.msg("backward done")
        params = q.params_of(decoder)
        for param in params:
            assert(param.grad is not None)
            assert(param.grad.norm().data[0] > 0)
            print(tuple(param.size()), param.grad.norm().data[0])
        ttt.msg("all gradients non-zero")
        ttt.tock("tested dry-run")
    # endregion

    losses = q.lossarray(q.SeqCrossEntropyLoss(ignore_index=0),
                         q.SeqElemAccuracy(ignore_index=0),
                         q.SeqAccuracy(ignore_index=0), )

    optimizer = torch.optim.Adadelta(q.params_of(decoder), lr=lr, weight_decay=wreg)

    traindata, testdata = q.split([sm.matrix], random=1234)
    traindata, validdata = q.split(traindata, random=1234)

    train_loader = q.dataload(*traindata, batch_size=batsize, shuffle=True)
    valid_loader = q.dataload(*validdata, batch_size=batsize, shuffle=False)
    test_loader = q.dataload(*testdata, batch_size=batsize, shuffle=False)

    if devmode:
        sys.exit()

    q.train(decoder) \
        .train_on(train_loader, losses) \
        .optimizer(optimizer) \
        .clip_grad_norm(gradnorm) \
        .set_batch_transformer(
            lambda seq:
                (seq[:, :-1], seq[:, 1:])) \
        .valid_on(valid_loader, losses) \
        .cuda(cuda) \
        .train(epochs)

    results = q.test(decoder).on(test_loader, losses) \
        .set_batch_transformer(
            lambda seq:
                (seq[:, :-1], seq[:, 1:])) \
        .cuda(cuda) \
        .run()


if __name__ == "__main__":
    print("pytorch version: {}".format(torch.version.__version__))
    q.argprun(run_seq_teacher_forced)
    # q.argprun(run_seq2seq_teacher_forced_structured_output_tokens)
    # q.argprun(run_seq2seq_oracle)
    # q.argprun(run)