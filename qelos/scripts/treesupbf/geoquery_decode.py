import torch
import qelos as q
import numpy as np
import sys


OPT_LR = 0.1
OPT_EPOCHS = 100
OPT_BATSIZE = 20

OPT_WREG = 0.00001
OPT_DROPOUT = 0.2
OPT_GRADNORM = 5.

OPT_INPEMBDIM = 50
OPT_OUTEMBDIM = 50
OPT_INNERDIM = 100

_opt_test = True


def make_encoder(inpemb, inpembdim, encdim, dropout, ttt=None):
    ttt = q.ticktock("encoder test") if ttt is None else ttt
    encoder = q.RecurrentStack(
        inpemb,
        q.wire((-1, 0)),
        q.TimesharedDropout(dropout),
        q.wire((1, 0), mask=(1, 1)),
        q.BidirGRULayer(inpembdim, encdim // 2).return_final(True),
        q.wire((-1, 1)),
        q.TimesharedDropout(dropout),
        q.wire((3, 0), (-1, 0)),
        q.RecurrentLambda(lambda x, y: torch.cat([x, y], 2)),
        q.wire((-1, 0), mask=(1, 1)),
        q.BidirGRULayer(encdim + inpembdim, encdim // 2).return_final(True),
        q.wire((5, 1), (11, 1)),
        q.RecurrentLambda(lambda x, y: q.intercat([
            q.intercat(torch.chunk(x, 2, 2)),
            q.intercat(torch.chunk(y, 2, 2))])),
        q.wire((5, 0), (11, 0)),
        q.RecurrentLambda(lambda x, y: q.intercat([
            q.intercat(torch.chunk(x, 2, 1)),
            q.intercat(torch.chunk(y, 2, 1))])),
        q.wire((-1, 0), (-3, 0), (1, 1)),
    )

    if _opt_test:
        ttt.tick("testing encoder")
        # ttt.msg("encoder\n {} \nhas {} layers".format(encoder, len(encoder.layers)))
        test_input_symbols = q.var(np.random.randint(0, 44, (3, 10))).v
        test_encoder_output = encoder(test_input_symbols)
        ttt.msg("encoder return {} outputs".format(len(test_encoder_output)))
        ttt.msg("encoder output shapes: {}".format(" ".join([str(test_encoder_output_e.size()) for test_encoder_output_e in test_encoder_output])))
        assert(test_encoder_output[1].size() == (3, 10, encdim * 2))
        assert(test_encoder_output[0].size() == (3, encdim * 2))
        ttt.tock("tested encoder (output shapes)")
    return encoder


def load_data(p="../../../datasets/geoquery/", trainp="train.txt", testp="test.txt", reverse_input=False):
    tt = q.ticktock("dataloader")
    tt.tick("loading data")
    ism = q.StringMatrix()
    ism.tokenize = lambda x: x.split()
    osm = q.StringMatrix(indicate_start=True)
    osm.tokenize = lambda x: x.split()

    numtrain = 0
    with open(p+trainp) as f:
        for line in f:
            inp, out = line.split("\t")
            if reverse_input:
                inp = " ".join(inp.split()[::-1])
            ism.add(inp)
            osm.add(out)
            numtrain += 1

    with open(p+testp) as f:
        for line in f:
            inp, out = line.split("\t")
            if reverse_input:
                inp = " ".join(inp.split()[::-1])
            ism.add(inp)
            osm.add(out)

    ism.finalize()
    osm.finalize()

    print(ism[0])
    print(osm[0])
    print(ism[numtrain])
    print(osm[numtrain])

    # check overlap
    trainseqs = set()
    testseqs = set()
    for i in range(ism.matrix.shape[0]):
        if i < numtrain:
            trainseqs.add(ism[i] + " - " + osm[i])
        else:
            testseqs.add(ism[i] + " - " + osm[i])
    tt.msg("overlap: {}/{} of test occur in train ({})".format(
        len(testseqs & trainseqs), len(testseqs), len(trainseqs)))
    tt.tock("data loaded")
    trainmats = (ism.matrix[:numtrain], osm.matrix[:numtrain])
    testmats = (ism.matrix[numtrain:], osm.matrix[numtrain:])

    return trainmats, testmats, ism.D, osm.D


def run_seq2seq_reproduction(lr=OPT_LR, epochs=OPT_EPOCHS, batsize=OPT_BATSIZE,
                             wreg=OPT_WREG, dropout=OPT_DROPOUT, gradnorm=OPT_GRADNORM,
                             inpembdim=OPT_INPEMBDIM, outembdim=OPT_OUTEMBDIM, innerdim=OPT_INNERDIM,
                             cuda=False, gpu=0,
                             validontest=False):
    if validontest:
        print("VALIDATING ON TEST: WONG !!!")
    print("SEQSEQ REPRODUCTION")
    if cuda:    torch.cuda.set_device(gpu)
    tt = q.ticktock("script")
    ttt = q.ticktock("test")
    trainmats, testmats, inpD, outD = load_data(reverse_input=True)

    inpemb = q.WordEmb(inpembdim, worddic=inpD)
    outemb = q.WordEmb(outembdim, worddic=outD)
    linout = q.WordLinout(innerdim + innerdim, worddic=outD)

    encoder = make_encoder(inpemb, inpembdim, innerdim//2, dropout, ttt=ttt)

    layers = (torch.nn.Dropout(0),
              q.GRUCell(outembdim, innerdim),
              q.GRUCell(innerdim, innerdim))

    decoder_top = q.AttentionContextDecoderTop(q.Attention().dot_gen(),
                                               linout, ctx2out=False)

    decoder_core = q.DecoderCore(outemb, *layers)
    decoder_cell = q.ModularDecoderCell(decoder_core, decoder_top)
    decoder_cell.set_runner(q.TeacherForcer())
    decoder = decoder_cell.to_decoder()

    class EncDecAtt(torch.nn.Module):
        def __init__(self, _encoder, _decoder, **kwargs):
            super(EncDecAtt, self).__init__(**kwargs)
            self.encoder = _encoder
            self.decoder = _decoder

        def forward(self, inpseq, outinpseq):
            final_encoding, all_encoding, mask = self.encoder(inpseq)
            self.decoder.set_init_states(None, final_encoding)
            decoding = self.decoder(outinpseq,
                                    ctx=all_encoding,
                                    ctx_0=final_encoding,
                                    ctxmask=mask)
            return decoding

    encdec = EncDecAtt(encoder, decoder)

    if validontest:
        traindata = trainmats
        validdata = testmats
    else:
        traindata, validdata = q.split(trainmats, random=True)
    train_loader = q.dataload(*traindata, batch_size=batsize, shuffle=True)
    valid_loader = q.dataload(*validdata, batch_size=batsize, shuffle=False)
    test_loader = q.dataload(*testmats, batch_size=batsize, shuffle=False)

    losses = q.lossarray(q.SeqCrossEntropyLoss(ignore_index=0),
                         q.SeqElemAccuracy(ignore_index=0),
                         q.SeqAccuracy(ignore_index=0))

    optimizer = torch.optim.Adadelta(q.params_of(encdec), lr=lr, weight_decay=wreg)

    q.train(encdec).train_on(train_loader, losses).optimizer(optimizer)\
        .clip_grad_norm(gradnorm) \
        .set_batch_transformer(lambda x, y: (x, y[:, :-1], y[:, 1:]))\
        .valid_on(valid_loader, losses)\
        .cuda(cuda).train(epochs)

    results = q.test(encdec).on(test_loader, losses)\
        .set_batch_transformer(lambda x, y: (x, y[:, :-1], y[:, 1:]))\
        .cuda(cuda).run()

    # print(encdec)


if __name__ == "__main__":
    q.argprun(run_seq2seq_reproduction)
