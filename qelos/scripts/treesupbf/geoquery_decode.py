import torch
import qelos as q
import numpy as np
from qelos.scripts.treesupbf.trees import Node
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
            out = "( {} )".format(out)
            ism.add(inp)
            osm.add(out)
            numtrain += 1

    with open(p+testp) as f:
        for line in f:
            inp, out = line.split("\t")
            if reverse_input:
                inp = " ".join(inp.split()[::-1])
            out = "( {} )".format(out)
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


def parse_query_tree(x, _toprec=True, redro=False):    # "lambda $0 e ( and ( state:t $0 ) ( next_to:t $0 s0 ) )"
    if _toprec:
        x = x.strip().split()
    assert(x[0] == "(")
    parlevel = 0
    head = None
    children = []
    i = 0
    while i < len(x):
        xi = x[i]
        if xi == "(":     # parentheses open
            parlevel += 1
            if parlevel == 2:
                subtree, remainder = parse_query_tree(x[i:] + [], _toprec=False, redro=redro)
                children.append(subtree)
                x = remainder
                parlevel -= 1
                i = 0
            elif parlevel == 1:
                i += 1
            else:
                raise q.SumTingWongException("unexpected parlevel")
        elif xi == ")":
            parlevel -= 1
            i += 1
            break
        elif parlevel == 1:       # current head incoming
            if head is None:
                head = Node(xi)
                if _toprec and redro:
                    head.label = "0"
            else:
                children.append(Node(xi))
            i += 1
    head.children = tuple(children)
    if head.name == "and" or head.name == "or":
        pass
        if redro:
            for child in head.children:
                child.label = "0"
    else:
        for j, child in enumerate(head.children):
            if redro:
                child.label = str(j + 1)
            else:
                child.order = j + 1
    if _toprec:
        return head
    else:
        return head, x[i:]
    # if i == len(x):
    #     return head
    # else:
    #     return head, x[i:]






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

    # validation with freerunner
    freerunner = q.FreeRunner()
    valid_decoder_cell = q.ModularDecoderCell(decoder_core, decoder_top)
    valid_decoder_cell.set_runner(freerunner)
    valid_decoder = valid_decoder_cell.to_decoder()
    valid_encdec = EncDecAtt(encoder, valid_decoder)

    from qelos.scripts.treesupbf.pasdecode import TreeAccuracy

    rev_outD = {v: k for k, v in outD.items()}

    def treeparser(x):  # 1D of output word ids
        treestring = " ".join([rev_outD[xe] for xe in x if xe != 0])
        tree = parse_query_tree(treestring)
        return tree

    validlosses = q.lossarray(q.SeqCrossEntropyLoss(ignore_index=0),
                              q.SeqElemAccuracy(ignore_index=0),
                              q.SeqAccuracy(ignore_index=0),
                              TreeAccuracy(ignore_index=0, treeparser=treeparser))

    q.train(encdec).train_on(train_loader, losses).optimizer(optimizer)\
        .clip_grad_norm(gradnorm) \
        .set_batch_transformer(lambda x, y: (x, y[:, :-1], y[:, 1:]))\
        .valid_with(valid_encdec).valid_on(valid_loader, validlosses)\
        .cuda(cuda).train(epochs)

    results = q.test(valid_encdec).on(test_loader, validlosses)\
        .set_batch_transformer(lambda x, y: (x, y[:, :-1], y[:, 1:]))\
        .cuda(cuda).run()

    # print(encdec)


def run_noisy_parse():
    treen = parse_query_tree("( lambda $0 e ( and ( state:t $0 ) ( and ( next_to:t $0 s0 ) ( next_to:t $0 s0 ) ) ) ) ( ) ) ) ) ) )")
    tree = parse_query_tree("( lambda $0 e ( and ( state:t $0 ) ( and ( next_to:t $0 s0 ) ( next_to:t $0 s0 ) ) ) )")
    print(tree.pp())
    print(treen.pp())
    # sys.exit()


def run_some_tests():
    #
    # q.embed()

    run_noisy_parse()
    #
    # q.embed()

    tree = parse_query_tree("( lambda $0 e ( and ( state:t $0 ) ( next_to:t $0 s0 ) ) )")
    secondtree = parse_query_tree("( lambda $0 e ( and ( next_to:t $0 s0 ) ( state:t $0 ) ) )")
    print("different orderings equal: {}".format(tree == secondtree))
    print(tree.pptree())
    tree = "( argmax $0 ( and ( river:t $0 ) ( exists $1 ( and ( state:t $1 ) ( next_to:t $1 ( argmax $2 ( state:t $2 ) ( count $3 ( and ( state:t $3 ) ( next_to:t $2 $3 ) ) ) ) ) ) ) ) ( len:i $0 ) )"
    print(tree)
    tree = parse_query_tree(tree, redro=True)
    # print(tree.pptree())
    print(tree.ppdf(mode="par"))
    print(tree.pp())
    print(tree.pptree())

    import random
    tracker = tree.track()
    uniquelins = set()
    numsam = 5000
    for i in range(numsam):
        tracker.reset()
        nvt = tracker._nvt
        tokens = []
        while len(nvt) > 0:
            x = random.choice(list(nvt))
            tokens.append(x)
            nvt = tracker.nxt(x)
        lin = " ".join(tokens)
        recons = Node.parse(lin)
        assert (recons == tree)
        if numsam < 11:
            print(recons.pp())
            print(recons.pptree())
        uniquelins.add(lin)
    print("{} unique lins for tree".format(len(uniquelins)))


if __name__ == "__main__":
    # run_some_tests()
    q.argprun(run_seq2seq_reproduction)