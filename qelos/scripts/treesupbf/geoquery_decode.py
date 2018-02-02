import torch
import qelos as q
import numpy as np
from qelos.scripts.treesupbf.trees import Node, GroupTracker
from qelos.scripts.treesupbf.pasdecode import TreeAccuracy
from qelos.furnn import ParentStackCell
import random
import sys


OPT_LR = 0.0003
OPT_LR_DECAY = 0.99
OPT_EPOCHS = 100
OPT_BATSIZE = 20

OPT_WREG = 0.000001
OPT_DROPOUT = 0.5
OPT_GRADNORM = 5.

OPT_INPEMBDIM = 200
OPT_OUTEMBDIM = 200
OPT_INNERDIM = 200

OPT_ORACLEMODE = "esample"       # "sample" or "uniform" or ...

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


def load_data_trees(p="../../../datasets/geoquery/", trainp="train.txt", testp="test.txt", reverse_input=False):
    tt = q.ticktock("dataloader")
    tt.tick("loading data")
    ism = q.StringMatrix()
    ism.tokenize = lambda x: x.split()

    trees = []
    numtrain = 0

    with open(p+trainp) as f:
        for line in f:
            inp, out = line.split("\t")
            if reverse_input:
                inp = " ".join(inp.split()[::-1])
            out = "( {} )".format(out)
            ism.add(inp)
            trees.append(parse_query_tree(out))
            numtrain += 1

    with open(p+testp) as f:
        for line in f:
            inp, out = line.split("\t")
            if reverse_input:
                inp = " ".join(inp.split()[::-1])
            out = "( {} )".format(out)
            ism.add(inp)
            trees.append(parse_query_tree(out))

    ism.finalize()
    tracker = GroupTracker(trees)

    print(ism[0])
    print(tracker[0].root.pptree())

    print(tracker[:5][0].root.pptree())

    # check number of different linearizations
    first = True
    for i in range(80, 85):
        numsam = 200
        uniquelins = set()
        for j in range(numsam):
            tracker[i].reset()
            nvt = tracker[i]._nvt
            tokens = []
            while len(nvt) > 0:
                x = random.choice(list(nvt))
                tokens.append(x)
                nvt = tracker[i].nxt(x)
            lin = " ".join(tokens)
            recons = Node.parse(lin)
            assert (recons == tracker[i].root)
            if numsam < 11:
                print(recons.pp())
                print(recons.pptree())
            uniquelins.add(lin)
        tracker[i].reset()
        print("{} unique lins for tree {}".format(len(uniquelins), tracker[i].root.pp()))
        if first:
            for uniquelin in uniquelins:
                print(uniquelin)
        first = False

    # check overlap
    trainseqs = set()
    testseqs = set()
    for i in range(ism.matrix.shape[0]):
        if i < numtrain:
            trainseqs.add(ism[i] + " - " + tracker[i].root.pp(arbitrary=False))
        else:
            testseqs.add(ism[i] + " - " + tracker[i].root.pp(arbitrary=False))
    tt.msg("overlap: {}/{} of test occur in train ({})".format(
        len(testseqs & trainseqs), len(testseqs), len(trainseqs)))

    tt.tock("data loaded")
    return ism, tracker, numtrain


def load_data(p="../../../datasets/geoquery/", trainp="train.txt", testp="test.txt", reverse_input=False):
    tt = q.ticktock("dataloader")
    tt.tick("loading data")
    ism = q.StringMatrix()
    ism.tokenize = lambda x: x.split()
    osm = q.StringMatrix(indicate_start=True, indicate_end=False)
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
    """ given a query in lambda, generates partially ordered tree.
        redro: ??? TODO """
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


class PredCutter(object):
    def __init__(self, D, **kw):
        super(PredCutter, self).__init__()
        self.D = D
        self.paropen = D["("]
        self.parclose = D[")"]

    def __call__(self, a, ignore_index=None):   # a: 2D of ints
        for i in range(len(a)):
            numpar = 0
            fillzero = False
            for j in range(len(a[i])):
                if not fillzero:
                    if a[i, j] == self.paropen:
                        numpar += 1
                    elif a[i, j] == self.parclose:
                        numpar -= 1
                        if numpar == 0:
                            fillzero = True
                else:
                    if ignore_index is not None:
                        a[i, j] = ignore_index
        return a


def run_seq2seq_reproduction(lr=OPT_LR, lrdecay=OPT_LR_DECAY, epochs=OPT_EPOCHS, batsize=OPT_BATSIZE,
                             wreg=OPT_WREG, dropout=OPT_DROPOUT, gradnorm=OPT_GRADNORM,
                             embdim=-1,
                             inpembdim=OPT_INPEMBDIM, outembdim=OPT_OUTEMBDIM, innerdim=OPT_INNERDIM,
                             cuda=False, gpu=0,
                             validontest=False):
    settings = locals().copy()
    logger = q.Logger(prefix="geoquery_s2s_repro")
    logger.save_settings(**settings)
    logger.update_settings(version="4")
    logger.update_settings(completed=False)

    if validontest:
        print("VALIDATING ON TEST: WONG !!!")
    print("SEQSEQ REPRODUCTION")
    if cuda:    torch.cuda.set_device(gpu)
    tt = q.ticktock("script")
    ttt = q.ticktock("test")
    trainmats, testmats, inpD, outD = load_data(reverse_input=True)

    if embdim > 0:
        tt.msg("embdim overrides inpembdim and outembdim")
        inpembdim, outembdim = embdim, embdim

    inpemb = q.WordEmb(inpembdim, worddic=inpD)     # TODO glove embeddings
    outemb = q.WordEmb(outembdim, worddic=outD)
    linout = q.WordLinout(innerdim + innerdim, worddic=outD)

    # encoder = make_encoder(inpemb, inpembdim, innerdim//2, dropout, ttt=ttt)/
    encoderstack = q.RecStack(
        q.wire((0, 0), mask_t=(0, {"mask_t"}), t=(0, {"t"})),
        q.LSTMCell(inpembdim, innerdim, dropout_in=dropout, dropout_rec=None),
    ).to_layer().return_final().return_mask()
    encoder = q.RecurrentStack(
        inpemb,
        encoderstack,
    )

    # test
    # testencret = encoder(q.var(trainmats[0][:5]).v)

    layers = (q.LSTMCell(outembdim, innerdim, dropout_in=dropout, dropout_rec=None),
              )

    decoder_top = q.AttentionContextDecoderTop(q.Attention().dot_gen(),
                                               q.Dropout(dropout),
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
            self.dropout = q.Dropout(dropout)

        def forward(self, inpseq, outinpseq):
            final_encoding, all_encoding, mask = self.encoder(inpseq)
            # self.decoder.set_inew loss in trainer for stupidmodelnit_states(None, final_encoding)
            encoderstates = self.encoder.layers[1].cell.get_states(inpseq.size(0))
            initstates = [self.dropout(initstate) for initstate in encoderstates]
            self.decoder.set_init_states(*initstates)
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
    # q.embed()
    train_loader = q.dataload(*traindata, batch_size=batsize, shuffle=True)
    valid_loader = q.dataload(*validdata, batch_size=batsize, shuffle=False)
    test_loader = q.dataload(*testmats, batch_size=batsize, shuffle=False)

    losses = q.lossarray(q.SeqCrossEntropyLoss(ignore_index=0),
                         q.SeqElemAccuracy(ignore_index=0),
                         q.MacroBLEU(ignore_index=0, predcut=PredCutter(outD)),
                         q.SeqAccuracy(ignore_index=0))

    # validation with freerunner
    freerunner = q.FreeRunner()
    valid_decoder_cell = q.ModularDecoderCell(decoder_core, decoder_top)
    valid_decoder_cell.set_runner(freerunner)
    valid_decoder = valid_decoder_cell.to_decoder()
    valid_encdec = EncDecAtt(encoder, valid_decoder)

    rev_outD = {v: k for k, v in outD.items()}

    def treeparser(x):  # 1D of output word ids
        treestring = " ".join([rev_outD[xe] for xe in x if xe != 0])
        tree = parse_query_tree(treestring)
        return tree

    validlosses = q.lossarray(#q.SeqCrossEntropyLoss(ignore_index=0),
                              q.MacroBLEU(ignore_index=0, predcut=PredCutter(outD)),
                              q.SeqAccuracy(ignore_index=0),
                              TreeAccuracy(ignore_index=0, treeparser=treeparser))

    logger.update_settings(optimizer="rmsprop")
    optim = torch.optim.RMSprop(q.paramgroups_of(encdec), lr=lr, weight_decay=wreg)

    lrsched = torch.optim.lr_scheduler.ExponentialLR(optim, lrdecay)

    q.train(encdec).train_on(train_loader, losses)\
        .optimizer(optim)\
        .clip_grad_norm(gradnorm) \
        .set_batch_transformer(lambda x, y: (x, y[:, :-1], y[:, 1:]))\
        .valid_with(valid_encdec).valid_on(valid_loader, validlosses)\
        .cuda(cuda)\
        .hook(logger)\
        .hook(lrsched) \
        .train(epochs)

    logger.update_settings(completed=True)

    results = q.test(valid_encdec).on(test_loader, validlosses)\
        .set_batch_transformer(lambda x, y: (x, y[:, :-1], y[:, 1:]))\
        .cuda(cuda).run()

    # print(encdec)


def make_multilinout(lindim, grouptracker=None, tie_weights=False, chained=False, ttt=None):
    assert(grouptracker is not None)
    ttt = q.ticktock("linout test") if ttt is None else ttt
    inpD, outD = grouptracker.D_in, grouptracker.D

    # region symbols to cores and ctrls
    symbols_is_last = np.zeros((len(outD),), dtype="int64")
    symbols_is_leaf = np.zeros((len(outD),), dtype="int64")
    for k, v in sorted(outD.items(), key=lambda (x, y): y):
        ksplits = k.split("*")
        assert(ksplits[0] in inpD)
        symbols_is_last[v] = "LS" in ksplits[1:] or k in "<MASK> <STOP> <NONE> <ROOT>".split()
        symbols_is_leaf[v] = "NC" in ksplits[1:] or k in "<MASK> <START> <STOP> <NONE>".split()

    symbols2ctrl = symbols_is_last * 2 + symbols_is_leaf + 1
    symbols2ctrl[outD["<MASK>"]] = 0
    symbols2cores = np.zeros((len(outD),), dtype="int64")
    for symbol in outD:
        symbols2cores[outD[symbol]] = inpD[symbol.split("*")[0]]
    assert(inpD["<MASK>"] == 0 == outD["<MASK>"])
    assert(inpD["<START>"] == 1 == outD["<START>"])
    assert(inpD["<STOP>"] == 2 == outD["<STOP>"])
    assert(inpD["<ROOT>"] == 3 == outD["<ROOT>"])
    assert(inpD["<NONE>"] == 4 == outD["<NONE>"])
    # endregion

    class StrucSMO(torch.nn.Module):
        def __init__(self, indim, worddic=None, chained=False, **kw):
            super(StrucSMO, self).__init__(**kw)
            self.dim = indim
            self.D = worddic
            self.chained = chained
            self.coreout = q.WordLinout(self.dim+2, worddic=inpD, bias=False)
            annD = {"A": 0, "NC": 1, "LS": 2, "NCLS": 3}
            self.annout = q.WordLinout(self.dim, worddic=annD, bias=False)
            self.annemb = q.WordEmb(self.dim, worddic=annD)
            self.annemb.embedding.weight.data.fill_(0)

            appendix = torch.zeros(1, 4, 2)
            if self.chained:
                appendix[:, 1, 0] = 1.      # NC
                appendix[:, 2, 1] = 1.      # LS
                appendix[:, 3, :] = 1.      # NCLS
            self.appendix = q.val(appendix).v

            coreprobmask = torch.ones(1, 4, self.coreout.lin.weight.size(0))
            coreprobmask[:, :, self.coreout.D["<MASK>"]] = 0
            coreprobmask[:, :, self.coreout.D["<START>"]] = 0
            coreprobmask[:, :, self.coreout.D["<STOP>"]] = 0
            coreprobmask[:, :, self.coreout.D["<ROOT>"]] = 0
            coreprobmask[:, :, self.coreout.D["<NONE>"]] = 0

            coreprobmask[:, 3, self.coreout.D["<MASK>"]] = 1
            coreprobmask[:, 1, self.coreout.D["<START>"]] = 1
            coreprobmask[:, 3, self.coreout.D["<STOP>"]] = 1
            coreprobmask[:, 2, self.coreout.D["<ROOT>"]] = 1
            coreprobmask[:, 3, self.coreout.D["<NONE>"]] = 1

            self.coreprobmask = q.val(coreprobmask).v

        def forward(self, x):
            # predict structure:
            ctrlprobs = self.annout(x)
            ctrlprobs = torch.nn.LogSoftmax(1)(ctrlprobs).unsqueeze(2)
            # prepare core pred
            appendix = self.appendix.repeat(x.size(0), 1, 1)
            xx = x.unsqueeze(1)
            addition = self.annemb.embedding.weight.unsqueeze(0)
            if not self.chained:
                addition = q.var(torch.zeros(addition.size())).cuda(addition).v
            xxx = xx + addition
            xxxx = torch.cat([appendix, xxx], 2)
            # core pred
            coreprobs = self.coreout(xxxx)
            coreprobs += torch.log(self.coreprobmask)
            coreprobs = torch.nn.LogSoftmax(2)(coreprobs)
            allprobs = ctrlprobs + coreprobs
            # join into D's space
            specialprobs, _ = torch.max(allprobs[:, :, :5], 1)
            _allprobs = allprobs[:, :, 5:]   # assumes special symbols are only five and they're first and only scored in one struct condition
            otherprobs = _allprobs.contiguous().view(x.size(0), -1)   # assumes all real symbols are combined with every struct decision
            retprobs = torch.cat([specialprobs, otherprobs], 1)

            return retprobs

    ret = StrucSMO(lindim, worddic=outD, chained=chained)
    # TODO TEST
    return ret, symbols2cores, symbols2ctrl


def make_outvecs(embdim, lindim, grouptracker=None, tie_weights=False, ttt=None):
    assert(grouptracker is not None)
    ttt = q.ticktock("linout test") if ttt is None else ttt

    # region PREPARE ------------------------------------
    inpD, outD = grouptracker.D_in, grouptracker.D
    # computed wordlinout for output symbols with topology annotations
    # dictionaries
    symbols_is_last = np.zeros((len(outD),), dtype="int64")
    symbols_is_leaf = np.zeros((len(outD),), dtype="int64")
    for k, v in sorted(outD.items(), key=lambda (x, y): y):
        ksplits = k.split("*")
        assert(ksplits[0] in inpD)
        symbols_is_last[v] = "LS" in ksplits[1:] or k in "<MASK> <STOP> <NONE> <ROOT>".split()
        symbols_is_leaf[v] = "NC" in ksplits[1:] or k in "<MASK> <START> <STOP> <NONE>".split()

    symbols2ctrl = symbols_is_last * 2 + symbols_is_leaf + 1
    symbols2ctrl[outD["<MASK>"]] = 0

    symbols2cores = np.zeros((len(outD),), dtype="int64")
    for symbol in outD:
        symbols2cores[outD[symbol]] = inpD[symbol.split("*")[0]]
    # endregion

    # region EMBEDDER ------------------------------------
    class StructEmbComputer(torch.nn.Module):
        def __init__(self, dim, **kw):
            super(StructEmbComputer, self).__init__(**kw)
            self.dim = dim
            self.coreemb = q.WordEmb(self.dim, worddic=inpD)
            self.annemb = q.WordEmb(self.dim, worddic={"<MASK>": 0, "A": 1, "NC": 2, "LS": 3, "NCLS": 4})
            self.annemb.embedding.weight.data.fill_(0)

        def forward(self, data):
            coreembs, mask = self.coreemb(data[:, 0])
            annembs, _ = self.annemb(data[:, 1])
            embs = coreembs + annembs
            return embs

    computer = StructEmbComputer(dim=embdim)
    computer_data = np.stack([symbols2cores, symbols2ctrl], axis=1)
    wordemb = q.ComputedWordEmb(data=computer_data, computer=computer, worddic=outD)
    # endregion

    # region LINOUT --------------------------------------
    if not tie_weights:
        computer = StructEmbComputer(dim=lindim)    # make new structembcomp
    else:
        assert(lindim == embdim)
    linout = q.ComputedWordLinout(data=computer_data, computer=computer, worddic=outD)
    # endregion

    if _opt_test:
        # TODO: test
        pass

    return wordemb, linout, symbols2cores, symbols2ctrl


def run_seq2tree_tf(lr=OPT_LR, lrdecay=OPT_LR_DECAY, epochs=OPT_EPOCHS, batsize=OPT_BATSIZE,
                             wreg=OPT_WREG, dropout=OPT_DROPOUT, gradnorm=OPT_GRADNORM,
                             embdim=-1,
                             inpembdim=OPT_INPEMBDIM, outembdim=OPT_OUTEMBDIM, innerdim=OPT_INNERDIM,
                             cuda=False, gpu=0, splitseed=14567,
                             decodermode="single", useattention=True,
                             validontest=False):
    settings = locals().copy()
    logger = q.Logger(prefix="geoquery_s2tree_tf")
    logger.save_settings(**settings)
    logger.update_settings(completed=False)
    logger.update_settings(version="2")
    # version "2": with strucSMO
    # verison "3": unchained strucSMO

    if validontest:
        print("VALIDATING ON TEST: WONG !!!")
    print("SEQ2TREE TF")
    if cuda:    torch.cuda.set_device(gpu)
    tt = q.ticktock("script")
    ttt = q.ticktock("test")
    ism, tracker, numtrain = load_data_trees(reverse_input=False)
    eids = np.arange(0, len(ism), dtype="int64")
    psm = q.StringMatrix()
    psm.set_dictionary(tracker.D)
    psm.tokenize = lambda x: x.split()
    for tree in tracker.trackables:
        treestring = tree.pp(arbitrary=False, _remove_order=True)
        assert(Node.parse(treestring) == tree)
        psm.add("<ROOT> " + treestring)
    psm.finalize()
    print(ism[0])
    print(psm[0])

    # region MODEL --------------------------------
    # region EMBEDDINGS ----------------------------------
    if embdim > 0:
        tt.msg("embdim overrides inpembdim and outembdim")
        inpembdim, outembdim = embdim, embdim

    linoutdim = innerdim + (innerdim if useattention else 0)

    inpemb = q.WordEmb(inpembdim, worddic=ism.D)  # TODO glove embeddings

    outemb, linout, symbols2cores, symbols2ctrl \
        = make_outvecs(outembdim, linoutdim, grouptracker=tracker, tie_weights=False, ttt=ttt)

    # linout, _symbols2cores, _symbols2ctrl = make_multilinout(linoutdim, grouptracker=tracker, tie_weights=False, ttt=ttt)
    # assert(np.all(symbols2cores == _symbols2cores) and np.all(symbols2ctrl == _symbols2ctrl))

    outemb = q.WordEmb(outembdim, worddic=tracker.D_in)
    # endregion

    # region ENCODER -------------------------------------
    # encoder = make_encoder(inpemb, inpembdim, innerdim//2, dropout, ttt=ttt)/
    encoderstack = q.RecStack(
        q.wire((0, 0), mask_t=(0, {"mask_t"}), t=(0, {"t"})),
        q.LSTMCell(inpembdim, innerdim, dropout_in=dropout, dropout_rec=None),
    ).to_layer().return_final().return_mask().reverse()
    encoder = q.RecurrentStack(
        inpemb,
        encoderstack,
    )
    # endregion

    # region DECODER -------------------------------------

    if decodermode == "double":
        layers = (q.LSTMCell(outembdim, innerdim//2, dropout_in=dropout, dropout_rec=None),
                  q.LSTMCell(outembdim, innerdim//2, dropout_in=dropout, dropout_rec=None),
                  )
    elif decodermode == "single":
        layers = q.LSTMCell(outembdim*2, innerdim, dropout_in=dropout, dropout_rec=None)
    if useattention:
        tt.msg("Attention: YES!")
        decoder_top = q.AttentionContextDecoderTop(q.Attention().dot_gen(),
                                               q.Dropout(dropout),
                                               linout, ctx2out=False)
    else:
        tt.msg("Attention: NO")
        decoder_top = q.DecoderTop(q.wire((0, 0)), q.Dropout(dropout), linout)

    decoder_core = ParentStackCell(outemb, layers)
    decoder_cell = q.ModularDecoderCell(decoder_core, decoder_top)
    decoder_cell.set_runner(q.TeacherForcer())
    decoder = decoder_cell.to_decoder()

    # endregion

    # region ENCDEC ---------------------------------------
    # wrap in encdec
    class EncDec(torch.nn.Module):
        def __init__(self, encoder, decoder, **kw):
            super(EncDec, self).__init__(**kw)
            self.encoder = encoder
            self.decoder = decoder

        def forward(self, inpseq, outinpseq, ctrlseq):
            final_encoding, all_encoding, mask = self.encoder(inpseq)
            # self.decoder.block.decoder_top.set_ctx(final_encoding)
            if decodermode == "double":
                self.decoder.set_init_states(None,
                                             final_encoding[:, :final_encoding.size(1) // 2],
                                             None,
                                             final_encoding[:, final_encoding.size(1) // 2:])
            elif decodermode == "single":
                self.decoder.set_init_states(None, final_encoding)
            self.decoder.set_init_states(None)
            decoding = self.decoder(outinpseq, ctrlseq)
            return decoding

    class EncDecAtt(torch.nn.Module):
        def __init__(self, encoder, decoder, **kw):
            super(EncDecAtt, self).__init__(**kw)
            self.encoder, self.decoder = encoder, decoder

        def forward(self, inpseq, outinpseq, ctrlseq):
            final_encoding, all_encoding, mask = self.encoder(inpseq)
            decoding = self.decoder(outinpseq, ctrlseq,
                                    ctx=all_encoding,
                                    ctx_0=final_encoding,
                                    ctxmask=mask)
            return decoding

    if useattention:
        encdec = EncDecAtt(encoder, decoder)
    else:
        encdec = EncDec(encoder, decoder)

    # endregion
    # endregion

    # region DATA LOADING
    traindata = [ism.matrix[:numtrain], psm.matrix[:numtrain, :-1], psm.matrix[:numtrain, 1:]]
    testdata = [ism.matrix[numtrain:], psm.matrix[numtrain:, :-1], psm.matrix[numtrain:, 1:]]

    if validontest:
        validdata = testdata
    else:
        traindata, validdata = q.split(traindata, random=splitseed)

    train_loader = q.dataload(*traindata, batch_size=batsize, shuffle=True)
    valid_loader = q.dataload(*validdata, batch_size=batsize, shuffle=False)
    test_loader = q.dataload(*testdata, batch_size=batsize, shuffle=False)
    # endregion

    symbols2cores = q.var(symbols2cores).cuda(cuda).v
    symbols2ctrl = q.var(symbols2ctrl).cuda(cuda).v

    def symbol2corenctrl(s):
        _cores = torch.index_select(symbols2cores, 0, s)
        _ctrl = torch.index_select(symbols2ctrl, 0, s)
        return _cores, _ctrl

    def inbtf(a, b, g):
        bflat = b.view(-1)
        _cores, _ctrl = symbol2corenctrl(bflat)
        _cores, _ctrl = _cores.view(b.size()), _ctrl.view(b.size())
        return a, _cores, _ctrl, g

    # validation with freerunner
    inparggetter = symbol2corenctrl

    if _opt_test:
        # test that produced cores consistent with outemb's D
        test_x = psm.matrix[80:85]
        revdin = {v: k for k, v in outemb.D.items()}
        for i in range(len(test_x)):
            print(tracker.pp(test_x[i]))
            test_cores, test_ctrls = symbol2corenctrl(q.var(test_x[i]).cuda(cuda).v)
            test_cores = test_cores.cpu().data.numpy()
            print(" ".join([revdin[test_cores_ij] for test_cores_ij
                            in test_cores if test_cores_ij != outemb.D["<MASK>"]]))

    freerunner = q.FreeRunner(inparggetter=inparggetter)

    # validation with freerunner
    valid_decoder_cell = q.ModularDecoderCell(decoder_core, decoder_top)
    valid_decoder_cell.set_runner(freerunner)
    valid_decoder = valid_decoder_cell.to_decoder()

    if useattention:
        valid_encdec = EncDecAtt(encoder, decoder)
    else:
        valid_encdec = EncDec(encoder, decoder)


    losses = q.lossarray(q.SeqCrossEntropyLoss(ignore_index=0),
                         #q.SeqNLLLoss(ignore_index=0),
                         q.MacroBLEU(ignore_index=0, predcut=BFTreePredCutter(tracker)),
                         q.SeqAccuracy(ignore_index=0))

    validlosses = q.lossarray(#q.SeqCrossEntropyLoss(ignore_index=0),
                              q.MacroBLEU(ignore_index=0, predcut=BFTreePredCutter(tracker)),
                              q.SeqAccuracy(ignore_index=0),
                              TreeAccuracy(ignore_index=0, treeparser=lambda x: Node.parse(tracker.pp(x))))

    logger.update_settings(optimizer="rmsprop")
    optim = torch.optim.RMSprop(q.paramgroups_of(encdec), lr=lr, weight_decay=wreg)

    lrsched = torch.optim.lr_scheduler.ExponentialLR(optim, lrdecay)

    q.train(encdec).train_on(train_loader, losses) \
        .optimizer(optim) \
        .clip_grad_norm(gradnorm) \
        .set_batch_transformer(inbtf) \
        .valid_with(valid_encdec).valid_on(valid_loader, validlosses) \
        .cuda(cuda) \
        .hook(logger) \
        .hook(lrsched) \
        .train(epochs)

    logger.update_settings(completed=True)

    results = q.test(valid_encdec).on(test_loader, validlosses) \
        .set_batch_transformer(inbtf) \
        .cuda(cuda).run()



def make_oracle(tracker, symbols2cores, symbols2ctrl, mode=OPT_ORACLEMODE, withannotations=False, cuda=False, ttt=None,
                trees=None):      # this line of args is for testing
    if ttt is None:
        ttt = q.ticktock("oraclemaker")

    symbols2cores_pt = q.var(symbols2cores).cuda(cuda).v
    symbols2ctrl_pt = q.var(symbols2ctrl).cuda(cuda).v

    def outsym2insymandctrl(x):
        cores = torch.index_select(symbols2cores_pt, 0, x)
        ctrls = torch.index_select(symbols2ctrl_pt, 0, x)
        return cores, ctrls

    inparggetter = lambda x: x if withannotations else outsym2insymandctrl

    print("oracle mode: {}".format(mode))
    oracle = q.DynamicOracleRunner(tracker=tracker,
                                   inparggetter=inparggetter,
                                   mode=mode,
                                   explore=0.)

    if _opt_test:
        ttt.tick("testing from out sym to core&ctrl")
        testtokens = "<MASK> <START> <ROOT> <NONE> <RARE> <RARE>*LS <RARE>*NC*LS major:t major:t*LS major:t*NC major:t*NC*LS $0 $0*LS $0*NC $0*NC*LS".split()
        testidxs = [tracker.D[xe] for xe in testtokens]
        testidxs_pt = q.var(np.asarray(testidxs)).cuda(cuda).v
        rinpdic = {v: k for k, v in tracker.D.items()}
        _out = outsym2insymandctrl(testidxs_pt)
        testcoreidxs_pt, testctrls_pt = _out
        testcoreidxs = list(testcoreidxs_pt.cpu().data.numpy())
        testcoretokens = [rinpdic[xe] for xe in testcoreidxs]
        expcoretokens = "<MASK> <START> <ROOT> <NONE> <RARE> <RARE> <RARE> major:t major:t major:t major:t $0 $0 $0 $0".split()
        assert (expcoretokens == testcoretokens)
        testctrlids = list(testctrls_pt.cpu().data.numpy())
        expected_ctrl_ids = [0, 2, 3, 4, 1, 3, 4, 1, 3, 2, 4, 1, 3, 2, 4]
        assert (expected_ctrl_ids == testctrlids)
        ttt.tock("tested")

    if _opt_test:  # TEST with dummy core decoder   # TODO: might have to change when without annotation
        ttt.tick("testing with dummy decoder")
        outemb = q.WordEmb(50, worddic=tracker.D_in)
        linout = q.WordLinout(50, worddic=tracker.D)
        oracle.inparggetter = outsym2insymandctrl

        class DummyCore(q.DecoderCore):
            def __init__(self, *x, **kw):
                super(DummyCore, self).__init__(*x, **kw)

            def forward(self, y_tm1, ctrl_tm1, ctx_t=None, t=None, outmask_t=None, **kw):
                cell_out = q.var(torch.randn(y_tm1.size(0), 50)).cuda(y_tm1).v
                return cell_out, {"t": t, "x_t_emb": outemb(y_tm1), "ctx_t": ctx_t, "mask": outmask_t}

            def reset_state(self):
                pass

        test_decoder_core = DummyCore(outemb)
        test_decoder_top = q.DecoderTop(q.wire((0, 0)), linout)
        test_decoder = q.ModularDecoderCell(test_decoder_core, test_decoder_top)
        test_decoder.set_runner(oracle)
        test_decoder = test_decoder.to_decoder()

        test_eids = q.var(np.arange(80, 84)).cuda(cuda).v
        test_start_symbols = q.var(np.ones((4,), dtype="int64") * linout.D["<ROOT>"]).cuda(cuda).v
        if cuda:
            test_decoder.cuda()
        out = test_decoder(test_start_symbols, eids=test_eids, maxtime=100)
        # get gold (! last timestep in out has no gold --> ensure one too many decoding timesteps)
        golds = oracle.goldacc
        seqs = oracle.seqacc
        golds = torch.stack(golds, 1)
        seqs = torch.stack(seqs, 1)
        out = out[:, :-1, :]
        # test if gold tree linearization produces gold tree
        for i in range(len(test_eids)):
            goldtree = Node.parse(tracker.pp(golds[i].cpu().data.numpy()))
            predtree = Node.parse(tracker.pp(seqs[i].cpu().data.numpy()))
            exptree = trees[test_eids.cpu().data.numpy()[i]]
            assert (exptree == goldtree)
            assert (exptree == predtree)

        # try loss
        loss = q.SeqCrossEntropyLoss(ignore_index=0)
        l = loss(out, golds)
        ttt.msg("loss: {}".format(l.data[0]))
        l.backward()
        params = q.params_of(test_decoder)
        core_params = q.params_of(test_decoder_core)
        linout_params = q.params_of(linout)
        outemb_params = q.params_of(outemb)
        for param in params:
            topass = False
            for outemb_param in outemb_params:
                if param.size() == outemb_param.size() and param is outemb_param:
                    topass = True
                    break
            if topass:
                pass
            else:
                assert (param.grad is not None)
                assert (param.grad.norm().data[0] > 0)
        ttt.tock("tested with dummy decoder")
        oracle.inparggetter = inparggetter

    return oracle


class BFTreePredCutter(object):
    def __init__(self, tracker, **kw):
        super(BFTreePredCutter, self).__init__()
        self.tracker = tracker

    def __call__(self, a, ignore_index=None):   # a: 2D of ints
        for i in range(len(a)):
            ass = self.tracker.pp(a[i])
            try:
                _, (tokens, remainder) = Node.parse(ass, _ret_remainder=True)
                a[i, -len(remainder):] = ignore_index
            except Exception as e:
                a[i, :] = ignore_index
        return a


def run_seq2seq_oracle(lr=OPT_LR, lrdecay=OPT_LR_DECAY, epochs=OPT_EPOCHS, batsize=OPT_BATSIZE,
                             wreg=OPT_WREG, dropout=OPT_DROPOUT, gradnorm=OPT_GRADNORM,
                             embdim=-1, oraclemode=OPT_ORACLEMODE,
                             inpembdim=OPT_INPEMBDIM, outembdim=OPT_OUTEMBDIM, innerdim=OPT_INNERDIM,
                             cuda=False, gpu=0, splitseed=1,
                             validontest=False):
    settings = locals().copy()
    logger = q.Logger(prefix="geoquery_s2s_oracle")
    logger.save_settings(**settings)
    logger.update_settings(version="2")
    logger.update_settings(completed=False)

    if validontest:
        print("VALIDATING ON TEST: WONG !!!")
    print("SEQ2SEQ ORACLE")
    if cuda:    torch.cuda.set_device(gpu)
    tt = q.ticktock("script")
    ttt = q.ticktock("test")
    ism, tracker, numtrain = load_data_trees(reverse_input=False)
    eids = np.arange(0, len(ism), dtype="int64")
    psm = q.StringMatrix()
    psm.set_dictionary(tracker.D)
    psm.tokenize = lambda x: x.split()
    for tree in tracker.trackables:
        treestring = tree.pp(arbitrary=False, _remove_order=True)
        assert(Node.parse(treestring) == tree)
        psm.add(treestring)
    psm.finalize()
    print(ism[0])
    print(psm[0])

    # region MODEL --------------------------------
    # EMBEDDINGS ----------------------------------
    if embdim > 0:
        tt.msg("embdim overrides inpembdim and outembdim")
        inpembdim, outembdim = embdim, embdim

    linoutdim = innerdim + innerdim

    inpemb = q.WordEmb(inpembdim, worddic=ism.D)  # TODO glove embeddings

    outemb, linout, symbols2cores, symbols2ctrl\
        = make_outvecs(outembdim, linoutdim, grouptracker=tracker, tie_weights=False, ttt=ttt)

    # TODO: try struct linout

    # region ENCODER -------------------------------------
    # encoder = make_encoder(inpemb, inpembdim, innerdim//2, dropout, ttt=ttt)/
    encoderstack = q.RecStack(
        q.wire((0, 0), mask_t=(0, {"mask_t"}), t=(0, {"t"})),
        q.LSTMCell(inpembdim, innerdim, dropout_in=dropout, dropout_rec=None),
    ).to_layer().return_final().return_mask().reverse()
    encoder = q.RecurrentStack(
        inpemb,
        encoderstack,
    )
    # endregion

    # region ORACLE --------------------------------------
    oracle = make_oracle(tracker, symbols2cores, symbols2ctrl, mode=oraclemode, ttt=ttt, withannotations=True,
                         trees=tracker.trackables)
    # endregion

    # region DECODER -------------------------------------
    # test
    # testencret = encoder(q.var(trainmats[0][:5]).v)

    layers = (q.LSTMCell(outembdim, innerdim, dropout_in=dropout, dropout_rec=None),
              )

    decoder_top = q.AttentionContextDecoderTop(q.Attention().dot_gen(),
                                               q.Dropout(dropout),
                                               linout, ctx2out=False)

    decoder_core = q.DecoderCore(outemb, *layers)
    decoder_cell = q.ModularDecoderCell(decoder_core, decoder_top)
    decoder_cell.set_runner(oracle)
    decoder = decoder_cell.to_decoder()
    # endregion

    # region ENCDEC ---------------------------------------
    class EncDecAtt(torch.nn.Module):
        def __init__(self, _encoder, _decoder, maxtime=None, **kwargs):
            super(EncDecAtt, self).__init__(**kwargs)
            self.encoder = _encoder
            self.decoder = _decoder
            self.maxtime = maxtime

        def forward(self, inpseq, decstarts, eids=None, maxtime=None):
            final_encoding, all_encoding, mask = self.encoder(inpseq)
            maxtime = self.maxtime if maxtime is None else maxtime
            # self.decoder.set_init_states(None, final_encoding)
            decoding = self.decoder(decstarts,
                                    ctx=all_encoding,
                                    ctx_0=final_encoding,
                                    ctxmask=mask,
                                    eids=eids,
                                    maxtime=maxtime)
            return decoding

    encdec = EncDecAtt(encoder, decoder)

    if _opt_test:
        ttt.tick("testing whole thing dry run")
        test_eids = q.var(np.arange(80, 83)).cuda(cuda).v
        test_start_symbols = q.var(np.ones((3,), dtype="int64") * linout.D["<START>"]).cuda(cuda).v
        test_inpseqs = q.var(ism.matrix[:3]).cuda(cuda).v
        if cuda:
            encdec.cuda()
        test_encdec_output = encdec(test_inpseqs, test_start_symbols, eids=test_eids, maxtime=100)
        out = test_encdec_output
        golds = oracle.goldacc
        seqs = oracle.seqacc
        golds = torch.stack(golds, 1)
        seqs = torch.stack(seqs, 1)
        out = out[:, :-1, :]
        # test if gold tree linearization produces gold tree
        for i in range(len(out)):
            goldtree = Node.parse(tracker.pp(golds[i].cpu().data.numpy()))
            predtree = Node.parse(tracker.pp(seqs[i].cpu().data.numpy()))
            exptree = tracker.trackables[test_eids.cpu().data[i]]
            assert (exptree == goldtree)
            assert (exptree == predtree)
        ttt.tock("tested whole dryrun").tick()

        loss = q.SeqCrossEntropyLoss(ignore_index=0)
        lossvalue = loss(out, golds)
        ttt.msg("value of Seq CE loss: {}".format(lossvalue))
        encdec.zero_grad()
        lossvalue.backward()
        ttt.msg("backward done")
        params = q.params_of(encdec)
        for param in params:
            assert (param.grad is not None)
            assert (param.grad.norm().data[0] > 0)
            # print(tuple(param.size()), param.grad.norm().data[0])
        ttt.tock("all gradients non-zero")

        loss = TreeAccuracy(treeparser=lambda x: Node.parse(tracker.pp(x)))
        lossvalue = loss(out, golds)
        ttt.msg("value of predicted Tree Accuracy: {}".format(lossvalue.data[0]))
        dummyout = q.var(torch.zeros(out.size())).cuda(cuda).v
        dummyout.scatter_(2, seqs.unsqueeze(2), 1)
        lossvalue = loss(dummyout, golds)
        ttt.msg("value of fed Tree Accuracy: {}".format(lossvalue.data[0]))
        loss = q.SeqAccuracy()
        lossvalue = loss(dummyout, golds)
        ttt.msg("value of SeqAccuracy on fed prediction: {}".format(lossvalue.data[0]))
        encdec.cpu()

        encdec.eval()
        assert (not oracle.training)
        ttt.msg("oracle switched to eval")
        encdec.train()
        assert (oracle.training)
        ttt.msg("oracle switched to training")
    # endregion
    # endregion

    # region DATA LOADING -------------------------
    startid = outemb.D["<ROOT>"]
    tt.msg("using startid {} ({}) from outemb".format(startid, "<ROOT>"))
    starts = np.ones((len(ism.matrix,)), dtype="int64") * startid

    traindata = [ism.matrix[:numtrain], starts[:numtrain], eids[:numtrain], psm.matrix[:numtrain], eids[:numtrain]]
    testdata = [ism.matrix[numtrain:], starts[numtrain:], eids[numtrain:], psm.matrix[numtrain:], eids[numtrain:]]

    if validontest:
        validdata = testdata
    else:
        splitseed = True if splitseed == 1 else False if splitseed == 0 else splitseed
        traindata, validdata = q.split(traindata, random=splitseed)

    train_loader = q.dataload(*[traindata[i] for i in [0, 1, 2, 4]], batch_size=batsize, shuffle=True)
    valid_loader = q.dataload(*[validdata[i] for i in [0, 1, 3]], batch_size=batsize, shuffle=False)
    test_loader = q.dataload(*[testdata[i] for i in [0, 1, 3]], batch_size=batsize, shuffle=False)
    # endregion
    logger.update_settings(completed=False)

    # region TRAINING --------------------
    # region SETTING ---------------------
    # validation with freerunner
    freerunner = q.FreeRunner(inparggetter=oracle.inparggetter)
    valid_decoder_cell = q.ModularDecoderCell(decoder_core, decoder_top)
    valid_decoder_cell.set_runner(freerunner)
    valid_decoder = valid_decoder_cell.to_decoder()
    valid_encdec = EncDecAtt(encoder, valid_decoder, maxtime=50)

    losses = q.lossarray(q.SeqCrossEntropyLoss(ignore_index=0),
                         q.MacroBLEU(ignore_index=0, predcut=BFTreePredCutter(tracker)),
                         q.SeqAccuracy(ignore_index=0),
                         TreeAccuracy(ignore_index=0, treeparser=lambda x: Node.parse(tracker.pp(x))))

    validlosses = q.lossarray(  # q.SeqCrossEntropyLoss(ignore_index=0),
        q.MacroBLEU(ignore_index=0, predcut=BFTreePredCutter(tracker)),
        q.SeqAccuracy(ignore_index=0),
        TreeAccuracy(ignore_index=0, treeparser=lambda x: Node.parse(tracker.pp(x))))

    logger.update_settings(optimizer="rmsprop")
    optim = torch.optim.RMSprop(q.paramgroups_of(encdec), lr=lr, weight_decay=wreg)

    lrsched = torch.optim.lr_scheduler.ExponentialLR(optim, lrdecay)

    out_btf = lambda _out: _out[:, :-1, :]
    gold_btf = lambda _eids: torch.stack(oracle.goldacc, 1)
    valid_gold_btf = lambda x: x
    # endregion

    # region TRAIN -----------------------
    q.train(encdec).train_on(train_loader, losses) \
        .optimizer(optim) \
        .clip_grad_norm(gradnorm) \
        .set_batch_transformer(None, out_btf, gold_btf) \
        .valid_with(valid_encdec).valid_on(valid_loader, validlosses) \
        .set_valid_batch_transformer(None, out_btf, valid_gold_btf) \
        .cuda(cuda) \
        .hook(logger) \
        .hook(lrsched, verbose=False) \
        .train(epochs)

    logger.update_settings(completed=True)

    results = q.test(valid_encdec).on(test_loader, validlosses) \
        .set_batch_transformer(lambda x, y: (x, y[:, :-1], y[:, 1:])) \
        .cuda(cuda).run()
    # endregion
    # endregion


def run_seq2seq_tf(lr=OPT_LR, lrdecay=OPT_LR_DECAY, epochs=OPT_EPOCHS, batsize=OPT_BATSIZE,
                     wreg=OPT_WREG, dropout=OPT_DROPOUT, gradnorm=OPT_GRADNORM,
                     embdim=-1,
                     inpembdim=OPT_INPEMBDIM, outembdim=OPT_OUTEMBDIM, innerdim=OPT_INNERDIM,
                     cuda=False, gpu=0, splitseed=1,
                     validontest=False):
    settings = locals().copy()
    logger = q.Logger(prefix="geoquery_s2s_tf")
    logger.save_settings(**settings)
    logger.update_settings(version="2")
    logger.update_settings(completed=False)

    if validontest:
        print("VALIDATING ON TEST: WONG !!!")
    print("SEQ2SEQ TF")
    if cuda:    torch.cuda.set_device(gpu)
    tt = q.ticktock("script")
    ttt = q.ticktock("test")
    ism, tracker, numtrain = load_data_trees(reverse_input=True)
    eids = np.arange(0, len(ism), dtype="int64")
    psm = q.StringMatrix()
    psm.set_dictionary(tracker.D)
    psm.tokenize = lambda x: x.split()
    for tree in tracker.trackables:
        treestring = tree.pp(arbitrary=False, _remove_order=True)
        assert (Node.parse(treestring) == tree)
        psm.add("<ROOT> " + treestring)
    psm.finalize()
    print(ism[0])
    print(psm[0])

    # region MODEL --------------------------------
    # EMBEDDINGS ----------------------------------
    if embdim > 0:
        tt.msg("embdim overrides inpembdim and outembdim")
        inpembdim, outembdim = embdim, embdim

    linoutdim = innerdim + innerdim

    inpemb = q.WordEmb(inpembdim, worddic=ism.D)  # TODO glove embeddings

    outemb, linout, symbols2cores, symbols2ctrl \
        = make_outvecs(outembdim, linoutdim, grouptracker=tracker, tie_weights=False, ttt=ttt)

    # TODO: try struct linout

    # region ENCODER -------------------------------------
    # encoder = make_encoder(inpemb, inpembdim, innerdim//2, dropout, ttt=ttt)/
    encoderstack = q.RecStack(
        q.wire((0, 0), mask_t=(0, {"mask_t"}), t=(0, {"t"})),
        q.LSTMCell(inpembdim, innerdim, dropout_in=dropout, dropout_rec=None),
    ).to_layer().return_final().return_mask()
    encoder = q.RecurrentStack(
        inpemb,
        encoderstack,
    )
    # endregion

    # region DECODER -------------------------------------
    # test
    # testencret = encoder(q.var(trainmats[0][:5]).v)

    layers = (q.LSTMCell(outembdim, innerdim, dropout_in=dropout, dropout_rec=None),
              )

    decoder_top = q.AttentionContextDecoderTop(q.Attention().dot_gen(),
                                               q.Dropout(dropout),
                                               linout, ctx2out=False)

    decoder_core = q.DecoderCore(outemb, *layers)
    decoder_cell = q.ModularDecoderCell(decoder_core, decoder_top)
    decoder_cell.set_runner(q.TeacherForcer())
    decoder = decoder_cell.to_decoder()


    # endregion

    # region ENCDEC ---------------------------------------
    class EncDecAtt(torch.nn.Module):
        def __init__(self, _encoder, _decoder, maxtime=None, **kwargs):
            super(EncDecAtt, self).__init__(**kwargs)
            self.encoder = _encoder
            self.decoder = _decoder
            self.maxtime = maxtime

        def forward(self, inpseq, outseq):
            final_encoding, all_encoding, mask = self.encoder(inpseq)
            # self.decoder.set_init_states(None, final_encoding)
            encoderstates = self.encoder.layers[1].cell.get_states(inpseq.size(0))
            initstates = [self.dropout(initstate) for initstate in encoderstates]
            self.decoder.set_init_states(*initstates)
            decoding = self.decoder(outseq,
                                    ctx=all_encoding,
                                    ctx_0=final_encoding,
                                    ctxmask=mask)
            return decoding

    encdec = EncDecAtt(encoder, decoder)
    # endregion
    # endregion

    # region DATA LOADING
    traindata = [ism.matrix[:numtrain], psm.matrix[:numtrain, :-1], psm.matrix[:numtrain, 1:]]
    testdata = [ism.matrix[numtrain:], psm.matrix[numtrain:, :-1], psm.matrix[numtrain:, 1:]]

    if validontest:
        validdata = testdata
    else:
        splitseed = True if splitseed == 1 else False if splitseed == 0 else splitseed
        traindata, validdata = q.split(traindata, random=splitseed)

    train_loader = q.dataload(*traindata, batch_size=batsize, shuffle=True)
    valid_loader = q.dataload(*validdata, batch_size=batsize, shuffle=False)
    test_loader = q.dataload(*testdata, batch_size=batsize, shuffle=False)
    # endregion

    freerunner = q.FreeRunner()
    valid_decoder_cell = q.ModularDecoderCell(decoder_core, decoder_top)
    valid_decoder_cell.set_runner(freerunner)
    valid_decoder = valid_decoder_cell.to_decoder()
    valid_encdec = EncDecAtt(encoder, valid_decoder)

    losses = q.lossarray(q.SeqCrossEntropyLoss(ignore_index=0),
                         # q.SeqNLLLoss(ignore_index=0),
                         q.MacroBLEU(ignore_index=0, predcut=BFTreePredCutter(tracker)),
                         q.SeqAccuracy(ignore_index=0),
                         TreeAccuracy(ignore_index=0, treeparser=lambda x: Node.parse(tracker.pp(x))))


    validlosses = q.lossarray(  # q.SeqCrossEntropyLoss(ignore_index=0),
        q.MacroBLEU(ignore_index=0, predcut=BFTreePredCutter(tracker)),
        q.SeqAccuracy(ignore_index=0),
        TreeAccuracy(ignore_index=0, treeparser=lambda x: Node.parse(tracker.pp(x))))

    logger.update_settings(optimizer="rmsprop")
    optim = torch.optim.RMSprop(q.paramgroups_of(encdec), lr=lr, weight_decay=wreg)

    lrsched = torch.optim.lr_scheduler.ExponentialLR(optim, lrdecay)

    q.train(encdec).train_on(train_loader, losses) \
        .optimizer(optim) \
        .clip_grad_norm(gradnorm) \
        .valid_with(valid_encdec).valid_on(valid_loader, validlosses) \
        .cuda(cuda) \
        .hook(logger) \
        .hook(lrsched, verbose=False) \
        .train(epochs)

    logger.update_settings(completed=True)

    results = q.test(valid_encdec).on(test_loader, validlosses) \
        .cuda(cuda).run()


def run_noisy_parse():
    treen = parse_query_tree("( lambda $0 e ( and ( state:t $0 ) ( and ( next_to:t $0 s0 ) ( next_to:t $0 s0 ) ) ) ) ( ) ) ) ) ) )")
    tree = parse_query_tree("( lambda $0 e ( and ( state:t $0 ) ( and ( next_to:t $0 s0 ) ( next_to:t $0 s0 ) ) ) )")
    print(tree.pp())
    print(treen.pp())
    print(treen.pptree())
    assert(treen == tree)
    p = Node.parse(tree.pp())
    print(p)
    p = p.pp() + " s0#2*NC*LS $0#1*NC s0#2*NC*LS"
    print(p)
    rp, rem = Node.parse(p, _ret_remainder=True)
    print(rp)
    print(rem)
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
    # run_noisy_parse()
    # load_data_trees()
    # run_some_tests()
    # q.argprun(run_seq2seq_reproduction)
    # q.argprun(run_seq2seq_oracle)
    # q.argprun(run_seq2tree_tf)
    q.argprun(run_seq2seq_tf)