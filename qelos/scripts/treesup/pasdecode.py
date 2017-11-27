import torch
import qelos as q
from qelos.scripts.treesup.pastrees import GroupTracker, generate_random_trees, Tree, BinaryTree, UnaryTree, LeafTree
import numpy as np
from collections import OrderedDict
import re


OPT_LR = 0.1
OPT_BATSIZE = 128
OPT_GRADNORM = 5.
OPT_EPOCHS = 50
OPT_NUMEX = 1000
OPT_INPEMBDIM = 50
OPT_OUTEMBDIM = 50
OPT_LINOUTDIM = 50
OPT_JOINT_LINOUT_MODE = "sum"
OPT_EXPLORE = 0.
OPT_DROPOUT = 0.3
OPT_ENCDIM = 100
OPT_DECDIM = 100

_opt_test = True
_tree_gen_seed = 1234


def run_seq2seq_teacher_forced(lr=OPT_LR,
                               batsize=OPT_BATSIZE,
                               epochs=OPT_EPOCHS,
                               numex=OPT_NUMEX,
                               gradnorm=OPT_GRADNORM,
                               inpembdim=OPT_INPEMBDIM,
                               outembdim=OPT_OUTEMBDIM,
                               linoutdim=OPT_LINOUTDIM,
                               encdim=OPT_ENCDIM,
                               decdim=OPT_DECDIM,
                               dropout=OPT_DROPOUT,
                               cuda=False,
                               gpu=1):
    if cuda:
        torch.cuda.set_device(gpu)
    tt = q.ticktock("script")
    ttt = q.ticktock("test")
    ism, tracker, eids, trees = load_synth_trees(n=numex)
    tt.msg("generated {} synthetic trees".format(ism.matrix.shape[0]))
    osm = q.StringMatrix(indicate_start=True)
    osm.tokenize = lambda x: x.split()
    for tree in trees:
        treestring = tree.pp(with_parentheses=False, arbitrary=True)
        osm.add(treestring)
    osm.finalize()
    if _opt_test:
        allsame = True
        for i in range(len(osm.matrix)):
            itree = Tree.parse(ism[i])
            otree = Tree.parse(osm[i, 1:])
            assert(itree == otree)
            allsame &= ism[i] == osm[i]
        assert(not allsame)
        ttt.msg("trees at output are differently structured from trees at input but are the same trees")

    inpemb = q.WordEmb(inpembdim, worddic=ism.D)
    outemb = q.WordEmb(outembdim, worddic=osm.D)
    linout = q.WordLinout(linoutdim, worddic=osm.D)

    # region initialize encoder
    encoder = q.RecurrentStack(
        inpemb,
        q.wire((-1, 0)),
        q.TimesharedDropout(dropout),
        q.wire((1, 0), mask=(1, 1)),
        q.BidirGRULayer(inpembdim, encdim//2).return_final(True),
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
        test_input_symbols = q.var(np.random.randint(0, 44, (3, 10))).cuda(cuda).v
        test_encoder_output = encoder(test_input_symbols)
        ttt.msg("encoder return {} outputs".format(len(test_encoder_output)))
        ttt.msg("encoder output shapes: {}".format(" ".join([str(test_encoder_output_e.size()) for test_encoder_output_e in test_encoder_output])))
        assert(test_encoder_output[1].size() == (3, 10, encdim * 2))
        assert(test_encoder_output[0].size() == (3, encdim * 2))
        ttt.tock("tested encoder (output shapes)")

    # endregion

    # region make decoder and put in enc/dec
    ctxdim = encdim * 2
    layers = (q.GRUCell(outembdim + ctxdim, decdim),
              q.GRUCell(decdim, linoutdim),)

    decoder_core = q.DecoderCore(outemb, *layers)
    decoder_top = q.StaticContextDecoderTop(linout)
    decoder_cell = q.ModularDecoderCell(decoder_core, decoder_top)
    decoder_cell.set_runner(q.TeacherForcer())
    decoder = decoder_cell.to_decoder()

    # wrap in encdec
    class EncDec(torch.nn.Module):
        def __init__(self, encoder, decoder, **kw):
            super(EncDec, self).__init__(**kw)
            self.encoder = encoder
            self.decoder = decoder

        def forward(self, inpseq, outinpseq):
            final_encoding, all_encoding, mask = self.encoder(inpseq)
            # self.decoder.block.decoder_top.set_ctx(final_encoding)
            decoding = self.decoder(outinpseq, ctx=final_encoding)
            return decoding

    encdec = EncDec(encoder, decoder)

    if _opt_test:
        ttt.tick("testing whole thing dry run")
        test_inpseqs = q.var(ism.matrix[:3]).cuda(cuda).v
        test_outinpseqs = q.var(osm.matrix[:3, :-1]).cuda(cuda).v

        test_encdec_output = encdec(test_inpseqs, test_outinpseqs)

        ttt.tock("tested whole dryrun")
        # TODO loss and gradients
        golds = q.var(osm.matrix[:3, 1:]).cuda(cuda).v
        loss = q.SeqCrossEntropyLoss(ignore_index=0)
        lossvalue = loss(test_encdec_output, golds)
        ttt.msg("value of Seq CE loss: {}".format(lossvalue))
        encdec.zero_grad()
        lossvalue.backward()
        ttt.msg("backward done")
        params = q.params_of(encdec)
        for param in params:
            assert (param.grad is not None)
            assert (param.grad.norm().data[0] > 0)
            print(tuple(param.size()), param.grad.norm().data[0])
        ttt.tock("all gradients non-zero")

    # print(encdec)

    # training
    losses = q.lossarray(q.SeqCrossEntropyLoss(ignore_index=0),
                         q.SeqElemAccuracy(ignore_index=0),
                         q.SeqAccuracy(ignore_index=0),)

    optimizer = torch.optim.Adadelta(q.params_of(encdec), lr=lr)

    traindata, testdata = q.split([ism.matrix, osm.matrix], random=1234)
    traindata, validdata = q.split(traindata, random=1234)

    train_loader = q.dataload(*traindata, batch_size=batsize, shuffle=True)
    valid_loader = q.dataload(*validdata, batch_size=batsize, shuffle=False)
    test_loader = q.dataload(*testdata, batch_size=batsize, shuffle=False)

    q.train(encdec)\
        .train_on(train_loader, losses)\
        .optimizer(optimizer)\
        .clip_grad_norm(gradnorm)\
        .set_batch_transformer(
            lambda inpseq, outseq:
                (inpseq, outseq[:, :-1], outseq[:, 1:]))\
        .valid_on(valid_loader, losses)\
        .cuda(cuda)\
        .train(epochs)

    results = q.test(encdec).on(test_loader, losses)\
        .set_batch_transformer(
            lambda inpseq, outseq:
                (inpseq, outseq[:, :-1], outseq[:, 1:]))\
        .cuda(cuda)\
        .run()


def load_synth_trees(n=1000):
    ism = q.StringMatrix()
    ism.tokenize = lambda x: x.split()
    numtrees = n
    trees = generate_random_trees(numtrees, seed=_tree_gen_seed)
    tracker = GroupTracker(trees)
    for tree in trees:
        treestring = tree.pp(with_parentheses=False, arbitrary=True)
        ism.add(treestring)
    ism.finalize()  # ism provides source sequences
    eids = np.arange(0, len(trees)).astype("int64")

    if _opt_test:  # TEST
        for eid in eids:
            assert (tracker.trackers[eid].root == Tree.parse(ism[eid]))
    return ism, tracker, eids, trees


def run(lr=OPT_LR,
        inpembdim=OPT_INPEMBDIM,
        outembdim=OPT_OUTEMBDIM,
        linoutdim=OPT_LINOUTDIM,
        encdim=OPT_ENCDIM,
        dropout=OPT_DROPOUT,
        linoutjoinmode=OPT_JOINT_LINOUT_MODE,
        explore=OPT_EXPLORE,
        cuda=False,
        ):
    tt = q.ticktock("script")
    ttt = q.ticktock("test")

    ism, tracker, eids, trees = load_synth_trees()

    # region build reps
    inpemb = q.WordEmb(inpembdim, worddic=ism.D)
    outemb = q.WordEmb(outembdim, worddic=tracker.D_in)

    # computed wordlinout for output symbols with topology annotations
    # dictionaries
    symbols_core_dic = OrderedDict()
    symbols_is_last_sibling = np.zeros((len(tracker.D),), dtype="int64")
    for k, v in tracker.D.items():
        ksplits = k.split("*")
        if not ksplits[0] in symbols_core_dic:
            symbols_core_dic[ksplits[0]] = len(symbols_core_dic)
        symbols_is_last_sibling[v] = "LS" in ksplits[1:] or k in "<MASK> <START> <STOP>".split()
    symbols2cores = np.zeros((len(tracker.D),), dtype="int64")
    for symbol in tracker.D:
        symbols2cores[tracker.D[symbol]] = symbols_core_dic[symbol.split("*")[0]]

    if _opt_test:
        ttt.tick("testing dictionaries")
        testtokens = "<MASK> <START> <STOP> <RARE> <RARE>*LS <RARE>*NC*LS BIN0 BIN0*LS UNI1 UNI1*LS LEAF2 LEAF2*LS".split()
        testidxs = np.asarray([tracker.D[xe] for xe in testtokens])
        testcoreidxs = symbols2cores[testidxs]
        expcoretokens = "<MASK> <START> <STOP> <RARE> <RARE> <RARE> BIN0 BIN0 UNI1 UNI1 LEAF2 LEAF2".split()
        rcoredic = {v: k for k, v in symbols_core_dic.items()}
        testcoretokens = [rcoredic[xe] for xe in testcoreidxs]
        assert(testcoretokens == expcoretokens)
        exp_is_last_sibling = [1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
        test_is_last_sibling = symbols_is_last_sibling[testidxs]
        assert(list(test_is_last_sibling) == exp_is_last_sibling)
        ttt.tock("tested dictionaries")

    # linouts
    linoutdim_core, linoutdim_ls = linoutdim, linoutdim

    if linoutjoinmode == "cat":
        linoutdim_core = linoutdim // 10 * 9
    symbols_core_emb = q.WordEmb(linoutdim_core, worddic=symbols_core_dic, no_maskzero=True)

    class JointLinoutComputer(torch.nn.Module):
        def __init__(self, coreemb, linoutdim, mode="sum", **kw):
            super(JointLinoutComputer, self).__init__(**kw)
            self.coreemb, self.mode = coreemb, mode
            lsembdim = linoutdim if mode != "cat" else linoutdim // 10
            self.lsemb = q.WordEmb(lsembdim, worddic={"NOLS": 0, "LS": 1})
            if mode in "muladd".split():
                self.lsemb_aux = q.WordEmb(lsembdim, worddic=self.lsemb.D)
            elif mode in "gate".split():
                self.lsemb_aux = q.Forward(lsembdim, lsembdim, activation="sigmoid", use_bias=False)
            elif mode in "bilinear linear".split():
                dim2 = lsembdim if mode in "bilinear".split() else 2 if mode in "linear".split() else None
                self.lsemb_aux = torch.nn.Bilinear(lsembdim, dim2, lsembdim, bias=False)

        def forward(self, data):
            coreembs, _ = self.coreemb(data[:, 0])
            lsembs, _ = self.lsemb(data[:, 1])
            embs = None
            if self.mode in "sum".split():
                embs = coreembs + lsembs
            elif self.mode in "mul flatgate".split():
                if self.mode in "flatgate".split():
                    lsembs = torch.nn.Sigmoid()(lsembs)
                embs = coreembs * lsembs
            elif self.mode in "cat".split():
                embs = torch.cat([coreembs, lsembs], 1)
            elif self.mode in "muladd".split():
                lsembs_aux = self.lsemb_aux(data[:, 1])
                embs = coreembs * lsembs
                embs = embs * lsembs_aux
            elif self.mode in "gate".split():
                gate = self.lsemb_aux(lsembs)
                embs = coreembs * gate
            elif self.mode in "bilinear linear".split():
                if self.mode in "linear".split():
                    lsembs = torch.stack([1 - data[:, 1], data[:, 1]], 1)
                embs = self.lsemb_aux(coreembs, lsembs)
            return embs

    symbols_linout_computer = JointLinoutComputer(symbols_core_emb, linoutdim, mode=linoutjoinmode)
    symbols_linout_data = np.stack([symbols2cores, symbols_is_last_sibling], axis=1)
    symbols_linout = q.ComputedWordLinout(data=symbols_linout_data,
                                          computer=symbols_linout_computer,
                                          worddic=tracker.D)

    linout = symbols_linout

    if _opt_test:       # TEST
        ttt.tick("testing linouts")
        symbols_linout_computer.lsemb.embedding.weight.data.fill_(0)
        testvecs = torch.autograd.Variable(torch.randn(3, linoutdim))
        test_linout_output = linout(testvecs).data.numpy()
        assert(np.allclose(test_linout_output[:, 3:(test_linout_output.shape[1]-3)//2+3], test_linout_output[:, (test_linout_output.shape[1]-3)//2+3:]))
        symbols_linout_computer.mode = "mul"
        test_linout_output = linout(testvecs).data.numpy()
        assert(np.allclose(test_linout_output, np.zeros_like(test_linout_output)))
        ttt.tock("tested")
        symbols_linout_computer.mode = "sum"
    # endregion

    # region create oracle
    symbols2ctrl = np.zeros_like(symbols2cores)
    for symbol in tracker.D:
        nochildren = symbol[:4] == "LEAF" or symbol in "<STOP> <MASK>".split() \
                     or "NC" in symbol.split("*")[1:]
        haschildren = not nochildren
        hassiblings = not bool(symbols_is_last_sibling[tracker.D[symbol]])     # not "LS" in symbol.split("*")[1:] and not symbol in "<MASK> <START> <STOP>".split()
        ctrl = (1 if hassiblings else 3) if haschildren else (2 if hassiblings else 4) if symbol != "<MASK>" else 0
        symbols2ctrl[tracker.D[symbol]] = ctrl

    symbols2cores_pt = q.var(symbols2cores).cuda(cuda).v
    symbols2ctrl_pt = q.var(symbols2ctrl).cuda(cuda).v

    def outsym2insymandctrl(x):
        cores = torch.index_select(symbols2cores_pt, 0, x)
        ctrls = torch.index_select(symbols2ctrl_pt, 0, x)
        return cores, ctrls

    oracle = q.DynamicOracleRunner(tracker=tracker,
                                   inparggetter=outsym2insymandctrl,
                                   mode="sample",
                                   explore=explore)

    if _opt_test:  # test from out sym to core and ctrl
        ttt.tick("testing from out sym to core&ctrl")
        testtokens = "<MASK> <START> <STOP> <RARE> <RARE>*LS <RARE>*NC*LS BIN0 BIN0*LS UNI1 UNI1*LS LEAF2 LEAF2*LS".split()
        testidxs = [linout.D[xe] for xe in testtokens]
        testidxs_pt = q.var(np.asarray(testidxs)).cuda(cuda).v
        rinpdic = {v: k for k, v in outemb.D.items()}
        testcoreidxs_pt, testctrls_pt = outsym2insymandctrl(testidxs_pt)
        testcoreidxs = list(testcoreidxs_pt.data.numpy())
        testcoretokens = [rinpdic[xe] for xe in testcoreidxs]
        expected_core_tokens = "<MASK> <START> <STOP> <RARE> <RARE> <RARE> BIN0 BIN0 UNI1 UNI1 LEAF2 LEAF2".split()
        assert (expected_core_tokens == testcoretokens)
        testctrlids = list(testctrls_pt.data.numpy())
        expected_ctrl_ids = [0, 3, 4, 1, 3, 4, 1, 3, 1, 3, 2, 4]
        assert (expected_ctrl_ids == testctrlids)
        ttt.tock("tested")

    if _opt_test:  # TEST with dummy core decoder
        ttt.tick("testing with dummy decoder")

        class DummyCore(q.DecoderCore):
            def __init__(self, *x, **kw):
                super(DummyCore, self).__init__(*x, **kw)

            def forward(self, y_tm1, ctrl_tm1, ctx_t=None, t=None, outmask_t=None, **kw):
                cell_out = q.var(torch.randn((y_tm1.size(0), linoutdim))).cuda(y_tm1).v
                return cell_out, {"t": t, "x_t_emb": outemb(y_tm1), "ctx_t": ctx_t, "mask": outmask_t}

            def reset_state(self):
                pass

        test_decoder_core = DummyCore(outemb)
        test_decoder_top = q.DecoderTop(linout)
        test_decoder = q.ModularDecoderCell(test_decoder_core, test_decoder_top)
        test_decoder.set_runner(oracle)
        test_decoder = test_decoder.to_decoder()

        test_eids = q.var(np.arange(0, 3)).cuda(cuda).v
        test_start_symbols = q.var(np.ones((3,), dtype="int64") * linout.D["<START>"]).cuda(cuda).v

        out = test_decoder(test_start_symbols, eids=test_eids, maxtime=100)
        # get gold (! last timestep in out has no gold --> ensure one too many decoding timesteps)
        golds = oracle.goldacc
        seqs = oracle.seqacc
        golds = torch.stack(golds, 1)
        seqs = torch.stack(seqs, 1)
        out = out[:, :-1, :]
        # test if gold tree linearization produces gold tree
        for i in range(len(out)):
            goldtree = Tree.parse(tracker.pp(golds[i].cpu().data.numpy()))
            predtree = Tree.parse(tracker.pp(seqs[i].cpu().data.numpy()))
            exptree = trees[i]
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
    # endregion

    # region initialize encoder
    encoder = q.RecurrentStack(
        inpemb,
        q.wire((-1, 0)),
        q.TimesharedDropout(dropout),
        q.wire((1, 0), mask=(1, 1)),
        q.BidirGRULayer(inpembdim, encdim//2).return_final(True),
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
        test_input_symbols = q.var(np.random.randint(0, 44, (3, 10))).cuda(cuda).v
        test_encoder_output = encoder(test_input_symbols)
        ttt.msg("encoder return {} outputs".format(len(test_encoder_output)))
        ttt.msg("encoder output shapes: {}".format(" ".join([str(test_encoder_output_e.size()) for test_encoder_output_e in test_encoder_output])))
        assert(test_encoder_output[1].size() == (3, 10, encdim * 2))
        assert(test_encoder_output[0].size() == (3, encdim * 2))
        ttt.tock("tested encoder (output shapes)")

    # endregion

    # region make decoder and put in enc/dec
    ctxdim = encdim * 2
    cells = (q.GRUCell(outembdim+ctxdim, linoutdim//2),
             q.GRUCell(outembdim+ctxdim, linoutdim//2),)

    decoder_core = q.TwoStackCell(outemb, cells)
    decoder_top = q.StaticContextDecoderTop(linout)
    decoder_cell = q.ModularDecoderCell(decoder_core, decoder_top)
    decoder_cell.set_runner(oracle)
    decoder = decoder_cell.to_decoder()

    # wrap in encdec
    class EncDec(torch.nn.Module):
        def __init__(self, encoder, decoder, **kw):
            super(EncDec, self).__init__(**kw)
            self.encoder = encoder
            self.decoder = decoder

        def forward(self, inpseq, dec_starts, eids=None, maxtime=None):
            final_encoding, all_encoding, mask = self.encoder(inpseq)
            # self.decoder.block.decoder_top.set_ctx(final_encoding)
            decoding = self.decoder(dec_starts, ctx=final_encoding, eids=eids, maxtime=maxtime)
            return decoding

    encdec = EncDec(encoder, decoder)

    if _opt_test:
        ttt.tick("testing whole thing dry run")
        test_eids = q.var(np.arange(0, 3)).cuda(cuda).v
        test_start_symbols = q.var(np.ones((3,), dtype="int64") * linout.D["<START>"]).cuda(cuda).v
        test_inpseqs = q.var(ism.matrix[:3]).cuda(cuda).v
        test_encdec_output = encdec(test_inpseqs, test_start_symbols, eids=test_eids, maxtime=100)
        out = test_encdec_output
        golds = oracle.goldacc
        seqs = oracle.seqacc
        golds = torch.stack(golds, 1)
        seqs = torch.stack(seqs, 1)
        out = out[:, :-1, :]
        # test if gold tree linearization produces gold tree
        for i in range(len(out)):
            goldtree = Tree.parse(tracker.pp(golds[i].cpu().data.numpy()))
            predtree = Tree.parse(tracker.pp(seqs[i].cpu().data.numpy()))
            exptree = trees[i]
            assert (exptree == goldtree)
            assert (exptree == predtree)
        ttt.tock("tested whole dryrun")
        # TODO loss and gradients
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
            print(tuple(param.size()), param.grad.norm().data[0])
        ttt.tock("all gradients non-zero")

    # endregion



if __name__ == "__main__":
    q.argprun(run_seq2seq_teacher_forced)