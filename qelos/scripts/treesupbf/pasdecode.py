# TODO: port script from DF pasdecode.py
# can be copied with small changes and some cleaning
#
#

import torch
import sys
import qelos as q
from qelos.scripts.treesupbf.trees import GroupTracker, generate_random_trees, Node
import numpy as np
from collections import OrderedDict
import re

# region defaults
OPT_LR = 0.1
OPT_BATSIZE = 128
OPT_GRADNORM = 5.
OPT_EPOCHS = 50
OPT_NUMEX = 1000
OPT_INPEMBDIM = 50
OPT_OUTEMBDIM = 50
OPT_LINOUTDIM = 50
OPT_JOINT_LINOUT_MODE = "sum"
OPT_ORACLE_MODE = "uniform"
OPT_EXPLORE = 0.
OPT_DROPOUT = 0.3
OPT_ENCDIM = 100
OPT_DECDIM = 100
OPT_USEATTENTION = False
OPT_INPLINMODE = "bf"       # "df" or "bf" or "dfpar"
OPT_REMOVE_ANNOTATION = False
# endregion

_opt_test = True
_tree_gen_seed = 1234


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


def load_synth_trees(n=1000, inplin="bf"):      # inplin: "df" or "bf" or "dfpar"
    ism = q.StringMatrix()
    ism.tokenize = lambda x: x.split()
    numtrees = n
    trees = generate_random_trees(numtrees, seed=_tree_gen_seed)
    tracker = GroupTracker(trees)
    for tree in trees:
        if inplin == "df":
            treestring = tree.ppdf(mode="ann", arbitrary=True)
        elif inplin == "dfpar":
            treestring = tree.ppdf(mode="par", arbitrary=True)
        elif inplin == "bf":
            treestring = tree.pp(arbitrary=True)
        else:
            raise q.SumTingWongException("unrecognized mode: {}".format(inplin))
        ism.add(treestring)
    ism.finalize()  # ism provides source sequences
    eids = np.arange(0, len(trees)).astype("int64")

    if _opt_test and inplin == "bf":  # TEST
        for eid in eids:
            try:
                assert (tracker.trackers[eid].root == Node.parse(ism[eid]))
            except AssertionError as e:
                print(tracker.trackers[eid].root.pptree())
                print(Node.parse(ism[eid]).pptree())
                tracker.trackers[eid].root == Node.parse(ism[eid])
                raise e
    return ism, tracker, eids, trees


class TreeAccuracy(q.DiscreteLoss):
    def __init__(self, size_average=True, ignore_index=None,
                 treeparser=None, **kw):
        """ needs a treeparser that transforms sequences of integer ids to tree objects that support equality """
        super(TreeAccuracy, self).__init__(size_average=size_average, ignore_index=ignore_index, **kw)
        self.treeparser = treeparser

    def forward(self, x, gold, mask=None):
        if mask is not None and mask.data[0, 1] > 1:  # batchable sparse
            mask = q.batchablesparse2densemask(mask)
        if mask is not None:
            x = x + torch.log(mask.float())
        ignoremask = self._get_ignore_mask(gold)
        maxes, best = torch.max(x, 2)
        same = torch.ByteTensor(best.size(0))
        same.fill_(False)
        for i in range(best.size(0)):
            try:
                best_tree = self.treeparser(best[i].cpu().data.numpy())
            except Exception as e:
                best_tree = None
            gold_tree = self.treeparser(gold[i].cpu().data.numpy())
            same[i] = best_tree is not None and best_tree == gold_tree
        same = q.var(same).cuda(x).v
        acc = torch.sum(same.float())
        total = float(same.size(0))
        if self.size_average:
            acc = acc / total
        # if ignoremask is not None:
        #     same.data = same.data | ~ ignoremask.data
        return acc


def make_computed_linout(outD, inpD, linoutdim, linoutjoinmode, ttt=None):
    ttt = q.ticktock("linout test") if ttt is None else ttt
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

    # corectrl2symbols = np.zeros((len(symbols_core_dic), 4), dtype="int64")
    # for specialsym in "<MASK> <STOP> <START> <NONE> <ROOT>".split():
    #     corectrl2symbols[symbols_core_dic[specialsym], :] = symbols_core_dic[specialsym]
    # for k, v in outD.items():
    #     _a = symbols_core_dic[k.split("*")[0]]
    #     _b = symbols2ctrl[v]
    #     corectrl2symbols[_a, _b] = v

    symbols2cores = np.zeros((len(outD),), dtype="int64")
    for symbol in outD:
        symbols2cores[outD[symbol]] = inpD[symbol.split("*")[0]]

    if _opt_test:
        ttt.tick("testing dictionaries")
        testtokens = "<MASK> <START> <ROOT> <NONE> <RARE> <RARE>*LS <RARE>*NC*LS N0 N0*LS N0*NC N0*NC*LS N85 N85*LS N85*NC N85*NC*LS".split()
        testidxs = np.asarray([outD[xe] for xe in testtokens])
        testcoreidxs = symbols2cores[testidxs]
        expcoretokens = "<MASK> <START> <ROOT> <NONE> <RARE> <RARE> <RARE> N0 N0 N0 N0 N85 N85 N85 N85".split()
        rcoredic = {v: k for k, v in inpD.items()}
        testcoretokens = [rcoredic[xe] for xe in testcoreidxs]
        assert (testcoretokens == expcoretokens)
        exp_is_last = [1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        test_is_last = symbols_is_last[testidxs]
        assert (list(test_is_last) == exp_is_last)
        exp_is_leaf = [1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1]
        test_is_leaf = symbols_is_leaf[testidxs]
        assert (list(test_is_leaf) == exp_is_leaf)
        ttt.tock("tested dictionaries")

    # linouts
    linoutdim_core, linoutdim_ls = linoutdim, linoutdim

    if linoutjoinmode == "cat":
        linoutdim_core = linoutdim // 10 * 9
    symbols_core_emb = q.WordEmb(linoutdim_core, worddic=inpD, no_maskzero=True)

    class JointLinoutComputer(torch.nn.Module):
        def __init__(self, coreemb, linoutdim, mode="sum", **kw):
            super(JointLinoutComputer, self).__init__(**kw)
            self.coreemb, self.mode = coreemb, mode
            lsembdim = linoutdim if mode != "cat" else linoutdim // 10
            self.lsemb = q.WordEmb(lsembdim, worddic={"<MASK>": 0, "A": 1, "LS": 2, "NC": 3, "NCLS": 4})
            if mode in "muladd".split():
                self.lsemb_aux = q.WordEmb(lsembdim, worddic=self.lsemb.D)
            elif mode in "gate".split():
                self.lsemb_aux = q.Forward(lsembdim, lsembdim, activation="sigmoid", use_bias=False)
            elif mode in "bilinear linear".split():     # TODO: fix for the added NC
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
    symbols_linout_data = np.stack([symbols2cores, symbols2ctrl], axis=1)
    symbols_linout = q.ComputedWordLinout(data=symbols_linout_data,
                                          computer=symbols_linout_computer,
                                          worddic=outD)

    linout = symbols_linout

    if _opt_test:  # TEST
        ttt.tick("testing linouts")
        symbols_linout_computer.lsemb.embedding.weight.data.fill_(0)        # TODO: !! might have influence on training
        testvecs = torch.autograd.Variable(torch.randn(3, linoutdim))
        test_linout_output = linout(testvecs).data.numpy()
        assert (np.allclose(test_linout_output[:, 5:(test_linout_output.shape[1] - 5) // 2 + 5],
                            test_linout_output[:, (test_linout_output.shape[1] - 5) // 2 + 5:],
                            atol=1e-4))
        symbols_linout_computer.mode = "mul"
        test_linout_output = linout(testvecs).data.numpy()
        assert (np.allclose(test_linout_output, np.zeros_like(test_linout_output)))
        ttt.tock("tested")
        symbols_linout_computer.mode = "sum"
        # endregion
    return linout, symbols2cores, symbols2ctrl


def make_oracle(tracker, symbols2cores, symbols2ctrl, explore, cuda=False, mode=OPT_ORACLE_MODE,
                ttt=None, linout=None, outemb=None, trees=None, linoutdim=None):
    if ttt is None:
        ttt = q.ticktock("oraclemaker")
    # symbols2ctrl += 1
    # symbols2ctrl[tracker.D["<MASK>"]] = 0

    symbols2cores_pt = q.var(symbols2cores).cuda(cuda).v
    symbols2ctrl_pt = q.var(symbols2ctrl).cuda(cuda).v

    def outsym2insymandctrl(x):
        cores = torch.index_select(symbols2cores_pt, 0, x)
        ctrls = torch.index_select(symbols2ctrl_pt, 0, x)
        return cores, ctrls

    print("oracle mode: {}".format(mode))

    oracle = q.DynamicOracleRunner(tracker=tracker,
                                   inparggetter=outsym2insymandctrl,
                                   mode=mode,
                                   explore=explore)

    if _opt_test:  # test from out sym to core and ctrl
        ttt.tick("testing from out sym to core&ctrl")
        testtokens = "<MASK> <START> <ROOT> <NONE> <RARE> <RARE>*LS <RARE>*NC*LS N0 N0*LS N0*NC N0*NC*LS N85 N85*LS N85*NC N85*NC*LS".split()
        testidxs = [linout.D[xe] for xe in testtokens]
        testidxs_pt = q.var(np.asarray(testidxs)).cuda(cuda).v
        rinpdic = {v: k for k, v in outemb.D.items()}
        _out = outsym2insymandctrl(testidxs_pt)
        testcoreidxs_pt, testctrls_pt = _out
        testcoreidxs = list(testcoreidxs_pt.cpu().data.numpy())
        testcoretokens = [rinpdic[xe] for xe in testcoreidxs]
        expcoretokens = "<MASK> <START> <ROOT> <NONE> <RARE> <RARE> <RARE> N0 N0 N0 N0 N85 N85 N85 N85".split()
        assert (expcoretokens == testcoretokens)
        testctrlids = list(testctrls_pt.cpu().data.numpy())
        expected_ctrl_ids = [0, 2, 3, 4, 1, 3, 4, 1, 3, 2, 4, 1, 3, 2, 4]
        assert (expected_ctrl_ids == testctrlids)
        ttt.tock("tested")

    if _opt_test:  # TEST with dummy core decoder
        ttt.tick("testing with dummy decoder")

        class DummyCore(q.DecoderCore):
            def __init__(self, *x, **kw):
                super(DummyCore, self).__init__(*x, **kw)

            def forward(self, y_tm1, ctrl_tm1, ctx_t=None, t=None, outmask_t=None, **kw):
                cell_out = q.var(torch.randn(y_tm1.size(0), linoutdim)).cuda(y_tm1).v
                return cell_out, {"t": t, "x_t_emb": outemb(y_tm1), "ctx_t": ctx_t, "mask": outmask_t}

            def reset_state(self):
                pass

        test_decoder_core = DummyCore(outemb)
        test_decoder_top = q.DecoderTop(q.wire((0,0)), linout)
        test_decoder = q.ModularDecoderCell(test_decoder_core, test_decoder_top)
        test_decoder.set_runner(oracle)
        test_decoder = test_decoder.to_decoder()

        test_eids = q.var(np.arange(0, 3)).cuda(cuda).v
        test_start_symbols = q.var(np.ones((3,), dtype="int64") * linout.D["<ROOT>"]).cuda(cuda).v
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
        for i in range(len(out)):
            goldtree = Node.parse(tracker.pp(golds[i].cpu().data.numpy()))
            predtree = Node.parse(tracker.pp(seqs[i].cpu().data.numpy()))
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

    return oracle


def run(lr=OPT_LR,
        inpembdim=OPT_INPEMBDIM,
        outembdim=OPT_OUTEMBDIM,
        linoutdim=OPT_LINOUTDIM,
        encdim=OPT_ENCDIM,
        decdim=OPT_DECDIM,
        dropout=OPT_DROPOUT,
        linoutjoinmode=OPT_JOINT_LINOUT_MODE,
        oraclemode=OPT_ORACLE_MODE,
        explore=OPT_EXPLORE,
        cuda=False,
        ):
    tt = q.ticktock("script")
    ttt = q.ticktock("test")

    ism, tracker, eids, trees \
        = load_synth_trees()

    # region build reps
    inpemb = q.WordEmb(inpembdim, worddic=ism.D)
    outemb = q.WordEmb(outembdim, worddic=tracker.D_in)

    linout, symbols2cores, symbols_is_last_sibling \
        = make_computed_linout(tracker.D, linoutdim, linoutjoinmode, ttt=ttt)

    oracle \
        = make_oracle(tracker, symbols2cores, symbols_is_last_sibling, explore, cuda=cuda, mode=oraclemode,
                      ttt=ttt, linout=linout, outemb=outemb, linoutdim=linoutdim, trees=trees)

    encoder \
        = make_encoder(inpemb, inpembdim, encdim, dropout, ttt=ttt)

    # region make decoder and put in enc/dec
    ctxdim = encdim * 2
    cells = (q.RecStack(
                q.GRUCell(outembdim+ctxdim, decdim),
                q.GRUCell(decdim, linoutdim//2)),
             q.RecStack(
                 q.GRUCell(outembdim + ctxdim, decdim),
                 q.GRUCell(decdim, linoutdim // 2)),)

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
            print(tuple(param.size()), param.grad.norm().data[0])
        ttt.tock("all gradients non-zero")

        loss = TreeAccuracy(treeparser=lambda x: Tree.parse(tracker.pp(x)))
        lossvalue = loss(out, golds)
        ttt.msg("value of predicted Tree Accuracy: {}".format(lossvalue))
        dummyout = q.var(torch.zeros(out.size())).cuda(cuda).v
        dummyout.scatter_(2, seqs.unsqueeze(2), 1)
        lossvalue = loss(dummyout, golds)
        ttt.msg("value of dummy predicted Tree Accuracy: {}".format(lossvalue))
        loss = q.SeqAccuracy()
        lossvalue = loss(dummyout, golds)
        ttt.msg("value of SeqAccuracy on dummy prediction: {}".format(lossvalue))


    # endregion


# DELETE
def run_seq2seq_teacher_forced(lr=OPT_LR,
                               batsize=OPT_BATSIZE,
                               epochs=OPT_EPOCHS,
                               numex=OPT_NUMEX,
                               gradnorm=OPT_GRADNORM,
                               useattention=OPT_USEATTENTION,
                               inpembdim=OPT_INPEMBDIM,
                               outembdim=OPT_OUTEMBDIM,
                               encdim=OPT_ENCDIM,
                               decdim=OPT_DECDIM,
                               dropout=OPT_DROPOUT,
                               cuda=False,
                               gpu=1):
    if cuda:
        torch.cuda.set_device(gpu)
    decdim = decdim * 2     # more equivalent to twostackcell ?
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

    ctxdim = encdim * 2
    if useattention:
        linoutdim = ctxdim + decdim
    else:
        linoutdim = decdim

    inpemb = q.WordEmb(inpembdim, worddic=ism.D)
    outemb = q.WordEmb(outembdim, worddic=osm.D)
    linout = q.WordLinout(linoutdim, worddic=osm.D)

    encoder = make_encoder(inpemb, inpembdim, encdim, dropout, ttt=ttt)

    # region make decoder and put in enc/dec
    layers = (q.GRUCell(outembdim + ctxdim, decdim),
              q.GRUCell(decdim, decdim),)

    if useattention:
        tt.msg("USING ATTENTION!!!")
        decoder_top = q.AttentionContextDecoderTop(q.Attention().dot_gen(),
                                                   linout, ctx2out=True)
    else:
        tt.msg("NOT using attention !!!")
        decoder_top = q.StaticContextDecoderTop(linout)

    decoder_core = q.DecoderCore(outemb, *layers)
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

    class EncDecAtt(torch.nn.Module):
        def __init__(self, encoder, decoder, **kw):
            super(EncDecAtt, self).__init__(**kw)
            self.encoder, self.decoder = encoder, decoder

        def forward(self, inpseq, outinpseq):
            final_encoding, all_encoding, mask = self.encoder(inpseq)
            decoding = self.decoder(outinpseq,
                                    ctx=all_encoding,
                                    ctx_0=final_encoding,
                                    ctxmask=mask)
            return decoding

    if useattention:
        encdec = EncDecAtt(encoder, decoder)
    else:
        encdec = EncDec(encoder, decoder)

    if _opt_test:
        ttt.tick("testing whole thing dry run")
        test_inpseqs = q.var(ism.matrix[:3]).v
        test_outinpseqs = q.var(osm.matrix[:3, :-1]).v

        test_encdec_output = encdec(test_inpseqs, test_outinpseqs)

        ttt.tock("tested whole dryrun")
        # TODO loss and gradients
        golds = q.var(osm.matrix[:3, 1:]).v
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


def make_embedder(dim=None, worddic=None, inpD=None):  # makes structured embedder
    symbols_is_last = np.zeros((len(worddic),), dtype="int64")
    symbols_is_leaf = np.zeros((len(worddic),), dtype="int64")
    for k, v in sorted(worddic.items(), key=lambda (x, y): y):
        ksplits = k.split("*")
        assert(ksplits[0] in inpD)
        symbols_is_last[v] = "LS" in ksplits[1:] or k in "<MASK> <STOP> <NONE> <ROOT>".split()
        symbols_is_leaf[v] = "NC" in ksplits[1:] or k in "<MASK> <START> <STOP> <NONE>".split()

    symbols2ctrl = symbols_is_last * 2 + symbols_is_leaf + 1
    symbols2ctrl[worddic["<MASK>"]] = 0

    symbols2cores = np.zeros((len(worddic),), dtype="int64")
    for symbol in worddic:
        symbols2cores[worddic[symbol]] = inpD[symbol.split("*")[0]]

    class StructEmbComputer(torch.nn.Module):
        def __init__(self, dim, **kw):
            super(StructEmbComputer, self).__init__(**kw)
            self.dim = dim
            self.coreemb = q.WordEmb(self.dim, worddic=inpD)
            self.annemb = q.WordEmb(self.dim, worddic={"<MASK>": 0, "A": 1, "LS": 2, "NC": 3, "NCLS": 4})
            self.annemb.embedding.weight.data.fill_(0)

        def forward(self, data):
            coreembs, mask = self.coreemb(data[:, 0])
            annembs, _ = self.annemb(data[:, 1])
            embs = coreembs + annembs
            return embs

    computer = StructEmbComputer(dim=dim)
    computer_data = np.stack([symbols2cores, symbols2ctrl], axis=1)
    wordemb = q.ComputedWordEmb(data=computer_data, computer=computer, worddic=worddic)
    return wordemb


# DELETE
def run_seq2seq_teacher_forced_structured_output_tokens(
        lr=OPT_LR,
        batsize=OPT_BATSIZE,
        epochs=OPT_EPOCHS,
        numex=OPT_NUMEX,
        gradnorm=OPT_GRADNORM,
        useattention=OPT_USEATTENTION,
        inpembdim=OPT_INPEMBDIM,
        outembdim=OPT_OUTEMBDIM,
        encdim=OPT_ENCDIM,
        decdim=OPT_DECDIM,
        linoutjoinmode=OPT_JOINT_LINOUT_MODE,
        dropout=OPT_DROPOUT,
        inplinmode=OPT_INPLINMODE,
        removeannotation=OPT_REMOVE_ANNOTATION,  # remove annotation from teacher forced decoder input
        cuda=False,
        gpu=1):

    print("SEQ2SEQ + TF + Structured tokens backup")
    if removeannotation:
        print("decoder input does NOT contain structure annotation")
    else:
        print("decoder input DOES contain structure annotation")
    if cuda:
        torch.cuda.set_device(gpu)
    decdim = decdim * 2  # more equivalent to twostackcell ?
    tt = q.ticktock("script")
    ttt = q.ticktock("test")
    ism, tracker, eids, trees = load_synth_trees(n=numex, inplin=inplinmode)
    tt.msg("generated {} synthetic trees".format(ism.matrix.shape[0]))
    # print(ism[0])
    osm = q.StringMatrix(indicate_start=True)
    osm.tokenize = lambda x: x.split()
    psm = q.StringMatrix()
    psm.set_dictionary(tracker.D)
    psm.tokenize = lambda x: x.split()
    trackerDbackup = {k: v for k, v in tracker.D.items()}
    # psm.protectedwords = "<MASK> <RARE> <START> <STOP>".split()
    for tree in trees:
        treestring = tree.pp(arbitrary=True)
        treestring_in = treestring  # .replace("*LS", "").replace("*NC", "")
        if removeannotation:
            # tt.msg("removing annotation")
            treestring_in = treestring_in.replace("*LS", "").replace("*NC", "")
        osm.add(treestring_in)
        psm.add(treestring)
    assert (psm.D == trackerDbackup)
    osm.finalize()
    psm.finalize()
    print(ism[0])
    print(osm[0])
    print(psm[0])

    ctxdim = encdim * 2
    if useattention:
        linoutdim = ctxdim + decdim
    else:
        linoutdim = decdim

    # linout = q.WordLinout(linoutdim, worddic=psm.D)
    assert (psm.D == trackerDbackup)
    linout, symbols2cores, symbols2ctrl \
        = make_computed_linout(psm.D, linoutdim, linoutjoinmode, ttt=ttt)

    # inpemb = make_embedder(dim=inpembdim, worddic=ism.D)
    if removeannotation:
        outemb = q.WordEmb(outembdim, worddic=osm.D)
    else:
        outemb = make_embedder(dim=outembdim, worddic=osm.D)
    inpemb = q.WordEmb(inpembdim, worddic=ism.D)

    encoder = make_encoder(inpemb, inpembdim, encdim, dropout, ttt=ttt)

    # region make decoder and put in enc/dec
    layers = (q.GRUCell(outembdim + ctxdim, decdim),
              q.GRUCell(decdim, decdim),)

    if useattention:
        tt.msg("USING ATTENTION !!!")
        decoder_top = q.AttentionContextDecoderTop(q.Attention().dot_gen(),
                                                   linout, ctx2out=True)
    else:
        tt.msg("NOT using attention !!!")
        decoder_top = q.StaticContextDecoderTop(linout)

    decoder_core = q.DecoderCore(outemb, *layers)
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


    class EncDecAtt(torch.nn.Module):
        def __init__(self, encoder, decoder, **kw):
            super(EncDecAtt, self).__init__(**kw)
            self.encoder, self.decoder = encoder, decoder

        def forward(self, inpseq, outinpseq):
            final_encoding, all_encoding, mask = self.encoder(inpseq)
            decoding = self.decoder(outinpseq,
                                    ctx=all_encoding,
                                    ctx_0=final_encoding,
                                    ctxmask=mask)
            return decoding


    if useattention:
        encdec = EncDecAtt(encoder, decoder)
    else:
        encdec = EncDec(encoder, decoder)

    if _opt_test:
        ttt.tick("testing whole thing dry run")
        test_inpseqs = q.var(ism.matrix[:3]).v
        test_outinpseqs = q.var(osm.matrix[:3, :-1]).v

        test_encdec_output = encdec(test_inpseqs, test_outinpseqs)
        ttt.tock("tested whole dryrun")

        golds = q.var(osm.matrix[:3, 1:]).v
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
            # print(tuple(param.size()), param.grad.norm().data[0])
        ttt.tock("all gradients non-zero")

    # print(encdec)

    # training
    losses = q.lossarray(q.SeqCrossEntropyLoss(ignore_index=0),
                         q.SeqElemAccuracy(ignore_index=0),
                         q.SeqAccuracy(ignore_index=0),
                         TreeAccuracy(ignore_index=0, treeparser=lambda x: Node.parse(tracker.pp(x))))

    optimizer = torch.optim.Adadelta(q.params_of(encdec), lr=lr)

    traindata, testdata = q.split([ism.matrix, osm.matrix, psm.matrix], random=1234)
    traindata, validdata = q.split(traindata, random=1234)

    train_loader = q.dataload(*traindata, batch_size=batsize, shuffle=True)
    valid_loader = q.dataload(*validdata, batch_size=batsize, shuffle=False)
    test_loader = q.dataload(*testdata, batch_size=batsize, shuffle=False)

    q.train(encdec) \
        .train_on(train_loader, losses) \
        .optimizer(optimizer) \
        .clip_grad_norm(gradnorm) \
        .set_batch_transformer(
        lambda inpseq, outseq, predseq:
        (inpseq, outseq[:, :-1], predseq)) \
        .valid_on(valid_loader, losses) \
        .cuda(cuda) \
        .train(epochs)

    results = q.test(encdec).on(test_loader, losses) \
        .set_batch_transformer(
        lambda inpseq, outseq, predseq:
        (inpseq, outseq[:, :-1], predseq)) \
        .cuda(cuda) \
        .run()


# TODO: validate with freerunner
def run_seq2seq_teacher_forced_structured_output_tokens_and_freerun_valid(
               lr=OPT_LR,
               batsize=OPT_BATSIZE,
               epochs=OPT_EPOCHS,
               numex=OPT_NUMEX,
               gradnorm=OPT_GRADNORM,
               useattention=OPT_USEATTENTION,
               inpembdim=OPT_INPEMBDIM,
               outembdim=OPT_OUTEMBDIM,
               encdim=OPT_ENCDIM,
               decdim=OPT_DECDIM,
               linoutjoinmode=OPT_JOINT_LINOUT_MODE,
               dropout=OPT_DROPOUT,
               inplinmode=OPT_INPLINMODE,
               removeannotation=OPT_REMOVE_ANNOTATION,     # remove annotation from teacher forced decoder input
               cuda=False,
               gpu=1):
    print("SEQ2SEQ + TF + Structured tokens")
    if removeannotation:    print("decoder input does NOT contain structure annotation")
    else:                   print("decoder input DOES contain structure annotation")
    if cuda:
        torch.cuda.set_device(gpu)
    decdim = decdim * 2     # more equivalent to twostackcell ?
    tt = q.ticktock("script")
    ttt = q.ticktock("test")
    ism, tracker, eids, trees = load_synth_trees(n=numex, inplin=inplinmode)
    tt.msg("generated {} synthetic trees".format(ism.matrix.shape[0]))
    # print(ism[0])
    osm = q.StringMatrix(indicate_start=True)
    osm.tokenize = lambda x: x.split()
    if removeannotation:
        osm.set_dictionary(tracker.D_in)
    else:
        osm.set_dictionary(tracker.D)
    psm = q.StringMatrix()
    psm.set_dictionary(tracker.D)
    psm.tokenize = lambda x: x.split()
    trackerDbackup = {k: v for k, v in tracker.D.items()}
    # psm.protectedwords = "<MASK> <RARE> <START> <STOP>".split()
    for tree in trees:
        treestring = tree.pp(arbitrary=True)
        treestring_in = treestring      #.replace("*LS", "").replace("*NC", "")
        if removeannotation:
            # tt.msg("removing annotation")
            treestring_in = treestring_in.replace("*LS", "").replace("*NC", "")
        osm.add(treestring_in)
        psm.add(treestring)
    assert(psm.D == trackerDbackup)
    osm.finalize()
    psm.finalize()
    print(ism[0])
    print(osm[0])
    print(psm[0])

    ctxdim = encdim * 2
    if useattention:
        linoutdim = ctxdim + decdim
    else:
        linoutdim = decdim

    # linout = q.WordLinout(linoutdim, worddic=psm.D)
    assert(psm.D == trackerDbackup)
    linout, symbols2cores, symbols2ctrl \
        = make_computed_linout(tracker.D, tracker.D_in, linoutdim, linoutjoinmode, ttt=ttt)

    # inpemb = make_embedder(dim=inpembdim, worddic=ism.D)
    if removeannotation:
        outemb = q.WordEmb(outembdim, worddic=tracker.D_in)
    else:
        outemb = make_embedder(dim=outembdim, worddic=tracker.D, inpD=tracker.D_in)
    inpemb = q.WordEmb(inpembdim, worddic=ism.D)

    encoder = make_encoder(inpemb, inpembdim, encdim, dropout, ttt=ttt)

    # region make decoder and put in enc/dec
    layers = (q.GRUCell(outembdim + ctxdim, decdim),
              q.GRUCell(decdim, decdim),)

    if useattention:
        tt.msg("USING ATTENTION !!!")
        decoder_top = q.AttentionContextDecoderTop(q.Attention().dot_gen(),
                                                   linout, ctx2out=True)
    else:
        tt.msg("NOT using attention !!!")
        decoder_top = q.StaticContextDecoderTop(linout)

    decoder_core = q.DecoderCore(outemb, *layers)
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

    class EncDecAtt(torch.nn.Module):
        def __init__(self, encoder, decoder, **kw):
            super(EncDecAtt, self).__init__(**kw)
            self.encoder, self.decoder = encoder, decoder

        def forward(self, inpseq, outinpseq):
            final_encoding, all_encoding, mask = self.encoder(inpseq)
            decoding = self.decoder(outinpseq,
                                    ctx=all_encoding,
                                    ctx_0=final_encoding,
                                    ctxmask=mask)
            return decoding

    if useattention:
        encdec = EncDecAtt(encoder, decoder)
    else:
        encdec = EncDec(encoder, decoder)

    if _opt_test:
        ttt.tick("testing whole thing dry run")
        test_inpseqs = q.var(ism.matrix[:3]).v
        test_outinpseqs = q.var(osm.matrix[:3, :-1]).v.contiguous()

        test_encdec_output = encdec(test_inpseqs, test_outinpseqs)
        ttt.tock("tested whole dryrun")

        golds = q.var(osm.matrix[:3, 1:]).v
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
            # print(tuple(param.size()), param.grad.norm().data[0])
        ttt.tock("all gradients non-zero")

    # print(encdec)

    # training
    losses = q.lossarray(q.SeqCrossEntropyLoss(ignore_index=0),
                         q.SeqElemAccuracy(ignore_index=0),
                         q.SeqAccuracy(ignore_index=0),
                         TreeAccuracy(ignore_index=0, treeparser=lambda x: Node.parse(tracker.pp(x))))

    validlosses = q.lossarray(TreeAccuracy(ignore_index=0, treeparser=lambda x: Node.parse(tracker.pp(x))))

    optimizer = torch.optim.Adadelta(q.params_of(encdec), lr=lr)

    # starts = np.ones((len(ism.matrix, )), dtype="int64") * startid

    traindata, testdata = q.split([ism.matrix, osm.matrix, psm.matrix], random=1234)
    traindata, validdata = q.split(traindata, random=1234)

    train_loader = q.dataload(*traindata, batch_size=batsize, shuffle=True)
    valid_loader = q.dataload(*validdata, batch_size=batsize, shuffle=False)
    test_loader = q.dataload(*testdata, batch_size=batsize, shuffle=False)

    # region make validation network with freerunner
    inparggetter = lambda x: x
    if removeannotation:
        symbols2cores_pt = q.var(symbols2cores).cuda(cuda).v
        inparggetter = lambda x: torch.index_select(symbols2cores_pt, 0, x)

    freerunner = q.FreeRunner(inparggetter=inparggetter)

    valid_decoder_cell = q.ModularDecoderCell(decoder_core, decoder_top)
    valid_decoder_cell.set_runner(freerunner)
    valid_decoder = valid_decoder_cell.to_decoder()
    #
    # class ValidEncDec(torch.nn.Module):
    #     def __init__(self, encoder, decoder, **kw):
    #         super(ValidEncDec, self).__init__(**kw)
    #         self.encoder = encoder
    #         self.decoder = decoder
    #
    #     def forward(self, inpseq, decstarts, eids=None, maxtime=None):
    #         final_encoding, all_encoding, mask = self.encoder(inpseq)
    #         # self.decoder.block.decoder_top.set_ctx(final_encoding)
    #         decoding = self.decoder(decstarts, ctx=final_encoding, eids=eids, maxtime=maxtime)
    #         return decoding
    #
    # class ValidEncDecAtt(torch.nn.Module):
    #     def __init__(self, encoder, decoder, **kw):
    #         super(ValidEncDecAtt, self).__init__(**kw)
    #         self.encoder, self.decoder = encoder, decoder
    #
    #     def forward(self, inpseq, decstarts, eids=None, maxtime=None):
    #         final_encoding, all_encoding, mask = self.encoder(inpseq)
    #         decoding = self.decoder(decstarts,
    #                                 ctx=all_encoding, ctx_0=final_encoding, ctxmask=mask,
    #                                 eids=eids, maxtime=maxtime)
    #         return decoding

    if useattention:
        valid_encdec = EncDecAtt(encoder, valid_decoder)
    else:
        valid_encdec = EncDec(encoder, valid_decoder)
    # endregion

    q.train(encdec)\
        .train_on(train_loader, losses)\
        .optimizer(optimizer)\
        .clip_grad_norm(gradnorm)\
        .set_batch_transformer(
            lambda inpseq, outseq, predseq:
                (inpseq, outseq[:, :-1], predseq))\
        .valid_with(valid_encdec) \
        .valid_on(valid_loader, validlosses)\
        .cuda(cuda)\
        .train(epochs)

    results = q.test(valid_encdec).on(test_loader, validlosses)\
        .set_batch_transformer(
            lambda inpseq, outseq, predseq:
                (inpseq, outseq[:, :-1], predseq))\
        .cuda(cuda)\
        .run()


# TODO: validate with freerunner
def run_seq2seq_oracle(lr=OPT_LR,
                       batsize=OPT_BATSIZE,
                       epochs=OPT_EPOCHS,
                       numex=OPT_NUMEX,
                       gradnorm=OPT_GRADNORM,
                       useattention=OPT_USEATTENTION,
                       inpembdim=OPT_INPEMBDIM,
                       outembdim=OPT_OUTEMBDIM,
                       encdim=OPT_ENCDIM,
                       decdim=OPT_DECDIM,
                       dropout=OPT_DROPOUT,
                       explore=OPT_EXPLORE,
                       linoutjoinmode=OPT_JOINT_LINOUT_MODE,
                       oraclemode=OPT_ORACLE_MODE,
                       inplinmode=OPT_INPLINMODE,
                       removeannotation=OPT_REMOVE_ANNOTATION,
                       cuda=False,
                       gpu=1):
    if cuda:
        torch.cuda.set_device(gpu)
    decdim = decdim * 2     # more equivalent to twostackcell ?
    tt = q.ticktock("script")
    ttt = q.ticktock("test")
    if useattention:
        tt.msg("using attention!!!")
    else:
        tt.msg("NOT using attention!!!")
    if removeannotation:    print("decoder input does NOT contain structure annotation")
    else:                   print("decoder input DOES contain structure annotation")
    ism, tracker, eids, trees = load_synth_trees(n=numex, inplin=inplinmode)
    tt.msg("generated {} synthetic trees".format(ism.matrix.shape[0]))
    psm = q.StringMatrix()
    psm.set_dictionary(tracker.D)
    psm.tokenize = lambda x: x.split()
    for tree in trees:
        treestring = tree.pp(arbitrary=True)
        psm.add(treestring)
    psm.finalize()
    print(ism[0])
    print(psm[0])

    ctxdim = encdim * 2
    if useattention:
        linoutdim = ctxdim + decdim
    else:
        linoutdim = decdim

    linout, symbols2cores, symbols2ctrl \
        = make_computed_linout(tracker.D, tracker.D_in, linoutdim, linoutjoinmode, ttt=ttt)

    # inpemb = make_embedder(dim=inpembdim, worddic=ism.D)
    # outemb = make_embedder(dim=outembdim, worddic=tracker.D_in)
    inpemb = q.WordEmb(inpembdim, worddic=ism.D)
    outemb = q.WordEmb(outembdim, worddic=tracker.D_in)

    oracle = make_oracle(tracker, symbols2cores, symbols2ctrl, explore, cuda, mode=oraclemode,
                         ttt=ttt, linout=linout, outemb=outemb, linoutdim=linoutdim, trees=trees)
    original_inparggetter = oracle.inparggetter

    if removeannotation:
        oracle.inparggetter = lambda x: original_inparggetter(x)[0]
    else:
        outemb = make_embedder(dim=outembdim, worddic=tracker.D, inpD=tracker.D_in)
        oracle.inparggetter = lambda x: x

    encoder = make_encoder(inpemb, inpembdim, encdim, dropout, ttt=ttt)

    # region make decoder and put in enc/dec
    layers = (q.GRUCell(outembdim + ctxdim, decdim),
              q.GRUCell(decdim, decdim),)

    if useattention:
        decoder_top = q.AttentionContextDecoderTop(q.Attention().dot_gen(),
                                                   linout, ctx2out=True)
    else:
        decoder_top = q.StaticContextDecoderTop(linout)

    decoder_core = q.DecoderCore(outemb, *layers)
    decoder_cell = q.ModularDecoderCell(decoder_core, decoder_top)
    decoder_cell.set_runner(oracle)
    decoder = decoder_cell.to_decoder()

    # wrap in encdec
    class EncDec(torch.nn.Module):
        def __init__(self, encoder, decoder, maxtime=None, **kw):
            super(EncDec, self).__init__(**kw)
            self.encoder = encoder
            self.decoder = decoder
            self.maxtime = maxtime

        def forward(self, inpseq, decstarts, eids=None, maxtime=None):
            final_encoding, all_encoding, mask = self.encoder(inpseq)
            # self.decoder.block.decoder_top.set_ctx(final_encoding)
            maxtime = self.maxtime if maxtime is None else maxtime
            decoding = self.decoder(decstarts, ctx=final_encoding, eids=eids, maxtime=maxtime)
            return decoding

    class EncDecAtt(torch.nn.Module):
        def __init__(self, encoder, decoder, maxtime=None, **kw):
            super(EncDecAtt, self).__init__(**kw)
            self.encoder, self.decoder = encoder, decoder
            self.maxtime = maxtime

        def forward(self, inpseq, decstarts, eids=None, maxtime=None):
            final_encoding, all_encoding, mask = self.encoder(inpseq)
            maxtime = self.maxtime if maxtime is None else maxtime
            decoding = self.decoder(decstarts,
                                    ctx=all_encoding, ctx_0=final_encoding, ctxmask=mask,
                                    eids=eids, maxtime=maxtime)
            return decoding

    if useattention:
        encdec = EncDecAtt(encoder, decoder)
    else:
        encdec = EncDec(encoder, decoder)

    if _opt_test:
        ttt.tick("testing whole thing dry run")
        test_eids = q.var(np.arange(0, 3)).cuda(cuda).v
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
            exptree = trees[i]
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
        ttt.msg("value of predicted Tree Accuracy: {}".format(lossvalue))
        dummyout = q.var(torch.zeros(out.size())).cuda(cuda).v
        dummyout.scatter_(2, seqs.unsqueeze(2), 1)
        lossvalue = loss(dummyout, golds)
        ttt.msg("value of dummy predicted Tree Accuracy: {}".format(lossvalue))
        loss = q.SeqAccuracy()
        lossvalue = loss(dummyout, golds)
        ttt.msg("value of SeqAccuracy on dummy prediction: {}".format(lossvalue))
        encdec.cpu()

        encdec.eval()
        assert(not oracle.training)
        ttt.msg("oracle switched to eval")
        encdec.train()
        assert(oracle.training)
        ttt.msg("oracle switched to training")

    # print(encdec)

    # training
    eids = np.arange(0, len(ism.matrix), dtype="int64")
    startid = outemb.D["<ROOT>"]
    tt.msg("using startid {} from outemb".format(startid))
    starts = np.ones((len(ism.matrix,)), dtype="int64") * startid

    alldata = [ism.matrix, starts, eids, psm.matrix, eids]
    traindata, testdata = q.split(alldata, random=1234)
    traindata, validdata = q.split(traindata, random=1234)

    train_loader = q.dataload(*[traindata[i] for i in [0, 1, 2, 4]], batch_size=batsize, shuffle=True)
    valid_loader = q.dataload(*[validdata[i] for i in [0, 1, 3]], batch_size=batsize, shuffle=False)
    test_loader = q.dataload(*[testdata[i] for i in [0, 1, 3]], batch_size=batsize, shuffle=False)

    # region make validation network with freerunner
    inparggetter = lambda x: x
    if removeannotation:
        symbols2cores_pt = q.var(symbols2cores).cuda(cuda).v
        inparggetter = lambda x: torch.index_select(symbols2cores_pt, 0, x)

    freerunner = q.FreeRunner(inparggetter=inparggetter)

    valid_decoder_cell = q.ModularDecoderCell(decoder_core, decoder_top)
    valid_decoder_cell.set_runner(freerunner)
    valid_decoder = valid_decoder_cell.to_decoder()
    if useattention:
        valid_encdec = EncDecAtt(encoder, valid_decoder, maxtime=50)
    else:
        valid_encdec = EncDec(encoder, valid_decoder, maxtime=50)
    # endregion

    losses = q.lossarray(q.SeqCrossEntropyLoss(ignore_index=0),
                         q.SeqElemAccuracy(ignore_index=0),
                         q.SeqAccuracy(ignore_index=0),
                         TreeAccuracy(ignore_index=0, treeparser=lambda x: Node.parse(tracker.pp(x))))

    validlosses = q.lossarray(TreeAccuracy(ignore_index=0, treeparser=lambda x: Node.parse(tracker.pp(x))))

    optimizer = torch.optim.Adadelta(q.params_of(encdec), lr=lr)

    out_btf = lambda _out: _out[:, :-1, :]
    gold_btf = lambda _eids: torch.stack(oracle.goldacc, 1)
    valid_gold_btf = lambda x: x

    q.train(encdec) \
        .train_on(train_loader, losses) \
        .optimizer(optimizer) \
        .clip_grad_norm(gradnorm) \
        .set_batch_transformer(None, out_btf, gold_btf) \
        .valid_with(valid_encdec)\
        .set_valid_batch_transformer(None, out_btf, valid_gold_btf)\
        .valid_on(valid_loader, validlosses) \
        .cuda(cuda) \
        .train(epochs)

    q.embed()

    results = q.test(valid_encdec).on(test_loader, validlosses)\
        .set_batch_transformer(None, out_btf, valid_gold_btf)\
        .cuda(cuda)\
        .run()


def test_load_synth_trees(n=100):
    ism, tracker, eids, trees = load_synth_trees(n, inplin="df")
    for i in range(10):
        print(ism[i])
        print(trees[i].ppdf(mode="ann"))
        print(trees[i].pptree())


def test_make_computed_linout(n=100):
    ism, tracker, eids, trees = load_synth_trees(n=n, inplin="df")
    linout, symbols2cores, symbols2cores = make_computed_linout(tracker.D, OPT_LINOUTDIM, OPT_JOINT_LINOUT_MODE)


def test_make_oracle(n=100):
    ism, tracker, eids, trees = load_synth_trees(n=n, inplin="df")

    linout, symbols2cores, symbols2ctrl \
        = make_computed_linout(tracker.D, 10, "sum")

    inpemb = q.WordEmb(10, worddic=ism.D)
    outemb = q.WordEmb(10, worddic=tracker.D_in)

    oracle = make_oracle(tracker, symbols2cores, symbols2ctrl, False, False, trees=trees,
                         linout=linout, outemb=outemb, linoutdim=10)


if __name__ == "__main__":
    print("pytorch version: {}".format(torch.version.__version__))
    ### q.argprun(run_seq2seq_teacher_forced)
    # q.argprun(run_seq2seq_teacher_forced_structured_output_tokens)
    # q.argprun(run_seq2seq_teacher_forced_structured_output_tokens_and_oracle_valid)
    q.argprun(run_seq2seq_oracle)
    # q.argprun(run)
    # q.argprun(test_make_computed_linout)
    # q.argprun(test_make_oracle)