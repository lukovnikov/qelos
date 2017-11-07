from unittest import TestCase
import qelos as q
import numpy as np
from qelos.furnn import TwoStackCell
import torch


class TestMemGRUCell(TestCase):
    def test_shapes(self):
        x = q.var(np.random.random((2, 5, 3)).astype("float32")).v
        m = q.MemGRUCell(3, 4, memsize=3).to_layer()

        y = m(x)
        self.assertEqual(y.size(), (2, 5, 4))


class TestTwoStackDecoder(TestCase):

    def do_tst_it(self, innercell=lambda a, b: q.RecStack(q.GRUCell(a, b))):
        indim = 10
        outdim = 16
        data = [
            "<START> <ROOT> BIN1 LEAF1 LEAF2",
            "<START> <ROOT> BIN2 UNI1 LEAF1 LEAF2",
            "<START> <ROOT> BIN3 LEAF1 BIN4 LEAF2 LEAF3",
            "<START> <ROOT> LEAF4",
        ]
        ctrl = [
            [3, 3, 3, 2, 4, 0, 0],
            [3, 3, 3, 1, 4, 4, 0],
            [3, 3, 3, 2, 3, 2, 4],
            [3, 3, 4, 0, 0, 0, 0],
        ]
        ctrl = np.asarray(ctrl)

        sm = q.StringMatrix()
        sm.tokenize = lambda x: x.split()
        for data_e in data:
            sm.add(data_e)
        sm.finalize()

        dic = sm.D
        datamat = sm.matrix
        vocsize = max(dic.values()) + 1

        inneremb = q.WordEmb(indim//2, worddic=dic)
        innercell = innercell(indim, outdim)
        decoder_core = TwoStackCell(inneremb, innercell)
        decoder_top = q.Stack(
            torch.nn.Linear(outdim, vocsize),
            # torch.nn.Softmax(),
        )
        decodercell = q.ModularDecoderCell(decoder_core, decoder_top)
        decoder = decodercell.to_decoder()

        x = q.var(datamat).v
        ctrl = q.var(ctrl).v

        y = decoder(x, ctrl)
        pass

    def test_it_gru(self):
        innercell = lambda a, b: q.RecStack(q.GRUCell(a, b))
        self.do_tst_it(innercell)

    def test_it_catlstm(self):
        innercell = lambda a, b: q.RecStack(q.CatLSTMCell(a, b))
        self.do_tst_it(innercell)

    def test_it_dual_catlstm(self):
        innercell = lambda a, b: (q.RecStack(q.CatLSTMCell(a, b//2)),
                                  q.RecStack(q.CatLSTMCell(a, b//2)))
        self.do_tst_it(innercell)


class TestTwoStackContextDecoder(TestCase):
    def do_tst_it(self, innercell=lambda a, b: q.RecStack(q.GRUCell(a, b))):
        indim = 10
        ctxdim = 11
        outdim = 16
        data = [
            "<START> <ROOT> BIN1 LEAF1 LEAF2",
            "<START> <ROOT> BIN2 UNI1 LEAF1 LEAF2",
            "<START> <ROOT> BIN3 LEAF1 BIN4 LEAF2 LEAF3",
            "<START> <ROOT> LEAF4",
        ]
        ctrl = [
            [3, 3, 3, 2, 4, 0, 0],
            [3, 3, 3, 1, 4, 4, 0],
            [3, 3, 3, 2, 3, 2, 4],
            [3, 3, 4, 0, 0, 0, 0],
        ]
        ctrl = np.asarray(ctrl)

        ctx = np.random.random((4, ctxdim)).astype("float32")

        sm = q.StringMatrix()
        sm.tokenize = lambda x: x.split()
        for data_e in data:
            sm.add(data_e)
        sm.finalize()

        dic = sm.D
        datamat = sm.matrix
        vocsize = max(dic.values()) + 1

        inneremb = q.WordEmb(indim // 2, worddic=dic)
        innercell = innercell(indim + ctxdim, outdim)
        decoder_core = TwoStackCell(inneremb, innercell)
        decoder_top = q.StaticContextDecoderTop(
            q.argmap.spec(0),
            torch.nn.Linear(outdim, vocsize),
            # torch.nn.Softmax(),
        )
        decodercell = q.ModularDecoderCell(decoder_core, decoder_top)
        decoder = decodercell.to_decoder()

        x = q.var(datamat).v
        ctrl = q.var(ctrl).v
        ctx = q.var(ctx).v

        y = decoder(x, ctrl, ctx=ctx)
        pass

    def test_it_gru(self):
        innercell = lambda a, b: q.RecStack(q.GRUCell(a, b))
        self.do_tst_it(innercell)

    def test_it_catlstm(self):
        innercell = lambda a, b: q.RecStack(q.CatLSTMCell(a, b))
        self.do_tst_it(innercell)

    def test_it_dual_catlstm(self):
        innercell = lambda a, b: (q.RecStack(q.CatLSTMCell(a, b // 2)),
                                  q.RecStack(q.CatLSTMCell(a, b // 2)))
        self.do_tst_it(innercell)


class TestTwoStackStateInitDecoder(TestCase):
    def do_tst_it(self, innercell=lambda a, b: q.RecStack(q.GRUCell(a, b)),
                        statesetter=lambda *x, **kw: kw["ctx"],
                        dual=False):
        indim = 10
        ctxdim = 16 if not dual else 8
        outdim = 16
        data = [
            "<START> <ROOT> BIN1 LEAF1 LEAF2",
            "<START> <ROOT> BIN2 UNI1 LEAF1 LEAF2",
            "<START> <ROOT> BIN3 LEAF1 BIN4 LEAF2 LEAF3",
            "<START> <ROOT> LEAF4",
        ]
        ctrl = [
            [3, 3, 3, 2, 4, 0, 0],
            [3, 3, 3, 1, 4, 4, 0],
            [3, 3, 3, 2, 3, 2, 4],
            [3, 3, 4, 0, 0, 0, 0],
        ]
        ctrl = np.asarray(ctrl)

        ctx = np.random.random((4, ctxdim)).astype("float32")

        sm = q.StringMatrix()
        sm.tokenize = lambda x: x.split()
        for data_e in data:
            sm.add(data_e)
        sm.finalize()

        dic = sm.D
        datamat = sm.matrix
        vocsize = max(dic.values()) + 1

        inneremb = q.WordEmb(indim // 2, worddic=dic)
        innercell = innercell(indim, outdim)
        decoder_core = TwoStackCell(inneremb, innercell)
        decoder_top = q.DecoderTop(
            q.argmap.spec(0),
            torch.nn.Linear(outdim, vocsize),
            # torch.nn.Softmax(),
        )
        decodercell = q.ModularDecoderCell(decoder_core, decoder_top)
        decodercell.set_init_states_computer(statesetter)
        decoder = decodercell.to_decoder()

        x = q.var(datamat).v
        ctrl = q.var(ctrl).v
        ctx = q.var(ctx).v

        y = decoder(x, ctrl, ctx=ctx)
        pass

    def test_it_gru(self):
        innercell = lambda a, b: q.RecStack(q.GRUCell(a, b))
        statesetter = lambda *x, **kw: kw["ctx"]
        self.do_tst_it(innercell, statesetter)

    def test_it_catlstm(self):
        innercell = lambda a, b: q.RecStack(q.CatLSTMCell(a, b))
        statesetter = lambda *x, **kw: torch.cat([kw["ctx"], kw["ctx"]], 1)
        self.do_tst_it(innercell, statesetter)

    def test_it_dual_catlstm(self):
        innercell = lambda a, b: (q.RecStack(q.CatLSTMCell(a, b // 2)),
                                  q.RecStack(q.CatLSTMCell(a, b // 2)))

        def statesetter(*x, **kw):
            ctx = kw["ctx"]
            ancestralinit = (torch.cat([ctx, ctx], 1),)
            fraternalinit = (None,)
            return ancestralinit + fraternalinit

        self.do_tst_it(innercell, statesetter, dual=True)

    def test_it_dual_twolayer_catlstm(self):
        innercell = lambda a, b: (q.RecStack(
                                    q.CatLSTMCell(a, b),
                                    q.CatLSTMCell(b, b//2)),
                                  q.RecStack(
                                    q.CatLSTMCell(a, b),
                                    q.CatLSTMCell(b, b // 2)))

        def statesetter(*x, **kw):
            ctx = kw["ctx"]
            ancestralinit = (None, torch.cat([ctx, ctx], 1),)
            fraternalinit = (None, None)
            return ancestralinit + fraternalinit

        self.do_tst_it(innercell, statesetter, dual=True)


class TestTwoStackAttentionContextDecoder(TestCase):
    def do_tst_it(self, innercell=lambda a, b, c: q.RecStack(q.GRUCell(a+b, b))):
        indim = 10
        ctxdim = 16
        outdim = 16
        data = [
            "<START> <ROOT> BIN1 LEAF1 LEAF2",
            "<START> <ROOT> BIN2 UNI1 LEAF1 LEAF2",
            "<START> <ROOT> BIN3 LEAF1 BIN4 LEAF2 LEAF3",
            "<START> <ROOT> LEAF4",
        ]
        ctrl = [
            [3, 3, 3, 2, 4, 0, 0],
            [3, 3, 3, 1, 4, 4, 0],
            [3, 3, 3, 2, 3, 2, 4],
            [3, 3, 4, 0, 0, 0, 0],
        ]
        ctrl = np.asarray(ctrl)

        ctx = np.random.random((4, 5, ctxdim)).astype("float32")
        ctx_0 = np.random.random((4, ctxdim)).astype("float32")

        sm = q.StringMatrix()
        sm.tokenize = lambda x: x.split()
        for data_e in data:
            sm.add(data_e)
        sm.finalize()

        dic = sm.D
        datamat = sm.matrix
        vocsize = max(dic.values()) + 1

        inneremb = q.WordEmb(indim // 2, worddic=dic)
        innercell = innercell(indim, ctxdim//2, outdim)
        decoder_core = TwoStackCell(inneremb, innercell)
        decoder_top = q.AttentionContextDecoderTop(
            q.Attention().cosine_gen(),
            q.argmap.spec(0),
            torch.nn.Linear(outdim//2+ctxdim//2, vocsize),
            # torch.nn.Softmax(),
            split=True,
        )
        decodercell = q.ModularDecoderCell(decoder_core, decoder_top)
        decoder = decodercell.to_decoder()

        x = q.var(datamat).v
        ctrl = q.var(ctrl).v
        ctx = q.var(ctx).v
        ctx_0 = q.var(ctx_0).v

        y = decoder(x, ctrl, ctx=ctx, ctx_0=ctx_0)
        pass

    def test_it_gru(self):
        innercell = lambda a, b, c: q.RecStack(q.GRUCell(a + b, c))
        self.do_tst_it(innercell)

    def test_it_catlstm(self):
        innercell = lambda a, b, c: q.RecStack(q.CatLSTMCell(a + b, c))
        self.do_tst_it(innercell)

    def test_it_dual_catlstm(self):
        innercell = lambda a, b, c: (q.RecStack(q.CatLSTMCell(a//2 + b, c // 2)),
                                     q.RecStack(q.CatLSTMCell(a//2 + b, c // 2)))
        self.do_tst_it(innercell)