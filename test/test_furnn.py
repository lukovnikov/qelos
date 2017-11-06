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

    def test_it_gru(self):
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
        innercell = q.RecStack(q.GRUCell(indim, outdim))
        decoder_core = TwoStackCell(inneremb, innercell)
        decoder_top = q.Stack(
            torch.nn.Linear(outdim, vocsize),
            torch.nn.Softmax(),
        )
        decodercell = q.ModularDecoderCell(decoder_core, decoder_top)
        decoder = decodercell.to_decoder()

        x = q.var(datamat).v
        ctrl = q.var(ctrl).v

        y = decoder(x, ctrl)
        pass


    def test_it_catlstm(self):
        batsize = 5
        indim = 10
        outdim = 16
        x = q.var(torch.randn((batsize, indim))).v
        ctrl = q.var(np.random.randint(0, 3, (batsize,))).v
        innercell = q.RecStack(q.CatLSTMCell(indim, outdim))
        cell = TwoStackCell(innercell)

        y = cell(x, ctrl_tm1=ctrl)
        pass

    def test_it_dual_catlstm(self):
        batsize = 5
        indim = 10
        outdim = 8
        x = q.var(torch.randn((batsize, indim))).v
        ctrl = q.var(np.random.randint(0, 3, (batsize,))).v
        innercell = (q.RecStack(q.CatLSTMCell(indim, outdim)),
                     q.RecStack(q.CatLSTMCell(indim, outdim)))
        cell = TwoStackCell(innercell)

        y = cell(x, ctrl_tm1=ctrl)
        pass
