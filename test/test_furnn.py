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


class TestTwoStackCell(TestCase):
    def test_it_gru(self):
        batsize = 5
        indim = 10
        outdim = 16
        x = q.var(torch.randn((batsize, indim))).v
        ctrl = q.var(np.random.randint(0, 3, (batsize,))).v
        innercell = q.RecStack(q.GRUCell(indim, outdim))
        cell = TwoStackCell(innercell)

        y = cell(x, ctrl_tm1=ctrl)
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
