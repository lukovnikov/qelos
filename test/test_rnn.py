from __future__ import print_function
from __future__ import print_function
from unittest import TestCase
import qelos.rnn as rnn
from torch.autograd import Variable
import torch
import numpy as np


class TestGRU(TestCase):
    def test_gru_shapes(self):
        batsize = 5
        rnn.GRU.debug = True
        gru = rnn.GRU(9, 10)
        x_t = Variable(torch.FloatTensor(np.random.random((batsize, 9))))
        h_tm1 = Variable(torch.FloatTensor(np.random.random((batsize, 10))))
        gru.set_init_states(h_tm1)
        h_t, rg, ug = gru._forward(x_t, h_tm1)
        # simulate reset gate
        nprg = np.dot(x_t.data.numpy(), gru.reset_gate.W.data.numpy())
        nprg += np.dot(h_tm1.data.numpy(), gru.reset_gate.U.data.numpy())
        nprg += gru.reset_gate.b.data.numpy()
        self.assertTrue(np.allclose(rg, nprg))
        # simulate update gate
        npug = np.dot(x_t.data.numpy(), gru.update_gate.W.data.numpy())
        npug += np.dot(h_tm1.data.numpy(), gru.update_gate.U.data.numpy())
        npug += gru.update_gate.b.data.numpy()
        self.assertTrue(np.allclose(ug, npug))
        # output shape
        self.assertEqual((5, 10), h_t.data.numpy().shape)
        print(rg)

    def test_params_collected(self):
        gru = rnn.GRU(9, 10)
        param_gen = gru.parameters()
        params = []
        exp_params = set([gru.reset_gate.W, gru.reset_gate.U, gru.update_gate.W, gru.update_gate.U, gru.main_gate.W, gru.main_gate.U])
        for param in param_gen:
            for exp_param in exp_params:
                if param is exp_param:
                    exp_params = exp_params.difference(set([param]))
                    break
        self.assertTrue(len(exp_params) == 0)

    def test_zoneout(self):
        batsize = 5
        rnn.GRU.debug = False
        gru = rnn.GRU(9, 10, zoneout=0.5)
        x_t = Variable(torch.FloatTensor(np.random.random((batsize, 9))))
        h_tm1 = Variable(torch.FloatTensor(np.random.random((batsize, 10))))
        gru.set_init_states(h_tm1)
        h_t = gru(x_t)
        self.assertEqual((5, 10), h_t.data.numpy().shape)
        self.assertEqual(gru.training, True)
        gru.train(mode=False)
        self.assertEqual(gru.training, False)
        gru.reset_state()
        pred1 = gru(x_t)
        gru.reset_state()
        pred2 = gru(x_t)
        # must be equal in prediction mode
        print(pred1)
        print(pred2)
        self.assertTrue(np.allclose(pred1.data.numpy(), pred2.data.numpy()))
        gru.train(mode=True)
        self.assertEqual(gru.training, True)
        gru.reset_state()
        pred1 = gru(x_t)
        gru.reset_state()
        pred2 = gru(x_t)
        # must not be equal in prediction mode
        self.assertFalse(np.allclose(pred1.data.numpy(), pred2.data.numpy()))


class TestLSTM(TestCase):
    def test_lstm_shapes(self):
        batsize = 5
        rnn.LSTM.debug = True
        lstm = rnn.LSTM(9, 10)
        x_t = Variable(torch.FloatTensor(np.random.random((batsize, 9))))
        h_tm1 = Variable(torch.FloatTensor(np.random.random((batsize, 10))))
        c_tm1 = Variable(torch.FloatTensor(np.random.random((batsize, 10))))
        lstm.set_init_states(c_tm1, h_tm1)
        y_t = lstm(x_t)
        self.assertEqual((5, 10), y_t.data.numpy().shape)

    def test_zoneout(self):
        batsize = 5
        rnn.LSTM.debug = False
        lstm = rnn.LSTM(9, 10, zoneout=0.5)
        x_t = Variable(torch.FloatTensor(np.random.random((batsize, 9))))
        h_tm1 = Variable(torch.FloatTensor(np.random.random((batsize, 10))))
        c_tm1 = Variable(torch.FloatTensor(np.random.random((batsize, 10))))
        lstm.set_init_states(c_tm1, h_tm1)
        y_t = lstm(x_t)
        self.assertEqual((5, 10), y_t.data.numpy().shape)
        self.assertEqual(lstm.training, True)
        lstm.train(mode=False)
        self.assertEqual(lstm.training, False)
        lstm.reset_state()
        pred1 = lstm(x_t)
        lstm.reset_state()
        pred2 = lstm(x_t)
        # must be equal in prediction mode
        self.assertTrue(np.allclose(pred1.data.numpy(), pred2.data.numpy()))
        lstm.train(mode=True)
        self.assertEqual(lstm.training, True)
        lstm.reset_state()
        pred1 = lstm(x_t)
        lstm.reset_state()
        pred2 = lstm(x_t)
        # must not be equal in prediction mode
        self.assertFalse(np.allclose(pred1.data.numpy(), pred2.data.numpy()))


class Test_RNNLayer(TestCase):
    def test_lstm_layer_shapes(self):
        batsize = 5
        seqlen = 7
        rnn.LSTM.debug = False
        lstm = rnn.LSTM(9, 10)
        lstm = lstm.to_layer().return_all()
        x = Variable(torch.FloatTensor(np.random.random((batsize, seqlen, 9))))
        y = lstm(x)
        self.assertEqual((batsize, seqlen, 10), y.data.numpy().shape)

    def test_masked_gru(self):
        batsize = 3
        seqlen = 4
        rnn.GRU.debug = False
        gru = rnn.GRU(9, 10)
        gru = gru.to_layer().return_all().return_final()
        x = Variable(torch.FloatTensor(np.random.random((batsize, seqlen, 9))))
        m_val = np.asarray([[1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 1]])
        m = Variable(torch.FloatTensor(m_val))
        y_t, y = gru(x, mask=m)
        pred = y.data.numpy()
        self.assertTrue(np.allclose(y_t.data.numpy(), y.data.numpy()[:, -1]))
        # TODO write assertions

    def test_masked_gru_reverse(self):
        batsize = 3
        seqlen = 4
        rnn.GRU.debug = False
        gru = rnn.GRU(9, 10)
        gru = gru.to_layer().return_all().return_final()
        x = Variable(torch.FloatTensor(np.random.random((batsize, seqlen, 9))))
        m_val = np.asarray([[1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 1]])
        m = Variable(torch.FloatTensor(m_val))
        y_t, y = gru(x, mask=m, reverse=True)
        pred = y.data.numpy()
        self.assertTrue(np.allclose(y_t.data.numpy(), y.data.numpy()[:, 0]))
        # TODO write assertions

    def test_masked_gru_bidir(self):
        batsize = 3
        seqlen = 4
        rnn.GRU.debug = False
        gru = rnn.GRU(9, 5)
        gru2 = rnn.GRU(9, 5)
        layer = rnn.BiRNNLayer(gru, gru2, mode="cat").return_final().return_all()
        x = Variable(torch.FloatTensor(np.random.random((batsize, seqlen, 9))))
        m_val = np.asarray([[1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 1]])
        m = Variable(torch.FloatTensor(m_val))
        y_t, y = layer(x, mask=m)
        pred = y.data.numpy()
        # TODO write assertions


class TestRecStack(TestCase):
    def test_shapes(self):
        batsize = 5
        m = rnn.RecStack(rnn.GRU(9, 10), rnn.GRU(10, 11))
        x_t = Variable(torch.FloatTensor(np.random.random((batsize, 9))))
        h_tm1_a = Variable(torch.FloatTensor(np.random.random((batsize, 10))))
        h_tm1_b = Variable(torch.FloatTensor(np.random.random((batsize, 11))))
        m.set_init_states(h_tm1_a, h_tm1_b)
        y_t = m(x_t)
        self.assertEqual((batsize, 11), y_t.data.numpy().shape)

    def test_masked_gru_stack(self):
        batsize = 3
        seqlen = 4

        m = rnn.RecStack(rnn.GRU(9, 10), rnn.GRU(10, 11))
        x = Variable(torch.FloatTensor(np.random.random((batsize, seqlen, 9))))
        h_tm1_a = Variable(torch.FloatTensor(np.random.random((batsize, 10))))
        h_tm1_b = Variable(torch.FloatTensor(np.random.random((batsize, 11))))
        m.set_init_states(h_tm1_a, h_tm1_b)
        m = m.to_layer().return_final().return_all()

        mask_val = np.asarray([[1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 1]])
        mask = Variable(torch.FloatTensor(mask_val))

        y_t, y = m(x, mask=mask)
        self.assertTrue(np.allclose(y_t.data.numpy(), y.data.numpy()[:, -1]))
        # TODO write assertions

