from unittest import TestCase
from qelos.rnn import GRU, LSTM, BiRNNLayer
from torch.autograd import Variable
import torch
import numpy as np


class TestGRU(TestCase):
    def test_gru_shapes(self):
        batsize = 5
        GRU.debug = True
        gru = GRU(9, 10)
        x_t = Variable(torch.FloatTensor(np.random.random((batsize, 9))))
        h_tm1 = Variable(torch.FloatTensor(np.random.random((batsize, 10))))
        h_t, rg, ug = gru(x_t, h_tm1)
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
        gru = GRU(9, 10)
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
        GRU.debug = False
        gru = GRU(9, 10, zoneout=0.5)
        x_t = Variable(torch.FloatTensor(np.random.random((batsize, 9))))
        h_tm1 = Variable(torch.FloatTensor(np.random.random((batsize, 10))))
        h_t, y_t = gru(x_t, h_tm1)
        self.assertEqual((5, 10), h_t.data.numpy().shape)
        self.assertEqual(gru.training, True)
        gru.train(mode=False)
        self.assertEqual(gru.training, False)
        pred1, _ = gru(x_t, h_tm1)
        pred2, _ = gru(x_t, h_tm1)
        # must be equal in prediction mode
        self.assertTrue(np.allclose(pred1.data.numpy(), pred2.data.numpy()))
        gru.train(mode=True)
        self.assertEqual(gru.training, True)
        pred1, _ = gru(x_t, h_tm1)
        pred2, _ = gru(x_t, h_tm1)
        # must not be equal in prediction mode
        self.assertFalse(np.allclose(pred1.data.numpy(), pred2.data.numpy()))


class TestLSTM(TestCase):
    def test_lstm_shapes(self):
        batsize = 5
        LSTM.debug = True
        lstm = LSTM(9, 10)
        x_t = Variable(torch.FloatTensor(np.random.random((batsize, 9))))
        h_tm1 = Variable(torch.FloatTensor(np.random.random((batsize, 10))))
        c_tm1 = Variable(torch.FloatTensor(np.random.random((batsize, 10))))
        y_t, c_t, h_t = lstm(x_t, h_tm1, c_tm1)
        self.assertEqual((5, 10), h_t.data.numpy().shape)

    def test_zoneout(self):
        batsize = 5
        LSTM.debug = False
        lstm = LSTM(9, 10, zoneout=0.5)
        x_t = Variable(torch.FloatTensor(np.random.random((batsize, 9))))
        h_tm1 = Variable(torch.FloatTensor(np.random.random((batsize, 10))))
        c_tm1 = Variable(torch.FloatTensor(np.random.random((batsize, 10))))
        y_t, c_t, h_t = lstm(x_t, h_tm1, c_tm1)
        self.assertEqual((5, 10), h_t.data.numpy().shape)
        self.assertEqual(lstm.training, True)
        lstm.train(mode=False)
        self.assertEqual(lstm.training, False)
        pred1, pred1e, _ = lstm(x_t, h_tm1, c_tm1)
        pred2, pred2e, _ = lstm(x_t, h_tm1, c_tm1)
        # must be equal in prediction mode
        self.assertTrue(np.allclose(pred1.data.numpy(), pred2.data.numpy()))
        lstm.train(mode=True)
        self.assertEqual(lstm.training, True)
        pred1, pred1e, _ = lstm(x_t, h_tm1, c_tm1)
        pred2, pred2e, _ = lstm(x_t, h_tm1, c_tm1)
        # must not be equal in prediction mode
        self.assertFalse(np.allclose(pred1.data.numpy(), pred2.data.numpy()))


class Test_RNNLayer(TestCase):
    def test_lstm_layer_shapes(self):
        batsize = 5
        seqlen = 7
        LSTM.debug = False
        lstm = LSTM(9, 10)
        lstm = lstm.to_layer().return_all()
        x = Variable(torch.FloatTensor(np.random.random((batsize, seqlen, 9))))
        y = lstm(x)
        self.assertEqual((batsize, seqlen, 10), y.data.numpy().shape)

    def test_masked_gru(self):
        batsize = 3
        seqlen = 4
        GRU.debug = False
        gru = GRU(9, 10)
        gru = gru.to_layer().return_all()
        x = Variable(torch.FloatTensor(np.random.random((batsize, seqlen, 9))))
        m_val = np.asarray([[1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 1]])
        m = Variable(torch.FloatTensor(m_val))
        y = gru(x, mask=m)
        pred = y.data.numpy()
        # TODO write assertions

    def test_masked_gru_reverse(self):
        batsize = 3
        seqlen = 4
        GRU.debug = False
        gru = GRU(9, 10)
        gru = gru.to_layer().return_all().return_final()
        x = Variable(torch.FloatTensor(np.random.random((batsize, seqlen, 9))))
        m_val = np.asarray([[1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 1]])
        m = Variable(torch.FloatTensor(m_val))
        y_t, y = gru(x, mask=m, reverse=True)
        pred = y.data.numpy()
        # TODO write assertions

    def test_masked_gru_bidir(self):
        batsize = 3
        seqlen = 4
        GRU.debug = False
        gru = GRU(9, 5)
        gru2 = GRU(9, 5)
        layer = BiRNNLayer(gru, gru2, mode="cat").return_final().return_all()
        x = Variable(torch.FloatTensor(np.random.random((batsize, seqlen, 9))))
        m_val = np.asarray([[1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 1]])
        m = Variable(torch.FloatTensor(m_val))
        y_t, y = layer(x, mask=m)
        pred = y.data.numpy()
        # TODO write assertions

