from __future__ import print_function
from __future__ import print_function
from unittest import TestCase
from torch.autograd import Variable
from torch import nn
import torch
import numpy as np
import qelos as q


class TestGRU(TestCase):
    def test_gru_shapes(self):
        batsize = 5
        q.GRUCell.debug = True
        gru = q.GRUCell(9, 10)
        x_t = Variable(torch.FloatTensor(np.random.random((batsize, 9))))
        h_tm1 = Variable(torch.FloatTensor(np.random.random((batsize, 10))))
        gru.set_init_states(h_tm1)
        y_t = gru(x_t)
        self.assertEqual((5, 10), y_t.data.numpy().shape)

    # def test_gru_shapes(self):
    #     batsize = 5
    #     q.GRU.debug = True
    #     gru = q.GRU(9, 10)
    #     x_t = Variable(torch.FloatTensor(np.random.random((batsize, 9))))
    #     h_tm1 = Variable(torch.FloatTensor(np.random.random((batsize, 10))))
    #     gru.set_init_states(h_tm1)
    #     h_t, rg, ug = gru._forward(x_t, h_tm1)
    #     # simulate reset gate
    #     nprg = np.dot(x_t.datasets.numpy(), gru.reset_gate.W.datasets.numpy())
    #     nprg += np.dot(h_tm1.datasets.numpy(), gru.reset_gate.U.datasets.numpy())
    #     nprg += gru.reset_gate.b.datasets.numpy()
    #     self.assertTrue(np.allclose(rg, nprg))
    #     # simulate update gate
    #     npug = np.dot(x_t.datasets.numpy(), gru.update_gate.W.datasets.numpy())
    #     npug += np.dot(h_tm1.datasets.numpy(), gru.update_gate.U.datasets.numpy())
    #     npug += gru.update_gate.b.datasets.numpy()
    #     self.assertTrue(np.allclose(ug, npug))
    #     # output shape
    #     self.assertEqual((5, 10), h_t.datasets.numpy().shape)
    #     print(rg)
    #
    # def test_params_collected(self):
    #     gru = q.GRU(9, 10)
    #     param_gen = gru.parameters()
    #     params = []
    #     exp_params = set([gru.reset_gate.W, gru.reset_gate.U, gru.update_gate.W, gru.update_gate.U, gru.main_gate.W, gru.main_gate.U])
    #     for param in param_gen:
    #         for exp_param in exp_params:
    #             if param is exp_param:
    #                 exp_params = exp_params.difference(set([param]))
    #                 break
    #     self.assertTrue(len(exp_params) == 0)

    def test_zoneout(self):
        batsize = 5
        q.GRUCell.debug = False
        gru = q.GRUCell(9, 10, zoneout=0.5)
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
        # must not be equal in training mode
        self.assertFalse(np.allclose(pred1.data.numpy(), pred2.data.numpy()))

    def test_shared_dropout_rec(self):
        batsize = 5
        q.GRUCell.debug = False
        gru = q.GRUCell(9, 10, shared_dropout_rec=0.5)
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
        # must not be equal in training mode
        self.assertFalse(np.allclose(pred1.data.numpy(), pred2.data.numpy()))
        gru.reset_state()
        pred1 = gru(x_t)
        pred2 = gru(x_t)
        pred3 = gru(x_t)
        gru.train(mode=False)
        gru.reset_state()
        pred1 = gru(x_t)
        pred2 = gru(x_t)
        pred3 = gru(x_t)


class TestLSTM(TestCase):
    def test_lstm_shapes(self):
        batsize = 5
        lstm = q.LSTMCell(9, 10)
        x_t = Variable(torch.FloatTensor(np.random.random((batsize, 9))))
        h_tm1 = Variable(torch.FloatTensor(np.random.random((batsize, 10))))
        c_tm1 = Variable(torch.FloatTensor(np.random.random((batsize, 10))))
        lstm.set_init_states(c_tm1, h_tm1)
        y_t = lstm(x_t)
        self.assertEqual((5, 10), y_t.data.numpy().shape)

    def test_lstm_shapes_non_cudnn(self):
        batsize = 5
        lstm = q.LSTMCell(9, 10, use_cudnn_cell=False)
        x_t = Variable(torch.FloatTensor(np.random.random((batsize, 9))))
        h_tm1 = Variable(torch.FloatTensor(np.random.random((batsize, 10))))
        c_tm1 = Variable(torch.FloatTensor(np.random.random((batsize, 10))))
        lstm.set_init_states(c_tm1, h_tm1)
        y_t = lstm(x_t)
        self.assertEqual((5, 10), y_t.data.numpy().shape)

    def test_zoneout(self):
        batsize = 5
        q.LSTMCell.debug = False
        lstm = q.LSTMCell(9, 10, zoneout=0.5)
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
        q.LSTMCell.debug = False
        lstm = q.LSTMCell(9, 10)
        lstm = lstm.to_layer().return_all()
        x = Variable(torch.FloatTensor(np.random.random((batsize, seqlen, 9))))
        y = lstm(x)
        self.assertEqual((batsize, seqlen, 10), y.data.numpy().shape)

    def test_masked_gru(self):
        batsize = 3
        seqlen = 4
        q.GRUCell.debug = False
        gru = q.GRUCell(9, 10)
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
        q.GRUCell.debug = False
        gru = q.GRUCell(9, 10)
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
        q.GRUCell.debug = False
        gru = q.GRUCell(9, 5)
        gru2 = q.GRUCell(9, 5)
        layer = q.BiRNNLayer(gru, gru2, mode="cat").return_final().return_all()
        x = Variable(torch.FloatTensor(np.random.random((batsize, seqlen, 9))))
        m_val = np.asarray([[1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 1]])
        m = Variable(torch.FloatTensor(m_val))
        y_t, y = layer(x, mask=m)
        pred = y.data.numpy()
        # TODO write assertions


class TestRecStack(TestCase):
    def test_shapes(self):
        batsize = 5
        m = q.RecStack(q.GRUCell(9, 10), q.GRUCell(10, 11))
        x_t = Variable(torch.FloatTensor(np.random.random((batsize, 9))))
        h_tm1_a = Variable(torch.FloatTensor(np.random.random((batsize, 10))))
        h_tm1_b = Variable(torch.FloatTensor(np.random.random((batsize, 11))))
        m.set_init_states(h_tm1_a, h_tm1_b)
        y_t = m(x_t)
        self.assertEqual((batsize, 11), y_t.data.numpy().shape)

    def test_masked_gru_stack(self):
        batsize = 3
        seqlen = 4

        m = q.RecStack(q.GRUCell(9, 10), q.GRUCell(10, 11))
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


class TestGRULayer(TestCase):
    def test_shapes(self):
        batsize, seqlen, indim = 5, 3, 4
        m = q.GRULayer(indim, 6)
        data = Variable(torch.FloatTensor(np.random.random((batsize, seqlen, indim))))
        pred = m(data)
        print(pred)
        self.assertEqual((batsize, seqlen, 6), pred.size())
        self.assertEqual((batsize, 6), m.get_states(0)[0].size())

    def test_shapes_with_init_state(self):
        batsize, seqlen, indim = 5, 3, 4
        m = q.GRULayer(indim, 6)
        data = Variable(torch.FloatTensor(np.random.random((batsize, seqlen, indim))))
        h_0 = Variable(torch.FloatTensor(np.random.random((6,))))
        m.set_init_states(h_0)
        pred = m(data)
        print(pred)
        self.assertEqual((batsize, seqlen, 6), pred.size())
        self.assertEqual((batsize, 6), m.get_states(0)[0].size())

    def test_shapes_bidir(self):
        batsize, seqlen, indim = 5, 3, 4
        m = q.BidirGRULayer(indim, 6)
        data = Variable(torch.FloatTensor(np.random.random((batsize, seqlen, indim))))
        pred = m(data)
        print(pred)
        self.assertEqual((batsize, seqlen, 6*2), pred.size())
        self.assertEqual((batsize, 6), m.layer_fwd.get_states(0)[0].size())
        self.assertEqual((batsize, 6), m.layer_rev.get_states(0)[0].size())

    def test_mask(self):
        batsize, seqlen, indim = 5, 3, 4
        m = q.GRULayer(indim, 6).return_final()
        data = Variable(torch.FloatTensor(np.random.random((batsize, seqlen, indim))))
        mask = Variable(torch.LongTensor([[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 0, 0], ]))
        final, pred = m(data, mask=mask)
        print(pred)
        # self.assertFalse(True)
        self.assertEqual((batsize, seqlen, 6), pred.size())
        self.assertEqual((batsize, 6), m.get_states(0)[0].size())
        pred = pred.data.numpy()
        final = final.data.numpy()

        self.assertTrue(np.allclose(pred[0, 2, :], np.zeros_like(pred[0, 2, :])))
        self.assertTrue(np.allclose(pred[4, 1:, :], np.zeros_like(pred[4, 1:, :])))

        self.assertTrue(np.allclose(final[0, :], pred[0, 1, :]))
        self.assertTrue(np.allclose(final[4, :], pred[4, 0, :]))
        self.assertTrue(np.allclose(final[3, :], pred[3, 2, :]))

    def test_mask_bidir(self):
        batsize, seqlen, indim = 5, 3, 4
        m = q.BidirGRULayer(indim, 6).return_final()
        data = Variable(torch.FloatTensor(np.random.random((batsize, seqlen, indim))))
        mask = Variable(torch.LongTensor([[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 0, 0], ]))
        final, pred = m(data, mask=mask)
        print(pred)
        # self.assertFalse(True)
        self.assertEqual((batsize, seqlen, 6*2), pred.size())
        self.assertEqual((batsize, 6), m.layer_fwd.get_states(0)[0].size())
        pred = pred.data.numpy()
        final = final.data.numpy()

        self.assertTrue(np.allclose(pred[0, 2, :], np.zeros_like(pred[0, 2, :])))
        self.assertTrue(np.allclose(pred[4, 1:, :], np.zeros_like(pred[4, 1:, :])))

        self.assertTrue(np.allclose(final[0, :], pred[0, 1, :]))
        self.assertTrue(np.allclose(final[4, :], pred[4, 0, :]))
        self.assertTrue(np.allclose(final[3, :], pred[3, 2, :]))


class TestLSTMLayer(TestCase):
    def test_shapes(self):
        batsize, seqlen, indim = 5, 3, 4
        m = q.LSTMLayer(indim, 6)
        data = Variable(torch.FloatTensor(np.random.random((batsize, seqlen, indim))))
        pred = m(data)
        print(pred)
        self.assertEqual((batsize, seqlen, 6), pred.size())
        self.assertEqual((batsize, 6), m.get_states(0)[0].size())
        self.assertEqual((batsize, 6), m.get_states(0)[1].size())

    def test_shapes_bidir(self):
        batsize, seqlen, indim = 5, 3, 4
        m = q.BidirLSTMLayer(indim, 6)
        data = Variable(torch.FloatTensor(np.random.random((batsize, seqlen, indim))))
        pred = m(data)
        print(pred)
        self.assertEqual((batsize, seqlen, 6*2), pred.size())
        self.assertEqual((batsize, 6), m.layer_fwd.get_states(0)[0].size())
        self.assertEqual((batsize, 6), m.layer_fwd.get_states(0)[1].size())
        self.assertEqual((batsize, 6), m.layer_fwd.get_states(0)[0].size())
        self.assertEqual((batsize, 6), m.layer_fwd.get_states(0)[1].size())


class TestRecurrentStack(TestCase):
    def test_shapes(self):
        batsize, seqlen, vocsize, embdim, encdim = 5, 3, 20, 4, 6
        m = q.RecurrentStack(
            nn.Embedding(vocsize, embdim),
            q.GRULayer(embdim, encdim),
            q.Forward(encdim, vocsize),
            q.LogSoftmax()
        )
        data = Variable(torch.LongTensor(np.random.randint(0, vocsize, (batsize, seqlen))))
        pred = m(data)
        print(pred)
        self.assertEqual((batsize, seqlen, vocsize), pred.size())
        self.assertEqual((batsize, encdim), m.get_states(0)[0].size())