from __future__ import print_function
from unittest import TestCase
from qelos.basic import Softmax, LogSoftmax, SoftmaxLog
import torch
from torch.autograd import Variable
import numpy as np

from teafacto.core.base import tensorops as T, Val


class TestSoftmax(TestCase):
    def test_softmax_normal(self):
        b = Softmax()
        d = Variable(torch.FloatTensor(np.random.random((5, 3))))
        pred = b(d).data.numpy()
        predsums = np.sum(pred, axis=1)
        self.assertTrue(np.allclose(predsums, np.ones_like(predsums)))
        self.assertEqual(d.size(), pred.shape)

    def test_softmax_3D(self):
        b = Softmax()
        d = Variable(torch.FloatTensor(np.random.random((5, 4, 3))))
        pred = b(d).data.numpy()
        predsums = np.sum(pred, axis=2)
        self.assertTrue(np.allclose(predsums, np.ones_like(predsums)))
        self.assertEqual(d.size(), pred.shape)

    def test_softmax_5D(self):
        b = Softmax()
        d = Variable(torch.FloatTensor(np.random.random((7, 6, 5, 4, 3))))
        pred = b(d).data.numpy()
        predsums = np.sum(pred, axis=-1)
        self.assertTrue(np.allclose(predsums, np.ones_like(predsums)))
        self.assertEqual(d.size(), pred.shape)

    def test_softmax_normal_masked(self):
        b = Softmax()
        d = Variable(torch.FloatTensor(np.random.random((5, 3))))
        m = np.ones_like(d.data.numpy())
        m[:, 2] = 0
        m = Variable(torch.FloatTensor(m))
        pred = b(d, m).data.numpy()
        print(pred)
        self.assertTrue(np.allclose(np.zeros_like(pred[:, 2]), pred[:, 2]))
        self.assertEqual(d.size(), pred.shape)
        predsums = np.sum(pred, axis=-1)
        self.assertTrue(np.allclose(predsums, np.ones_like(predsums)))

    def test_softmax_3D_masked(self):
        b = Softmax()
        d = Variable(torch.FloatTensor(np.random.random((5, 4, 3))))
        m = np.ones_like(d.data.numpy())
        m[:, :, 2] = 0
        m = Variable(torch.FloatTensor(m))
        pred, mask = b(d, m)
        pred = pred.data.numpy()
        print(pred)
        self.assertTrue(np.allclose(np.zeros_like(pred[:, :, 2]), pred[:, :, 2]))
        self.assertEqual(d.size(), pred.shape)
        predsums = np.sum(pred, axis=-1)
        self.assertTrue(np.allclose(predsums, np.ones_like(predsums)))

    def test_softmax_3D_masked_equals_teafacto(self):       # TODO remove
        torch_model = Softmax()
        torch_data = Variable(torch.FloatTensor(np.random.random((5, 4, 3))))
        teafa_data = Val(torch_data.data.numpy())
        m = np.ones_like(torch_data.data.numpy())
        m[:, :, 2] = 0
        torch_mask = Variable(torch.FloatTensor(m))
        teafa_mask = Val(m)
        torch_pred, _ = torch_model(torch_data, torch_mask)
        torch_pred = torch_pred.data.numpy()
        teafa_pred = T.softmax(teafa_data, teafa_mask).eval()
        self.assertTrue(np.allclose(torch_pred, teafa_pred))

    def test_softmax_3D_prop_seq_mask(self):
        b = Softmax()
        d = Variable(torch.FloatTensor(np.random.random((5, 4, 3))))
        m = np.ones((5, 4))
        m[:, 2:] = 0
        m = Variable(torch.FloatTensor(m))
        pred, mask = b(d, m)
        predmask = mask.data.numpy()
        pred = pred.data.numpy()
        self.assertTrue(np.allclose(predmask, m.data.numpy()))
        predsums = np.sum(pred, axis=-1)
        self.assertTrue(np.allclose(predsums, np.ones_like(predsums)))

    def test_softmax_normal_with_temperature(self):
        b = Softmax(temperature=1e-6)
        d = Variable(torch.FloatTensor(np.random.random((5, 3))))
        pred = b(d).data.numpy()
        print(pred)

    def test_masked_softmax_numerical_stability(self):
        d = Variable(torch.FloatTensor(np.asarray([[-1e9, 1e9, 1], [-1e6, 1e6, 1], [-1e3, 1e3, 1], [-1e2, 1e2, 1], [-1e1, 1e1, 1], [-1, 1e2, 1], [1, 1e2, 1], [0.5, 1e2, 1]])))
        m = Variable(torch.FloatTensor(np.asarray([[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1]])))
        d2 = d[:, [0, 2]]
        o = Softmax()(d, m)
        pred = o.data.numpy()
        pred2 = Softmax()(d2).data.numpy()
        print(pred)
        print(pred2)
        self.assertTrue(np.allclose(pred[:, 1], np.zeros_like(pred[:, 1])))
        self.assertTrue(np.allclose(pred[:, [0, 2]], pred2))


class TestLogsoftmax(TestCase):
    def test_logsoftmax_normal(self):
        b = LogSoftmax()
        d = Variable(torch.FloatTensor(np.random.random((5, 3))))
        pred = b(d).data.numpy()
        predsums = np.sum(np.exp(pred), axis=1)
        self.assertTrue(np.allclose(predsums, np.ones_like(predsums)))
        self.assertEqual(d.size(), pred.shape)

    def test_logsoftmax_masked(self):
        b = LogSoftmax()
        d = Variable(torch.FloatTensor(np.random.random((5, 3))))
        m = np.ones_like(d.data.numpy())
        m[:, 2] = 0
        m = Variable(torch.FloatTensor(m))
        pred = b(d, m).data.numpy()
        print(pred)
        self.assertTrue(np.allclose(np.zeros_like(pred[:, 2]), np.exp(pred)[:, 2]))
        self.assertEqual(d.size(), pred.shape)
        predsums = np.sum(np.exp(pred), axis=-1)
        print(predsums)
        self.assertTrue(np.allclose(predsums, np.ones_like(predsums)))

    def test_logsoftmax_masked_same_as_softmax(self):
        lsm = LogSoftmax()
        d = Variable(torch.FloatTensor(np.random.random((5, 3))))
        m = np.ones_like(d.data.numpy())
        m[:, 2] = 0
        m = Variable(torch.FloatTensor(m))
        pred = lsm(d, m).data.numpy()
        print(pred)
        self.assertTrue(np.allclose(np.zeros_like(pred[:, 2]), np.exp(pred)[:, 2]))
        self.assertEqual(d.size(), pred.shape)
        predsums = np.sum(np.exp(pred), axis=-1)
        print(predsums)
        self.assertTrue(np.allclose(predsums, np.ones_like(predsums)))
        predexp = Softmax()(d, m).data.numpy()
        self.assertTrue(np.allclose(predexp, np.exp(pred)))

    def test_masked_logsoftmax_numerical_stability(self):
        d = Variable(torch.FloatTensor(np.asarray([[-1e9, 1e9, 1], [-1e6, 1e6, 1], [-1e3, 1e3, 1], [-1e2, 1e2, 1], [-1e1, 1e1, 1], [-1, 1e2, 1], [1, 1e2, 1], [0.5, 1e2, 1]])))
        m = Variable(torch.FloatTensor(np.asarray([[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1]])))
        d2 = d[:, [0, 2]]
        o = LogSoftmax()(d, m)
        pred = o.data.numpy()
        pred2 = LogSoftmax()(d2).data.numpy()
        print(pred)
        print(pred2)
        self.assertTrue(np.allclose(pred[:, 1], np.log(np.zeros_like(pred[:, 1]))))
        self.assertTrue(np.allclose(pred[:, [0, 2]], pred2, rtol=1e-3))



