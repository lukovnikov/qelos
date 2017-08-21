from unittest import TestCase
import qelos as q
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


class TestAttention(TestCase):
    def test_forward_attgen(self):
        batsize, seqlen, datadim, critdim, attdim = 5, 3, 4, 3, 7
        crit = Variable(torch.FloatTensor(np.random.random((batsize, critdim))))
        data = Variable(torch.FloatTensor(np.random.random((batsize, seqlen, datadim))))
        att = q.Attention().forward_gen(datadim, critdim, attdim)
        m = att.attgen
        pred = m(data, crit)
        pred = pred.data.numpy()
        self.assertEqual(pred.shape, (batsize, seqlen))
        self.assertTrue(np.allclose(np.sum(pred, axis=1), np.ones((pred.shape[0],))))

    def test_att_splitter(self):
        batsize, seqlen, datadim, critdim, attdim = 5, 3, 4, 3, 7
        crit = torch.FloatTensor(np.random.random((batsize, critdim)))
        data = torch.FloatTensor(np.random.random((batsize, seqlen, datadim)))
        att = q.Attention().forward_gen(datadim, critdim, attdim).split_data()
        attgendata = att.attgen.data_selector(data).numpy()
        attcondata = att.attcon.data_selector(data).numpy()
        recdata = np.concatenate([attgendata, attcondata], axis=2)
        self.assertTrue(np.allclose(data.numpy(), recdata))

    def test_forward_attgen_w_mask(self):
        batsize, seqlen, datadim, critdim, attdim = 5, 6, 4, 3, 7
        crit = Variable(torch.FloatTensor(np.random.random((batsize, critdim))))
        data = Variable(torch.FloatTensor(np.random.random((batsize, seqlen, datadim))))
        maskstarts = np.random.randint(1, seqlen, (batsize,))
        mask = np.ones((batsize, seqlen), dtype="int32")
        for i in range(batsize):
            mask[i, maskstarts[i]:] = 0
        mask = Variable(torch.FloatTensor(mask*1.))
        att = q.Attention().forward_gen(datadim, critdim, attdim)
        m = att.attgen
        pred = m(data, crit, mask=mask)
        pred = pred.data.numpy()
        print(pred)
        self.assertEqual(pred.shape, (batsize, seqlen))
        self.assertTrue(np.allclose(mask.data.numpy(), pred > 0))
        self.assertTrue(np.allclose(np.sum(pred, axis=1), np.ones((pred.shape[0],))))

    def test_forward_attgen_w_mask_3D(self):
        batsize, seqlen, datadim, critdim, attdim, critseqlen = 5, 6, 4, 3, 7, 8
        crit = Variable(torch.FloatTensor(np.random.random((batsize, critseqlen, critdim))))
        data = Variable(torch.FloatTensor(np.random.random((batsize, seqlen, datadim))))
        maskstarts = np.random.randint(1, seqlen, (batsize,))
        mask = np.ones((batsize, seqlen), dtype="int32")
        for i in range(batsize):
            mask[i, maskstarts[i]:] = 0
        mask = Variable(torch.FloatTensor(mask*1.))
        att = q.Attention().forward_gen(datadim, critdim, attdim)
        m = att.attgen
        pred = m(data, crit, mask=mask)
        pred = pred.data.numpy()
        print(pred)
        self.assertEqual(pred.shape, (batsize, critseqlen, seqlen))
        self.assertTrue(np.allclose(mask[:, None, :].repeat(1, critseqlen, 1).data.numpy(), pred > 0))
        self.assertTrue(np.allclose(np.sum(pred, axis=2), np.ones((pred.shape[0],pred.shape[1]))))
