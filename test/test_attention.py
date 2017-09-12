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
        self.assertTrue(np.allclose(np.sum(pred, axis=2), np.ones((pred.shape[0],pred.shape[1]))))
        self.assertTrue(np.allclose(mask[:, None, :].repeat(1, critseqlen, 1).data.numpy(), pred > 0))


from qelos.aiayn import ScaledDotProductAttention


class TestScaledDotProductAttention(TestCase):
    def test_equivalent_to_qelos(self):
        m = ScaledDotProductAttention(10, attn_dropout=0)
        refm = q.Attention().dot_gen().scale_pow(0.5)

        Q = q.var(np.random.random((5,4,10)).astype("float32")).v
        K = q.var(np.random.random((5,6,10)).astype("float32")).v
        V = q.var(np.random.random((5,6,11)).astype("float32")).v

        ctx, atn = m(Q, K, V)
        refatn = refm.attgen(K, Q)
        refctx = refm.attcon(V, refatn)

        print(atn)
        print(refatn)

        self.assertTrue(np.allclose(atn.data.numpy(), refatn.data.numpy()))
        self.assertTrue(np.allclose(ctx.data.numpy(), refctx.data.numpy()))

    def test_equivalent_to_qelos_masked(self):
        m = ScaledDotProductAttention(10, attn_dropout=0)
        refm = q.Attention().dot_gen().scale_pow(0.5)

        Q = q.var(np.random.random((5, 1, 10)).astype("float32")).v
        K = q.var(np.random.random((5, 6, 10)).astype("float32")).v
        M = q.var(np.asarray([
                              [1, 0, 0, 0, 0, 0],
                              [1, 1, 1, 0, 0, 0],
                              [1, 1, 1, 0, 0, 0],
                              [1, 1, 1, 1, 0, 0],
                              [1, 1, 1, 1, 1, 1],])).v
        V = q.var(np.random.random((5, 6, 11)).astype("float32")).v

        ctx, atn = m(Q, K, V, attn_mask=(-1*M+1).byte().data.unsqueeze(1))
        refatn = refm.attgen(K, Q, mask=M)
        refctx = refm.attcon(V, refatn)

        print(atn)
        print(refatn)

        self.assertTrue(np.allclose(atn.data.numpy(), refatn.data.numpy()))
        self.assertTrue(np.allclose(ctx.data.numpy(), refctx.data.numpy()))


from qelos.aiayn import MultiHeadAttention


class TestMultiHeadAttention(TestCase):
    def test_equivalent_to_qelos(self):
        m = MultiHeadAttention(4, 16, 10, 12, 0)
        mym = q.MultiHeadAttention(4, 16, 10, 12, 0)
        mym.w_qs, mym.w_ks, mym.w_vs = m.w_qs, m.w_ks, m.w_vs
        mym.proj, mym.layer_norm = m.proj, m.layer_norm

        Q = q.var(np.random.random((5, 1, 16)).astype("float32")).v
        K = q.var(np.random.random((5, 6, 16)).astype("float32")).v
        M = q.var(np.asarray([
                              [1, 0, 0, 0, 0, 0],
                              [1, 1, 1, 0, 0, 0],
                              [1, 1, 1, 0, 0, 0],
                              [1, 1, 1, 1, 0, 0],
                              [1, 1, 1, 1, 1, 1],])).v
        V = q.var(np.random.random((5, 6, 16)).astype("float32")).v

        outs, atts = m(Q, K, V, (-1*M+1).byte().data.unsqueeze(1))

        self.assertEqual(outs.size(), (5, 1, 16))
        self.assertEqual(atts.size(), (20, 1, 6))

        myouts, myatts = mym(Q, K, V, M)

        m_em = q.get_emitted("mha")
        mym_em = q.get_emitted("mymha")
        for k in m_em:
            self.assertTrue(np.allclose(m_em[k].data.numpy(), mym_em[k].data.numpy()))

        self.assertTrue(np.allclose(myouts.data.numpy(), outs.data.numpy()))
        self.assertTrue(np.allclose(myatts.data.numpy(), atts.data.numpy()))

