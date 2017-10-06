from __future__ import print_function
from unittest import TestCase
import qelos as q
from torch.autograd import Variable
import torch
from torch import nn
import numpy as np


class TestEmptyWordEmb(TestCase):
    def test_it(self):
        dic = dict(zip(map(chr, range(97, 122)), range(1,122-97+1)))
        dic[q.WordEmb.masktoken] = 0
        m = q.ZeroWordEmb(10, worddic=dic)
        embedding, mask = m(Variable(torch.LongTensor([0,1,2])))
        self.assertEqual(embedding.size(), (3, 10))
        self.assertTrue(np.allclose(mask.data.numpy(), [0,1,1]))
        self.assertTrue(np.allclose(np.zeros((3, 10)), embedding.data.numpy()))

    def test_overridden(self):
        dic = dict(zip(map(chr, range(97, 122)), range(1,122-97+1)))
        dic[q.WordEmb.masktoken] = 0
        m = q.ZeroWordEmb(10, worddic=dic)
        dic = dict(zip(map(chr, range(97, 122)), range(0, 122 - 97)))
        mo = q.WordEmb(10, worddic=dic)
        moe = m.override(mo)
        emb, mask = moe(Variable(torch.LongTensor([0,1,2])))
        self.assertEqual(emb.size(), (3, 10))
        self.assertTrue(np.allclose(mask.data.numpy(), [0,1,1]))
        self.assertTrue(np.allclose(emb[0].data.numpy(), np.zeros((10,))))
        oemb, mask = mo(Variable(torch.LongTensor([0,0,1])))
        self.assertEqual(oemb.size(), (3, 10))
        self.assertTrue(mask is None)
        self.assertTrue(np.allclose(oemb.data.numpy()[1:], emb.data.numpy()[1:]))


class TestWordEmb(TestCase):
    def test_creation_simple(self):
        dic = dict(zip(map(chr, range(97, 122)), range(122-97)))
        m = q.WordEmb(10, worddic=dic)
        embedding, _ = m(Variable(torch.LongTensor([0,1,2])))
        self.assertEqual(embedding.size(), (3, 10))
        trueemb = m.embedding.weight.data.cpu().numpy()[0]
        self.assertTrue(np.allclose(trueemb, embedding[0].data.numpy()))

    def test_creation_masked(self):
        dic = dict(zip(map(chr, range(97, 122)), range(1, 122-97+1)))
        dic[q.WordEmb.masktoken] = 0
        m = q.WordEmb(10, worddic=dic)
        embedding, mask = m(Variable(torch.LongTensor([0, 1, 2])))
        self.assertEqual(embedding.size(), (3, 10))
        trueemb = m.embedding.weight.data.cpu().numpy()[1]
        self.assertTrue(np.allclose(trueemb, embedding[1].data.numpy()))
        self.assertTrue(np.allclose(embedding[0].data.numpy(), np.zeros((10,))))
        print(mask)
        self.assertTrue(np.allclose(mask.data.numpy(), [0,1,1]))


class TestAdaptedWordEmb(TestCase):
    def setUp(self):
        wdic = {"<MASK>": 0, "<RARE>": 1, "the": 10, "a": 5, "his": 50, "abracadabrqmsd--qsdfmqgf-": 6}
        wdic2 = {"<MASK>": 0, "<RARE>": 1, "the": 2, "a": 3, "his": 4, "abracadabrqmsd--qsdfmqgf-": 5, "qsdfqsdf": 7}
        self.adapted = q.WordEmb(50, worddic=wdic)
        self.vanilla = q.WordEmb(50, worddic=wdic, value=self.adapted.embedding.weight.data.numpy())
        self.adapted = self.adapted.adapt(wdic2)

    def test_map(self):
        self.assertEqual(self.adapted * "a", 3)
        self.assertEqual(self.adapted * "the", 2)
        self.assertEqual(self.adapted * "his", 4)
        self.assertEqual(self.adapted * "her", 1)
        self.assertEqual(self.vanilla * "a", 5)
        self.assertEqual(self.vanilla * "the", 10)
        self.assertEqual(self.vanilla * "her", 1)
        self.assertEqual(self.adapted * "qsdfqlmkdsjfmqlsdkjgmqlsjdf", 1)
        print(self.vanilla * "the", self.adapted * "the")
        self.assertTrue(np.allclose(self.vanilla % "the", self.adapted % "the"))
        self.assertTrue(np.allclose(self.vanilla % "his", self.adapted % "his"))

    def test_adapted_block(self):
        pred, mask = self.adapted(Variable(torch.LongTensor([self.adapted * x for x in "the a his".split()])))
        l = pred.sum()
        l.backward()
        grad = self.adapted.inner.embedding.weight.grad
        self.assertTrue(grad.norm().data.numpy()[0] > 0)

        vpred = np.asarray([self.vanilla % x for x in "the a his".split()])
        self.assertTrue(np.allclose(pred.data.numpy(), vpred))

        oovpred, mask = self.adapted(Variable(torch.LongTensor([6, 7])))  # two different kinds of OOV
        print(self.adapted % 6)
        print(self.vanilla % self.vanilla.raretoken)
        # TODO self.assertTrue(np.allclose(oovpred.datasets.numpy(), np.zeros_like(oovpred.datasets.numpy())))

    def test_adapted_prediction_shape(self):
        xval = np.random.randint(0, 3, (5, 4))
        x = Variable(torch.from_numpy(xval))
        pred, mask = self.adapted(x)
        self.assertEqual(pred.size(), (5, 4, 50))
        self.assertEqual(mask.size(), (5, 4))
        self.assertTrue(np.allclose(mask.data.numpy(), xval != 0))


class TestWordEmbOverriding(TestCase):
    def setUp(self):
        words = "<MASK> <RARE> the a his monkey inception key earlgrey"
        wdic = dict(zip(words.split(), range(0, len(words.split()))))
        overwords = "he his her mine cat monkey the interstellar grey key"
        overwdic = dict(zip(overwords.split(), range(0, len(overwords.split()))))
        self.baseemb = q.WordEmb(dim=50, worddic=wdic)
        self.overemb = q.WordEmb(dim=50, worddic=overwdic)
        self.emb = self.baseemb.override(self.overemb)
        pass

    def test_embed_masker(self):
        v = Variable(torch.from_numpy(np.random.randint(0, 5, (4, 3))))
        m, mask = self.emb(v)
        self.assertTrue(np.all((v.data.numpy() != 0) == mask.data.numpy()))

    def test_sameasover(self):
        words = "the his monkey key"
        pred, msk = self.emb(q.var(torch.LongTensor([self.emb * x for x in words.split()])).v)
        pred = pred.data.numpy()
        gpred, _ = self.overemb(q.var(torch.LongTensor([self.overemb * x for x in words.split()])).v)
        gpred = gpred.data.numpy()
        self.assertTrue(np.allclose(pred, gpred))

    def test_sameasbase(self):
        words = "inception earlgrey <MASK>"
        pred, mask = self.emb(q.var(torch.LongTensor([self.emb * x for x in words.split()])).v)
        pred = pred.data.numpy()
        gpred, msk = self.baseemb(q.var(torch.LongTensor([self.baseemb * x for x in words.split()])).v)
        gpred = gpred.data.numpy()
        self.assertTrue(np.allclose(pred, gpred))

    def test_notasover(self):
        words = "inception earlgrey"
        pred, mask = self.emb(q.var(torch.LongTensor([self.emb * x for x in words.split()])).v)
        pred = pred.data.numpy()
        gpred, _ = self.overemb(q.var(torch.LongTensor([self.baseemb * x for x in words.split()])).v)
        gpred = gpred.data.numpy()
        self.assertFalse(np.allclose(pred, gpred))

    def test_notasbase(self):
        words = "the his monkey key"
        pred, mask = self.emb(q.var(torch.LongTensor([self.emb * x for x in words.split()])).v)
        pred = pred.data.numpy()
        gpred, msk = self.baseemb(q.var(torch.LongTensor([self.baseemb * x for x in words.split()])).v)
        gpred = gpred.data.numpy()
        self.assertFalse(np.allclose(pred, gpred))


class TestGlove(TestCase):
    def setUp(self):
        q.PretrainedWordEmb.defaultpath = "../data/glove/miniglove.%dd"
        self.glove = q.PretrainedWordEmb(50)
        print(self.glove.defaultpath)

    def test_loaded(self):
        thevector = self.glove % "the"
        truevector = np.asarray([  4.18000013e-01,   2.49679998e-01,  -4.12420005e-01,
         1.21699996e-01,   3.45270008e-01,  -4.44569997e-02,
        -4.96879995e-01,  -1.78619996e-01,  -6.60229998e-04,
        -6.56599998e-01,   2.78430015e-01,  -1.47670001e-01,
        -5.56770027e-01,   1.46579996e-01,  -9.50950012e-03,
         1.16579998e-02,   1.02040000e-01,  -1.27920002e-01,
        -8.44299972e-01,  -1.21809997e-01,  -1.68009996e-02,
        -3.32789987e-01,  -1.55200005e-01,  -2.31309995e-01,
        -1.91809997e-01,  -1.88230002e+00,  -7.67459989e-01,
         9.90509987e-02,  -4.21249986e-01,  -1.95260003e-01,
         4.00710011e+00,  -1.85939997e-01,  -5.22870004e-01,
        -3.16810012e-01,   5.92130003e-04,   7.44489999e-03,
         1.77780002e-01,  -1.58969998e-01,   1.20409997e-02,
        -5.42230010e-02,  -2.98709989e-01,  -1.57490000e-01,
        -3.47579986e-01,  -4.56370004e-02,  -4.42510009e-01,
         1.87849998e-01,   2.78489990e-03,  -1.84110001e-01,
        -1.15139998e-01,  -7.85809994e-01])
        self.assertEqual(self.glove * "the", 2)
        self.assertTrue(np.allclose(thevector, truevector))
        self.assertEqual(self.glove.embedding.weight.size(), (4002, 50))


class TestComputedWordEmb(TestCase):
    def setUp(self):
        data = np.random.random((7, 10)).astype("float32")
        computer = nn.Linear(10, 15)
        worddic = "<MASK> <RARE> first second third fourth fifth"
        worddic = dict(zip(worddic.split(), range(len(worddic.split()))))
        self.emb = q.ComputedWordEmb(data=data, computer=computer, worddic=worddic)

    def test_shape(self):
        x = Variable(torch.LongTensor([0, 1, 2]))
        emb, msk = self.emb(x)
        print(msk)
        self.assertEqual(emb.size(), (3, 15))
        self.assertTrue(np.allclose(msk.data.numpy(), [[0,1,1]]))


class TestMergedWordEmb(TestCase):
    def setUp(self):
        worddic = "<MASK> <RARE> first second third fourth fifth"
        worddic = dict(zip(worddic.split(), range(len(worddic.split()))))
        self.emb1 = q.WordEmb(100, worddic=worddic)
        self.emb2 = q.WordEmb(100, worddic=worddic)

    def test_sum_merge(self):
        emb = self.emb1.merge(self.emb2, mode="sum")
        x = Variable(torch.LongTensor([0, 1, 2]))
        emb1res, msk1 = self.emb1(x)
        print(msk1)
        emb2res, msk2 = self.emb2(x)
        embres, msk = emb(x)
        self.assertTrue(np.allclose(embres.data.numpy(), emb1res.data.numpy() + emb2res.data.numpy()))

    def test_cat_merge(self):
        emb = self.emb1.merge(self.emb2, mode="cat")
        x = Variable(torch.LongTensor([0, 1, 2]))
        emb1res, msk1 = self.emb1(x)
        print(msk1)
        emb2res, msk2 = self.emb2(x)
        embres, msk = emb(x)
        self.assertTrue(np.allclose(embres.data.numpy(), np.concatenate([emb1res.data.numpy(), emb2res.data.numpy()], axis=1)))


class TestZeroWordLinout(TestCase):
    def setUp(self):
        worddic = "<MASK> <RARE> first second third fourth fifth"
        worddic = dict(zip(worddic.split(), range(len(worddic.split()))))
        self.linout = q.ZeroWordLinout(10, worddic=worddic)

    def test_it(self):
        x = Variable(torch.randn(7, 10))
        msk = Variable(torch.FloatTensor([[1,0,1,1,0,1,0]]*5 + [[0,1,0,0,1,0,1]]*2))
        y = self.linout(x, mask=msk)
        print(y)
        self.assertEqual(y.size(), (7, 7))
        self.assertTrue(np.allclose(y.data.numpy(), np.zeros_like(y.data.numpy())))

    def test_overridden(self):
        worddic = "second third fourth fifth"
        worddic = dict(zip(worddic.split(), range(len(worddic.split()))))
        linout = q.WordLinout(10, worddic=worddic)
        l = self.linout.override(linout)
        x = Variable(torch.randn(7, 10))
        msk = Variable(torch.FloatTensor([[1,0,1,1,0,1,0]]*5 + [[0,1,0,0,1,0,1]]*2))
        y = l(x, mask=msk)
        print(y)




class TestWordLinout(TestCase):
    def setUp(self):
        worddic = "<MASK> <RARE> first second third fourth fifth"
        worddic = dict(zip(worddic.split(), range(len(worddic.split()))))
        self.linout = q.WordLinout(10, worddic=worddic)

    def test_shape(self):
        x = Variable(torch.randn(7, 10))
        msk = Variable(torch.FloatTensor([[1,0,1,1,0,1,0]]*5 + [[0,1,0,0,1,0,1]]*2))
        y = self.linout(x, mask=msk)
        print(y)
        self.assertEqual(y.size(), (7, 7))
        # self.assertTrue(False)


class TestCosineWordLinout(TestCase):
    def setUp(self):
        worddic = "<MASK> <RARE> first second third fourth fifth sixth"
        worddic = dict(zip(worddic.split(), range(len(worddic.split()))))
        self.linout = q.WordLinout(10, worddic=worddic, cosnorm=True)

    def test_it(self):
        x = Variable(torch.randn(7, 10))
        y = self.linout(x)
        print(y)
        self.assertEqual(y.size(), (7, 8))
        self.assertTrue(np.all(y.data.numpy() < 1))
        self.assertTrue(np.all(y.data.numpy() > -1))
        x = x.data.numpy()
        w = self. linout.lin.weight.data.numpy()
        y = y.data.numpy()
        for i in range(7):
            for j in range(8):
                self.assertTrue(np.allclose(y[i, j], np.dot(x[i], w[j]) / (np.linalg.norm(x[i], 2) * np.linalg.norm(w[j], 2))))

        x = q.var(x).v
        ny, cosnorm = self.linout(x, _retcosnorm=True)
        ny = ny / x.norm(2, 1).unsqueeze(1)
        ny = ny / cosnorm.pow(1./2)
        self.assertTrue(np.allclose(ny.data.numpy(), y))


class TestPretrainedWordLinout(TestCase):
    def setUp(self):
        q.PretrainedWordLinout.defaultpath = "../data/glove/miniglove.%dd"
        self.glove = q.PretrainedWordLinout(50)
        print(self.glove.defaultpath)

    def test_loaded(self):
        thevector = self.glove % "the"
        truevector = np.asarray([
         4.18000013e-01,   2.49679998e-01,  -4.12420005e-01,
         1.21699996e-01,   3.45270008e-01,  -4.44569997e-02,
        -4.96879995e-01,  -1.78619996e-01,  -6.60229998e-04,
        -6.56599998e-01,   2.78430015e-01,  -1.47670001e-01,
        -5.56770027e-01,   1.46579996e-01,  -9.50950012e-03,
         1.16579998e-02,   1.02040000e-01,  -1.27920002e-01,
        -8.44299972e-01,  -1.21809997e-01,  -1.68009996e-02,
        -3.32789987e-01,  -1.55200005e-01,  -2.31309995e-01,
        -1.91809997e-01,  -1.88230002e+00,  -7.67459989e-01,
         9.90509987e-02,  -4.21249986e-01,  -1.95260003e-01,
         4.00710011e+00,  -1.85939997e-01,  -5.22870004e-01,
        -3.16810012e-01,   5.92130003e-04,   7.44489999e-03,
         1.77780002e-01,  -1.58969998e-01,   1.20409997e-02,
        -5.42230010e-02,  -2.98709989e-01,  -1.57490000e-01,
        -3.47579986e-01,  -4.56370004e-02,  -4.42510009e-01,
         1.87849998e-01,   2.78489990e-03,  -1.84110001e-01,
        -1.15139998e-01,  -7.85809994e-01])
        self.assertEqual(self.glove * "the", 2)
        self.assertTrue(np.allclose(thevector, truevector))
        self.assertEqual(self.glove.lin.weight.size(), (4002, 50))


class TestAdaptedWordLinout(TestCase):
    def setUp(self):
        wdic = {"<MASK>": 0, "<RARE>": 1, "the": 10, "a": 5, "his": 50, "abracadabrqmsd--qsdfmqgf-": 6}
        wdic2 = {"<MASK>": 0, "<RARE>": 1, "the": 2, "a": 3, "his": 4, "abracadabrqmsd--qsdfmqgf-": 5, "qsdfqsdf": 7}
        self.adapted = q.WordLinout(10, worddic=wdic, bias=False)
        self.vanilla = q.WordLinout(10, worddic=wdic, weight=self.adapted.lin.weight.data.numpy(), bias=False)
        self.adapted = self.adapted.adapt(wdic2)

    def test_map(self):
        self.assertEqual(self.adapted * "a", 3)
        self.assertEqual(self.adapted * "the", 2)
        self.assertEqual(self.adapted * "his", 4)
        self.assertEqual(self.adapted * "her", 1)
        self.assertEqual(self.vanilla * "a", 5)
        self.assertEqual(self.vanilla * "the", 10)
        self.assertEqual(self.vanilla * "her", 1)
        self.assertEqual(self.adapted * "qsdfqlmkdsjfmqlsdkjgmqlsjdf", 1)
        print(self.vanilla * "the", self.adapted * "the")
        print(self.vanilla % "the", self.adapted % "the")
        self.assertTrue(np.allclose(self.vanilla % "the", self.adapted % "the"))
        self.assertTrue(np.allclose(self.vanilla % "his", self.adapted % "his"))

    def test_adapted_block(self):
        pred = self.adapted(Variable(torch.FloatTensor(np.stack([self.adapted % x for x in "the a his".split()], axis=0))))
        l = pred.sum()
        l.backward()
        grad = self.adapted.inner.lin.weight.grad
        self.assertTrue(grad.norm().data.numpy()[0] > 0)

    def test_adapted_prediction_shape(self):
        xval = np.stack([self.adapted % "the", self.adapted % "a"], axis=0)
        x = Variable(torch.from_numpy(xval))
        pred = self.adapted(x)
        self.assertEqual(pred.size(), (2, 8))

    def test_cosined(self):
        EPS = 1e-6
        self.adapted.inner.cosnorm = True
        xval = np.stack([self.adapted % "the", self.adapted % "a"], axis=0)
        x = q.var(xval).v
        pred = self.adapted(x)
        self.assertEqual(pred.size(), (2, 8))
        prednp = pred.data.numpy()
        print(prednp)
        print(pred)
        self.assertTrue(np.all(pred.data.numpy() <= 1.+EPS))
        self.assertTrue(np.all(pred.data.numpy() >= -1-EPS))

        ny, cosnorm = self.adapted(x, _retcosnorm=True)
        ny = ny / x.norm(2, 1).unsqueeze(1)
        ny = ny / cosnorm.pow(1./2)

        self.assertTrue(np.allclose(ny.data.numpy(), prednp))


class TestOverriddenWordLinout(TestCase):
    def setUp(self):
        wdic = {"<MASK>": 0, "<RARE>": 1, "the": 10, "a": 5, "his": 50, "monkey": 6}
        wdic2 = {"<MASK>": 0, "<RARE>": 1, "the": 2, "a": 3, "his": 4, "abracadabrqmsd--qsdfmqgf-": 5, "qsdfqsdf": 7}
        self.base = q.WordLinout(10, worddic=wdic, bias=False)
        self.over = q.WordLinout(10, worddic=wdic2, bias=False)
        self.overridden = self.base.override(self.over)

    def test_shapes(self):
        x = Variable(torch.FloatTensor(np.stack([self.base % x for x in "the a his".split()], axis=0)))
        pred = self.overridden(x)
        self.assertEqual(pred.size(), (3, 51))
        basepred = self.base(x)
        overpred = self.over(x)

        l = pred.sum()
        l.backward()
        self.assertTrue(self.base.lin.weight.grad.norm().data[0] > 0)
        self.assertTrue(self.over.lin.weight.grad.norm().data[0] > 1)

        basepred = basepred.data.numpy()
        overpred = overpred.data.numpy()
        pred = pred.data.numpy()
        self.assertTrue(np.allclose(pred[:, 10], overpred[:, 2]))
        self.assertTrue(np.allclose(pred[:, 5], overpred[:, 3]))
        self.assertTrue(np.allclose(pred[:, 6], basepred[:, 6]))

    def test_cosined(self):
        EPS = 1e-12
        self.base.cosnorm = True
        self.over.cosnorm = True
        x = q.var(np.stack([self.base % x for x in "the a his".split()], axis=0)).v
        pred = self.overridden(x)
        self.assertEqual(pred.size(), (3, 51))
        prednp = pred.data.numpy()
        print(prednp)
        print(pred)
        self.assertTrue(np.all(pred.data.numpy() <= 1. + EPS))
        self.assertTrue(np.all(pred.data.numpy() >= -1 - EPS))

        ny, cosnorm = self.overridden(x, _retcosnorm=True)
        ny = ny / x.norm(2, 1).unsqueeze(1)
        ny = ny / cosnorm.pow(1. / 2)

        self.assertTrue(np.allclose(ny.data.numpy(), prednp))


class TestComputedWordLinout(TestCase):
    def setUp(self):
        data = np.random.random((7, 10)).astype("float32")
        computer = nn.Linear(10, 15)
        worddic = "<MASK> <RARE> first second third fourth fifth"
        worddic = dict(zip(worddic.split(), range(len(worddic.split()))))
        self.linout = q.ComputedWordLinout(data=data, computer=computer, worddic=worddic, bias=False)

    def test_basic(self):
        x = Variable(torch.randn(3, 15)).float()
        out = self.linout(x)
        self.assertEqual(out.size(), (3, 7))
        data = self.linout.data
        computer = self.linout.computer
        cout = torch.matmul(x, computer(data).t())
        self.assertTrue(np.allclose(cout.data.numpy(), out.data.numpy()))

    def test_cosiner(self):
        EPS = 1e-12
        self.linout.cosnorm = True
        x = Variable(torch.randn(3, 15)).float()
        out = self.linout(x)
        self.assertEqual(out.size(), (3, 7))
        self.assertTrue(np.all(out.data.numpy() <= 1. + EPS))
        self.assertTrue(np.all(out.data.numpy() >= -1 - EPS))
        data = self.linout.data
        computer = self.linout.computer
        cout = torch.matmul(x, computer(data).t())
        cout = cout / torch.norm(computer(data), 2, 1).unsqueeze(0)
        cout = cout / torch.norm(x, 2, 1).unsqueeze(1)
        self.assertTrue(np.allclose(cout.data.numpy(), out.data.numpy()))
        self.linout.cosnorm = False
        ny, cosnorm = self.linout(x, _retcosnorm=True)
        ny = ny / x.norm(2, 1).unsqueeze(1)
        ny = ny / cosnorm.pow(1. / 2)

        self.assertTrue(np.allclose(ny.data.numpy(), out.data.numpy()))

    def test_masked(self):
        x = Variable(torch.randn(3, 15)).float()
        msk_nonzero_batches = [0,0,0,1,1,2]
        msk_nonzero_values = [0,2,3,2,6,5]
        msk = np.zeros((3, 7)).astype("int32")
        msk[msk_nonzero_batches, msk_nonzero_values] = 1
        print(msk)
        msk = Variable(torch.from_numpy(msk))
        out = self.linout(x, mask=msk)
        self.assertEqual(out.size(), (3, 7))
        data = self.linout.data
        computer = self.linout.computer
        cout = torch.matmul(x, computer(data).t())
        cout = cout * msk.float()
        self.assertTrue(np.allclose(cout.data.numpy(), out.data.numpy()))

    def test_masked_with_rnn_computer(self):
        data = np.random.random((7, 5, 10)).astype("float32")
        computer = q.RecurrentStack(
            q.GRULayer(10, 15)
        ).return_final()
        worddic = "<MASK> <RARE> first second third fourth fifth"
        worddic = dict(zip(worddic.split(), range(len(worddic.split()))))
        linout = q.ComputedWordLinout(data=data, computer=computer, worddic=worddic)

        x = Variable(torch.randn(3, 15)).float()
        msk_nonzero_batches = [0, 0, 0, 1, 1, 2]
        msk_nonzero_values = [0, 2, 3, 2, 6, 5]
        msk = np.zeros((3, 7)).astype("int32")
        msk[msk_nonzero_batches, msk_nonzero_values] = 1
        print(msk)
        msk = Variable(torch.from_numpy(msk))
        out = linout(x, mask=msk)
        self.assertEqual(out.size(), (3, 7))
        data = linout.data
        computer = linout.computer
        cout = torch.matmul(x, computer(data).t())
        cout = cout * msk.float()
        self.assertTrue(np.allclose(cout.data.numpy(), out.data.numpy()))

    def test_all_masked(self):
        x = Variable(torch.randn(3, 15)).float()
        msk = np.zeros((3, 7)).astype("int32")
        print(msk)
        msk = Variable(torch.from_numpy(msk))
        out = self.linout(x, mask=msk)
        self.assertEqual(out.size(), (3, 7))
        data = self.linout.data
        computer = self.linout.computer
        cout = torch.matmul(x, computer(data).t())
        cout = cout * msk.float()
        self.assertTrue(np.allclose(cout.data.numpy(), out.data.numpy()))

    def test_masked_3D_data(self):
        self.linout.data = q.val(np.random.random((7, 10, 3)).astype(dtype="float32")).v
        self.linout.computer = q.GRULayer(3, 15).return_final("only")

        x = Variable(torch.randn(3, 15)).float()
        msk_nonzero_batches = [0, 0, 0, 1, 1, 2]
        msk_nonzero_values = [0, 2, 3, 2, 6, 5]
        msk = np.zeros((3, 7)).astype("int32")
        msk[msk_nonzero_batches, msk_nonzero_values] = 1
        print(msk)
        msk = Variable(torch.from_numpy(msk))
        out = self.linout(x, mask=msk)
        self.assertEqual(out.size(), (3, 7))
        data = self.linout.data
        computer = self.linout.computer
        cout = torch.matmul(x, computer(data).t())
        cout = cout * msk.float()
        self.assertTrue(np.allclose(cout.data.numpy(), out.data.numpy()))

    def test_basic_grad(self):
        x = Variable(torch.randn(3, 15)).float()
        y = Variable(torch.randn(3, 15)).float()
        out = self.linout(x)
        loss = out.sum()
        loss.backward()

        agrads = []
        for p in self.linout.parameters():
            if p.requires_grad:
                agrads.append(p.grad.data.numpy() + 0)

        out = self.linout(y)
        loss = out.sum()
        loss.backward()

        bgrads = []
        for p in self.linout.parameters():
            if p.requires_grad:
                bgrads.append(p.grad.data.numpy() + 0)

        pass


class TestMergedWordLinout(TestCase):
    def setUp(self):
        wd = dict(zip(map(lambda x: chr(x), range(100)), range(100)))
        self.base = q.WordLinout(50, worddic=wd, bias=False)
        self.merg = q.WordLinout(50, worddic=wd, bias=False)
        self.linout = self.base.merge(self.merg)

    def test_cosiner(self):
        self.linout.cosnorm = True
        x = q.var(np.random.random((5, 50)).astype("float32")).v
        pred = self.linout(x)
        self.assertTrue(np.all(pred.data.numpy() <= 1.))
        self.assertTrue(np.all(pred.data.numpy() >= -1.))

        ny, cosnorm = self.linout(x, _retcosnorm=True)
        ny = ny / x.norm(2, 1).unsqueeze(1)
        ny = ny / cosnorm.pow(1. / 2)

        self.assertTrue(np.allclose(ny.data.numpy(), pred.data.numpy()))






