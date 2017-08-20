from __future__ import print_function
from unittest import TestCase
import qelos as q
from torch.autograd import Variable
import torch
import numpy as np


class TestWordEmb(TestCase):
    def test_creation_simple(self):
        dic = dict(zip(map(chr, range(97, 122)), range(122-97)))
        m = q.WordEmb(10, worddic=dic)
        embedding = m(Variable(torch.LongTensor([0,1,2])))
        self.assertEqual(embedding.size(), (3, 10))
        trueemb = m.get_weight()[0]
        self.assertTrue(np.allclose(trueemb, embedding[0].data.numpy()))

    def test_creation_masked(self):
        dic = dict(zip(map(chr, range(97, 122)), range(1, 122-97+1)))
        dic[q.WordEmb.masktoken] = 0
        m = q.WordEmb(10, worddic=dic)
        embedding, mask = m(Variable(torch.LongTensor([0, 1, 2])))
        self.assertEqual(embedding.size(), (3, 10))
        trueemb = m.get_weight()[1]
        self.assertTrue(np.allclose(trueemb, embedding[1].data.numpy()))
        self.assertTrue(np.allclose(embedding[0].data.numpy(), np.zeros((10,))))
        print(mask)
        self.assertTrue(np.allclose(mask.data.numpy(), [0,1,1]))


class TestAdaptedWordEmb(TestCase):
    def setUp(self):
        wdic = {"<MASK>": 0, "<RARE>": 1, "the": 10, "a": 5, "his": 50, "abracadabrqmsd--qsdfmqgf-": 6}
        wdic2 = {"<MASK>": 0, "<RARE>": 1, "the": 2, "a": 3, "his": 4, "abracadabrqmsd--qsdfmqgf-": 5, "qsdfqsdf": 7}
        self.adapted = q.WordEmb(50, worddic=wdic)
        self.vanilla = q.WordEmb(50, worddic=wdic, value=self.adapted.get_weight())
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
        self.assertTrue(np.allclose(oovpred.data.numpy(), np.zeros_like(oovpred.data.numpy())))
