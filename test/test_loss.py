from __future__ import print_function
from unittest import TestCase
from torch.autograd import Variable
import torch
import qelos as q
import numpy as np


class TestSeqNLLLoss(TestCase):
    def test_same_as_numpy(self):
        self.EPS = 1e-6
        self.batsize = 100
        self.seqlen = 120
        self.time_avg = True
        self.size_avg = True
        self.ignore_idx = 0
        self.weight = None
        self.vocsize = 5
        self.dorun()
        self.time_avg = False
        self.dorun()
        self.size_avg = False
        self.dorun()
        self.ignore_idx = -100
        self.dorun()
        self.weight = np.asarray([0.1, 1, 1, 0.5, 3])
        self.dorun()

    def dorun(self):
        EPS = self.EPS
        batsize = self.batsize
        seqlen = self.seqlen
        time_avg = self.time_avg
        size_avg = self.size_avg
        ignore_idx = self.ignore_idx
        weight = self.weight
        vocsize = self.vocsize
        x = Variable(torch.FloatTensor(-np.random.random((batsize, seqlen, vocsize))), requires_grad=False)
        y = np.random.randint(0, vocsize, (batsize, seqlen))
        weight = Variable(torch.FloatTensor(weight)) if weight is not None else None
        if ignore_idx >= 0:
            y[0, :] = ignore_idx
        y = Variable(torch.LongTensor(y), requires_grad=False)
        loss = q.SeqNLLLoss(size_average=size_avg, time_average=time_avg, ignore_index=ignore_idx, weight=weight)\
            (x, y)
        x = x.data.numpy()
        y = y.data.numpy()
        weight = weight.data.numpy() if weight is not None else None
        mask = y != ignore_idx
        batch_acc = 0
        batch_total = 0
        for i in range(x.shape[0]):
            time_acc = 0
            time_total = 0
            for j in range(x.shape[1]):
                time_acc += -x[i, j, y[i, j]] * mask[i, j] * (weight[y[i, j]] if weight is not None else 1)
                time_total += 1 * mask[i, j]
            if time_avg:
                time_acc /= time_total + EPS
            batch_acc += time_acc
            batch_total += 1
        nploss = batch_acc
        if size_avg:
            nploss /= batch_total
        print(loss)
        print(nploss)
        self.assertTrue(np.isclose(loss.data.numpy()[0], nploss))



class TestSeqAccuracy(TestCase):
    def test_same_as_numpy(self):
        self.EPS = 1e-6
        self.batsize = 100
        self.seqlen = 3
        self.size_avg = True
        self.ignore_idx = 0
        self.vocsize = 2
        self.dorun()
        self.dorun()
        self.size_avg = False
        self.dorun()
        self.ignore_idx = -100
        self.dorun()
        self.dorun()

    def dorun(self):
        EPS = self.EPS
        batsize = self.batsize
        seqlen = self.seqlen
        size_avg = self.size_avg
        ignore_idx = self.ignore_idx
        vocsize = self.vocsize
        x = Variable(torch.FloatTensor(-np.random.random((batsize, seqlen, vocsize))), requires_grad=False)
        y = np.random.randint(0, vocsize, (batsize, seqlen))

        if ignore_idx >= 0:
            y[0, :] = ignore_idx
        y = Variable(torch.LongTensor(y), requires_grad=False)
        loss = q.SeqAccuracy(size_average=size_avg, ignore_index=ignore_idx)\
            (x, y)

        print(loss)

        x = x.data.numpy()
        x = np.argmax(x, axis=2)
        y = y.data.numpy()
        mask = y != ignore_idx
        x = x * mask
        y = y * mask

        batch_acc = 0
        batch_total = 0
        for i in range(x.shape[0]):
            time_acc = 0
            if np.all(x[i] == y[i]):
                time_acc = 1
            batch_acc += time_acc
            batch_total += 1
        nploss = batch_acc * 1.
        if size_avg:
            nploss /= batch_total
        print(loss)
        print(nploss)
        self.assertTrue(np.isclose(loss.data.numpy()[0], nploss))



class TestSeqElemAccuracy(TestCase):
    def test_same_as_numpy(self):
        self.EPS = 1e-6
        self.batsize = 100
        self.seqlen = 3
        self.size_avg = True
        self.ignore_idx = 0
        self.vocsize = 3
        self.dorun()
        self.dorun()
        self.size_avg = False
        self.dorun()
        self.ignore_idx = -100
        self.dorun()
        self.dorun()

    def dorun(self):
        EPS = self.EPS
        batsize = self.batsize
        seqlen = self.seqlen
        size_avg = self.size_avg
        ignore_idx = self.ignore_idx
        vocsize = self.vocsize
        x = Variable(torch.FloatTensor(-np.random.random((batsize, seqlen, vocsize))), requires_grad=False)
        y = np.random.randint(0, vocsize, (batsize, seqlen))

        if ignore_idx >= 0:
            y[0, :] = ignore_idx
        y = Variable(torch.LongTensor(y), requires_grad=False)
        loss = q.SeqElemAccuracy(size_average=size_avg, ignore_index=ignore_idx)\
            (x, y)

        print(loss)

        x = x.data.numpy()
        x = np.argmax(x, axis=2)
        y = y.data.numpy()
        mask = y != ignore_idx
        x = x * mask
        y = y * mask

        same = x == y
        same = same * mask
        total = np.sum(mask)
        acc = np.sum(same)
        nploss = acc * 1.
        if size_avg:
            nploss /= total
        print(loss)
        print(nploss)
        loss = loss[0].data[0]
        self.assertTrue(np.isclose(loss, nploss))
