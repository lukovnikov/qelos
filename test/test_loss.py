from __future__ import print_function
from unittest import TestCase
from torch.autograd import Variable
import torch
import qelos as q
import numpy as np


class TestSeqNLLLoss(TestCase):
    def test_same_as_numpy(self):
        EPS = 1e-6
        batsize = 100
        seqlen = 120
        time_avg = True
        size_avg = True
        ignore_idx = 0
        x = Variable(torch.FloatTensor(-np.random.random((batsize, seqlen, 5))), requires_grad=False)
        y = np.random.randint(0, 1, (batsize, seqlen))
        y[0, :] = ignore_idx
        y = Variable(torch.LongTensor(y), requires_grad=False)
        loss = q.SeqNLLLoss(size_average=size_avg, time_average=time_avg, ignore_index=ignore_idx)(x, y)
        x = x.data.numpy()
        y = y.data.numpy()
        mask = y != ignore_idx
        batch_acc = 0
        batch_total = 0
        for i in range(x.shape[0]):
            time_acc = 0
            time_total = 0
            for j in range(x.shape[1]):
                time_acc += -x[i, j, y[i, j]] * mask[i, j]
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

