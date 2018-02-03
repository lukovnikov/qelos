from unittest import TestCase
from qelos.qutils import seq_pack, seq_unpack
import qelos as q
import torch
import numpy as np


class TestSeq_pack(TestCase):
    def test_it(self):
        seq = q.var(torch.randn(5, 10, 4)).v
        lens = np.random.randint(1, 10, (5,))
        mask = np.zeros((5, 10))
        for i in range(mask.shape[0]):
            mask[i, :lens[i]] = 1
        print(mask)
        mask = q.var(mask).v.byte()
        o, us = seq_pack(seq, mask)
        # print(seq[:, :, 0])
        # print(o[:, :, 0])
        # print(o)
        print(us)
        recons, remask = seq_unpack(o, us)
        print(recons)
        seq = seq.cpu().data.numpy()
        seq *= mask.cpu().data.numpy()[:, :, np.newaxis]
        recons = recons.cpu().data.numpy()
        self.assertTrue(np.allclose(seq[:, :recons.shape[1]], recons))
