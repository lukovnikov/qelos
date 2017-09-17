from unittest import TestCase
import qelos as q
import numpy as np


class TestSeqConv(TestCase):
    def test_seq_conv_shape(self):
        x = q.var(np.random.random((3, 5, 4)).astype("float32")).v
        m = q.SeqConv(4, 6, window=3)
        y = m(x)
        self.assertEqual(y.size(), (3, 5, 6))

    def test_seq_conv_with_mask(self):
        x = q.var(np.random.random((3, 5, 4)).astype("float32")).v
        mask = q.var(np.asarray([[1,1,1,0,0],[1,0,0,0,0],[1,1,1,1,1],])).v == 1
        m = q.SeqConv(4, 6, window=3)
        y = m(x, mask=mask)
        self.assertEqual(y.size(), (3, 5, 6))
        mask = mask.unsqueeze(2).data.numpy()
        y = y.data.numpy()
        self.assertTrue(np.linalg.norm(y * (1 - mask)) == 0)


