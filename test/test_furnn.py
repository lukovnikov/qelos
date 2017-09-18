from unittest import TestCase
import qelos as q
import numpy as np


class TestMemGRUCell(TestCase):
    def test_shapes(self):
        x = q.var(np.random.random((2, 5, 3)).astype("float32")).v
        m = q.MemGRUCell(3, 4, memsize=3).to_layer()

        y = m(x)
        self.assertEqual(y.size(), (2, 5, 4))
