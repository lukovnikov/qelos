from unittest import TestCase

import qelos as q
import numpy as np


class TestSplit(TestCase):
    def test_split(self):
        data = np.arange(0, 100)
        splits = q.split([data], splits=(80, 20), random=False)
        print(splits)
        self.assertTrue(np.allclose(splits[0][0], np.arange(0, 80)))
        self.assertTrue(np.allclose(splits[1][0], np.arange(80, 100)))

    def test_split_random(self):
        data = np.arange(0, 100)
        splits = q.split([data], splits=(80, 20), random=True)
        print(splits)
        self.assertEqual(splits[0][0].shape, (80,))
        self.assertEqual(splits[1][0].shape, (20,))
        for i in range(1, len(splits[0][0])):
            self.assertTrue(splits[0][0][i-1] < splits[0][0][i])
        for i in range(1, len(splits[1][0])):
            self.assertTrue(splits[1][0][i-1] < splits[1][0][i])
