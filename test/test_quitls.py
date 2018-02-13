from unittest import TestCase
import qelos as q
import numpy as np
import torch


class TestIntercat(TestCase):
    def test_3D(self):
        x = torch.zeros(5, 4, 10)
        y = torch.ones(5, 4, 10)
        z = q.intercat([x, y], -1)
        print(z)
        exp_z = np.zeros((5, 4, 20))
        exp_z[:, :, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]] = 1
        print(exp_z)
        self.assertTrue(np.allclose(exp_z, z.cpu().numpy()))

    def test_3D_ax1(self):
        x = torch.zeros(5, 4, 10)
        y = torch.ones(5, 4, 10)
        z = q.intercat([x, y], 1)
        print(z)
        exp_z = np.zeros((5, 8, 10))
        exp_z[:, [1, 3, 5, 7], :] = 1
        print(exp_z)
        self.assertTrue(np.allclose(exp_z, z.cpu().numpy()))
