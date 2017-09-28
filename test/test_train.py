from unittest import TestCase
import qelos as q
import numpy as np
import torch
from torch.utils.data import DataLoader


class TestTensorDataset(TestCase):
    def test_numpy_construction(self):
        x = np.random.random((100, 5)).astype(dtype="float32")
        dataset = q.TensorDataset(x)
        x1 = x[1]
        xd1 = dataset[1][0]
        self.assertTrue(np.allclose(x1, xd1.numpy()))
        print(type(xd1))
        self.assertTrue(isinstance(xd1, torch.FloatTensor))

    def test_iter_single_tensor(self):
        x = np.arange(0, 100)
        dataset = q.TensorDataset(x)
        dl = DataLoader(dataset, shuffle=True, batch_size=10)
        epoch1batches = []
        epoch2batches = []
        for batch in dl:
            epoch1batches.append(batch[0].numpy())
        for batch in dl:
            epoch2batches.append(batch[0].numpy())
        for batcha, batchb in zip(epoch1batches, epoch2batches):
            self.assertFalse(np.allclose(batcha, batchb))

        epoch1 = np.concatenate(epoch1batches)
        epoch2 = np.concatenate(epoch2batches)

        print(epoch1)
        print(epoch2)
        self.assertEqual(set(epoch1), set(epoch2))

    def test_iter_single_tensor_error(self):
        x = np.arange(0, 100)
        dataset = q.TensorDataset(x)
        dl = DataLoader(dataset, shuffle=True, batch_size=10)
        epoch1batches = []
        epoch2batches = []
        batches = []
        dl_iter = iter(dl)
        def fn():
            for i in range(200):       # 1000 batches, 10 * data
                batch = next(dl_iter)[0].numpy()
                batches.append(batch)
        self.assertRaises(StopIteration, fn)


class TestEval(TestCase):
    def test_it(self):
        data = q.var(np.random.random((10,5)).astype("float32")).v.data
        m = torch.nn.Linear(5, 4)
        dl = q.dataload(data, batch_size=3)
        out = q.eval(m).on(dl).run()

        self.assertEqual(out.size(), (10, 4))
        refout = m(q.var(data).v)
        self.assertTrue(np.allclose(out.data.numpy(), refout.data.numpy()))
