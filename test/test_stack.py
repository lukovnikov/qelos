from unittest import TestCase
import qelos as q
import numpy as np
from torch import nn
import torch


class TestStack(TestCase):
    def test_bypass_stack(self):
        data = q.var(np.random.random((3,5)).astype(dtype="float32")).v
        stack = q.Stack(
            q.Forward(5, 5),
            q.argsave.spec(a=0),
            q.Forward(5, 5),
            q.Forward(5, 5),
            q.argmap.spec(0, ["a"]),
            q.Lambda(lambda x, y: torch.cat([x, y], 1)),
            q.Forward(10, 7)
            )
        out = stack(data)
        print(out)
        self.assertEqual(out.size(), (3, 7))

    def test_dynamic_bypass_stack(self):
        data = q.var(np.random.random((3,5)).astype(dtype="float32")).v
        stack = q.Stack()
        nlayers = 5
        for i in range(nlayers):
            stack.add(
                q.argsave.spec(a=0),
                q.Forward(5, 5),
                q.Forward(5, 5),
                q.argmap.spec(0, ["a"]),
                q.Lambda(lambda x, y: x+y)
            )
        out = stack(data)
        print(out)
        self.assertEqual(out.size(), (3, 5))

        out.sum().backward()

        forwards = []
        for layer in stack.layers:
            if isinstance(layer, q.Forward):
                self.assertTrue(layer.lin.weight.grad is not None)
                self.assertTrue(layer.lin.bias.grad is not None)
                print(layer.lin.weight.grad.norm(2))
                self.assertTrue(layer.lin.weight.grad.norm(2).data[0] > 0)
                self.assertTrue(layer.lin.bias.grad.norm(2).data[0] > 0)
                forwards.append(layer)

        self.assertEqual(len(forwards), nlayers * 2)
