from unittest import TestCase
import qelos as q
import torch
import numpy as np
from qelos.rnn import FastestGRUEncoderLayer, FastestGRUEncoder


class TestFastLSTMEncoderLayer(TestCase):
    def test_it(self):
        batsize, seqlen, indim, dim = 3, 4, 5, 12
        vecs = q.var(torch.randn(batsize, seqlen, indim)).v
        mask = np.asarray([[1, 1, 0, 0],
                           [1, 0, 0, 0],
                           [1, 1, 1, 1]]).astype("int64")
        mask = q.var(mask).v

        l = q.FastLSTMEncoderLayer(indim, dim, bidir=True, dropout_in=0.)

        out = l(vecs, mask=mask)
        y_n = l.y_n
        y_test = y_n[0, 0, :].contiguous().view(-1).cpu().data.numpy()
        out_test = out[0, 1, :dim].cpu().data.numpy()
        assert(np.allclose(y_test, out_test))
        y_test = y_n[0, 1, :].contiguous().view(-1).cpu().data.numpy()
        out_test = out[0, 0, dim:].cpu().data.numpy()
        assert(np.allclose(y_test, out_test))

        y_test = y_n[1, 0, :].contiguous().view(-1).cpu().data.numpy()
        out_test = out[1, 0, :dim].cpu().data.numpy()
        assert(np.allclose(y_test, out_test))
        y_test = y_n[1, 1, :].contiguous().view(-1).cpu().data.numpy()
        out_test = out[1, 0, dim:].cpu().data.numpy()
        assert(np.allclose(y_test, out_test))

        y_test = y_n[2, 0, :].contiguous().view(-1).cpu().data.numpy()
        out_test = out[2, 3, :dim].cpu().data.numpy()
        assert(np.allclose(y_test, out_test))
        y_test = y_n[2, 1, :].contiguous().view(-1).cpu().data.numpy()
        out_test = out[2, 0, dim:].cpu().data.numpy()
        assert(np.allclose(y_test, out_test))

        " test grad "
        vecs.requires_grad = True
        out = l(vecs, mask)
        loss = out[0].sum()
        loss.backward()
        vecs.grad
        vecsgrad = vecs.grad.cpu().data.numpy()
        assert(np.all(vecsgrad[1:] == 0))
        assert(np.all(vecsgrad[0, :2] != 0.))
        assert(np.all(vecsgrad[0, 2:] == 0))


    def test_lstm_encoder(self):
        batsize, seqlen, indim, dims = 3, 4, 6, (8, 12, 24)
        vecs = q.var(torch.randn(batsize, seqlen, indim)).v
        mask = np.asarray([[1, 1, 0, 0],
                           [1, 0, 0, 0],
                           [1, 1, 1, 1]]).astype("int64")
        mask = q.var(mask).v

        l = q.FastLSTMEncoder(indim, *dims, bidir=True, dropout_in=0.5)

        out = l(vecs, mask)

        dim = 24
        y_n = l.y_n[-1]
        y_test = y_n[0, 0, :].contiguous().view(-1).cpu().data.numpy()
        out_test = out[0, 1, :dim].cpu().data.numpy()
        assert(np.allclose(y_test, out_test))
        y_test = y_n[0, 1, :].contiguous().view(-1).cpu().data.numpy()
        out_test = out[0, 0, dim:].cpu().data.numpy()
        assert(np.allclose(y_test, out_test))

        y_test = y_n[1, 0, :].contiguous().view(-1).cpu().data.numpy()
        out_test = out[1, 0, :dim].cpu().data.numpy()
        assert(np.allclose(y_test, out_test))
        y_test = y_n[1, 1, :].contiguous().view(-1).cpu().data.numpy()
        out_test = out[1, 0, dim:].cpu().data.numpy()
        assert(np.allclose(y_test, out_test))

        y_test = y_n[2, 0, :].contiguous().view(-1).cpu().data.numpy()
        out_test = out[2, 3, :dim].cpu().data.numpy()
        assert(np.allclose(y_test, out_test))
        y_test = y_n[2, 1, :].contiguous().view(-1).cpu().data.numpy()
        out_test = out[2, 0, dim:].cpu().data.numpy()
        assert(np.allclose(y_test, out_test))

        print("done")


class TestFastestModules(TestCase):
    def test_layer(self):
        batsize, seqlen, indim, dim = 3, 4, 5, 12
        vecs = q.var(torch.randn(batsize, seqlen, indim)).v
        mask = np.asarray([[1, 1, 0, 0],
                           [1, 0, 0, 0],
                           [1, 1, 1, 1]]).astype("int64")
        mask = q.var(mask).v

        l = FastestGRUEncoderLayer(indim, dim, bidir=True, dropout_in=0., dropout_rec=0.5, bias=False)

        out = l(vecs, mask)

        print(l.h_n)

        loss = out.sum()
        loss.backward()

    def test_stack(self):
        batsize, seqlen, indim, dims = 3, 4, 5, (8, 12, 24)
        vecs = q.var(torch.randn(batsize, seqlen, indim)).v
        mask = np.asarray([[1, 1, 0, 0],
                           [1, 0, 0, 0],
                           [1, 1, 1, 1]]).astype("int64")
        mask = q.var(mask).v

        l = FastestGRUEncoder(indim, *dims, bidir=True, dropout_in=0.3, dropout_rec=0.5, bias=False)

        out = l(vecs, mask)

        loss = out.sum()
        loss.backward()


