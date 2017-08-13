from __future__ import print_function
from unittest import TestCase
from qelos.seq import Decoder, DecoderCell, ContextDecoderCell
from qelos.rnn import RecStack, GRU
from qelos.basic import Forward, Softmax
import torch, numpy as np
from torch import nn
from torch.autograd import Variable


class TestDecoder(TestCase):
    def test_simple_decoder_shape(self):
        batsize, seqlen, vocsize = 5, 4, 7
        embdim, encdim, outdim = 10, 16, 10
        # model def
        decoder_cell = DecoderCell(
            nn.Embedding(vocsize, embdim, padding_idx=0),
            GRU(embdim, encdim),
            Forward(encdim, vocsize),
            Softmax()
        )
        decoder = decoder_cell.to_decoder()
        # end model def
        data = np.random.randint(0, vocsize, (batsize, seqlen))
        data = Variable(torch.LongTensor(data))

        decoded = decoder(data)[0].data.numpy()
        self.assertEqual(decoded.shape, (batsize, seqlen, vocsize))     # shape check
        self.assertTrue(np.allclose(np.sum(decoded, axis=-1), np.ones_like(np.sum(decoded, axis=-1))))  # prob check

    def test_context_decoder_shape(self):
        batsize, seqlen, vocsize = 5, 4, 7
        embdim, encdim, outdim, ctxdim = 10, 16, 10, 8
        # model def
        decoder_cell = ContextDecoderCell(
            nn.Embedding(vocsize, embdim, padding_idx=0),
            GRU(embdim+ctxdim, encdim),
            Forward(encdim, vocsize),
            Softmax()
        )
        decoder = decoder_cell.to_decoder()
        # end model def
        data = np.random.randint(0, vocsize, (batsize, seqlen))
        data = Variable(torch.LongTensor(data))
        ctx = Variable(torch.FloatTensor(np.random.random((batsize, ctxdim))))

        decoded = decoder(data, ctx)[0].data.numpy()
        self.assertEqual(decoded.shape, (batsize, seqlen, vocsize))     # shape check
        self.assertTrue(np.allclose(np.sum(decoded, axis=-1), np.ones_like(np.sum(decoded, axis=-1))))  # prob check


