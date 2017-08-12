from __future__ import print_function
from unittest import TestCase
from qelos.seq import Decoder, SimpleDecoderCell
from qelos.rnn import RecStack, GRU
from qelos.basic import Forward, Softmax
import torch, numpy as np
from torch import nn
from torch.autograd import Variable


class TestDecoder(TestCase):
    def test_simple_decoder(self):
        batsize, seqlen, vocsize = 5, 4, 7
        embdim, encdim, outdim = 10, 16, 10
        recstack = RecStack(
            nn.Embedding(vocsize, embdim, padding_idx=0),
            GRU(embdim, encdim),
            Forward(encdim, vocsize),
            Softmax()
        )
        decoder_cell = SimpleDecoderCell(recstack)
        decoder = decoder_cell.to_decoder()
        data = np.random.randint(0, vocsize, (batsize, seqlen))
        data = Variable(torch.LongTensor(data))

        decoded = decoder(data)[0].data.numpy()
        self.assertEqual(decoded.shape, (batsize, seqlen, vocsize))
        self.assertTrue(np.allclose(np.sum(decoded, axis=-1), np.ones_like(np.sum(decoded, axis=-1))))
