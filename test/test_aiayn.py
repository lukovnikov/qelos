import qelos as q
from unittest import TestCase
import numpy as np


class AYNEncoderTest(TestCase):
    def test_encoder_shape(self):
        wdic = "<MASK> a b c d e f g h i j k l m n o p".split()
        wdic = dict(zip(wdic, range(len(wdic))))
        emb = q.WordEmb(16, worddic=wdic)
        m = q.AYNEncoder(emb, n_max_seq=7, n_layers=3, n_head=2,
                         d_k=4, d_v=6, d_word_vec=16, d_model=16,
                         d_inner_hid=20, dropout=0)
        src_seq = q.var(np.random.randint(1, max(wdic.values()), (5, 7))).v
        src_seq_mask_starts = np.random.randint(1, 7, (5,), dtype="int64")
        src_seq_mask = np.ones_like(src_seq.data.numpy())
        for i in range(5):
            src_seq_mask[i, :src_seq_mask_starts[i]] = 0
        src_seq_mask = q.var(src_seq_mask).v
        src_seq.masked_fill_(src_seq_mask.byte(), 0)
        src_pos = q.var(np.arange(0, 7, dtype="int64")).v
        src_pos = src_pos.unsqueeze(0).repeat(5, 1)

        out = m(src_seq, src_pos)
        print(out)
        self.assertEqual(out.size(), (5, 7, 16))

        loss = out.sum()
        loss.backward()

    def test_decoder_shape(self):
        wdic = "MASK> a b c d e f g h i j k l m n o p".split()
        wdic = dict(zip(wdic, range(len(wdic))))
        emb = q.WordEmb(16, worddic=wdic)
        m = q.AYNDecoder(emb, n_max_seq=7, n_layers=3, n_head=2,
                         d_k=4, d_v=6, d_word_vec=16, d_model=16,
                         d_inner_hid=20, dropout=0)
        src_seq = q.var(np.random.randint(1, max(wdic.values()), (5, 7))).v
        src_seq_mask_starts = np.random.randint(1, 7, (5,), dtype="int64")
        src_seq_mask = np.ones_like(src_seq.data.numpy())
        for i in range(5):
            src_seq_mask[i, :src_seq_mask_starts[i]] = 0
        src_seq_mask = q.var(src_seq_mask).v
        src_seq.masked_fill_(src_seq_mask.byte(), 0)
        src_pos = q.var(np.arange(0, 7, dtype="int64")).v
        src_pos = src_pos.unsqueeze(0).repeat(5, 1)

        ctx = q.var(np.random.random((5, 8, 16)).astype("float32")).v
        ctxmask = q.var(np.random.randint(0, 2, (5, 8))).v.byte()

        out = m(src_seq, src_pos, ctx, ctxmask)
        print(out)
        self.assertEqual(out.size(), (5, 7, 16))

        loss = out.sum()
        loss.backward()
