import qelos as q
import torch
from torch import nn
import numpy as np
from collections import OrderedDict


def make_subword_embedder(worddic, subworddic, word2subwordmat, embdim, numchan):
    """

    :param worddic:             maps words to ids
    :param subworddic:          maps subword units to ids
    :param word2subwordmat:      numpy matrix mapping word ids (rows) to sequences of subword-ids (values in row)
    :param embdim:              dimension of subword embedding
    :param numchan:             number of channels
    :return:                    word embedder
    """
    class Computer(nn.Module):
        def __init__(self, dim=10, numchannels=1, **kw):
            super(Computer, self).__init__(**kw)
            self.embdim = dim
            self.numchan = numchannels
            self.subemb = q.WordEmb(self.embdim * self.numchan, worddic=subworddic)

        def forward(self, x):
            _embedding, _ = self.subemb(x)   # 3D
            _wordemb = torch.sum(_embedding, 1)
            return _wordemb

    _emb = q.ComputedWordEmb(data=word2subwordmat, computer=Computer(dim=embdim, numchannels=numchan), worddic=worddic)

    class SubWordEmbedder(nn.Module):
        def __init__(self, emb):
            super(SubWordEmbedder, self).__init__()
            self.emb = emb

        def forward(self, x):
            _embedding, _ = self.emb(x)
            _wordemb = _embedding.view(x.size(0), self.emb.computer.numchan, -1)
            return _wordemb

        @property
        def numchannels(self):
            return self.emb.computer.numchan

        @property
        def embdim(self):
            return self.emb.computer.embdim

    emb = SubWordEmbedder(_emb)

    return emb


class SeqWordEmbWin(nn.Module):
    def __init__(self, chanwordemb):
        super(SeqWordEmbWin, self).__init__()
        self.chanwordemb = chanwordemb

    def forward(self, x):       # (batsize, seqlen) - word ids
        batsize, seqlen = x.size()
        x = torch.cat([x, q.var(torch.zeros(batsize, self.chanwordemb.numchannels).long()).cuda(x).v], 1)
        batsize, seqlen = x.size()
        _x = x.view(-1)
        _xemb = self.chanwordemb(_x)
        _xemb = _xemb.view(batsize, seqlen, self.chanwordemb.numchannels, self.chanwordemb.embdim)
        maxi = self.chanwordemb.numchannels
        out = _xemb[:, :seqlen-maxi, 0, :]
        for i in range(1, maxi):
            i_to = seqlen - maxi + i
            toadd = _xemb[:, i:i_to, i, :]
            out = out + toadd
        return out


if __name__ == "__main__":
    wd = {"<MASK>": 0, "cat": 1, "dog": 2, "elephant": 3}
    td = "<MASK> #ca cat at# #do dog og# #el ele lep eph pha han ant nt#"
    td = OrderedDict(zip(td.split(), range(len(td.split()))))
    embdim = 10
    numchan = 3
    m = [[0]*8,
         [1, 2, 3] + [0]*5,
         [4, 5, 6] + [0]*5,
         [7, 8, 9, 10, 11, 12, 13, 14]]
    m = np.asarray(m, dtype="int64")
    print(m)
    print(td)
    emb = make_subword_embedder(wd, td, m, embdim, numchan)

    # test
    test_x = q.var(np.asarray([[0, 1, 2, 3]]).T).v
    test_y = emb(test_x)
    print(test_y.size())
    assert(test_y.size() == (4, numchan, embdim))
    l = test_y.sum()
    l.backward()
    subemb = emb.emb.computer.subemb
    assert (np.all(subemb.embedding.weight.grad.data.numpy()[0] == np.zeros((embdim * numchan,))))
    print(subemb.embedding.weight.grad.data.numpy()[1])
    assert (np.all(subemb.embedding.weight.grad.data.numpy()[1:]))



    # test grads of 0 symbol in subemb
    subemb = emb.emb.computer.subemb
    test_x = q.var(np.asarray([[0, 1, 2, 3]])).v
    test_y, _ = subemb(test_x)
    print(test_y)
    l = torch.sum(test_y)
    l.backward()
    # print(subemb.embedding.weight.grad)
    assert(np.all(subemb.embedding.weight.grad.data.numpy()[0] == np.zeros((embdim*numchan,))))


    # test sequence mapper
    m = SeqWordEmbWin(emb)
    test_x = q.var(np.asarray([[1, 1], [2, 2]])).v
    test_y = m(test_x)
    print(test_y.size())
    pass