from __future__ import print_function
import torch
from torch import nn
import qelos as q
import numpy as np
from qelos.util import argprun, ticktock
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from IPython import embed
from matplotlib import pyplot as plt
import seaborn as sns


tt = ticktock("script")


class EncDec(nn.Module):
    def __init__(self, encoder, decoder, mode="fast"):
        super(EncDec, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mode = mode

    def forward(self, srcseq, tgtseq):
        enc = self.encoder(srcseq)
        encstates = self.encoder.get_states(srcseq.size(0))
        if self.mode == "fast":
            self.decoder.set_init_states(encstates[-1])
        else:
            self.decoder.set_init_states(encstates[-1], encstates[-1])
        dec = self.decoder(tgtseq, enc)
        return dec


def main(
        lr=0.1,
        epochs=100,
        batsize=32,
        embdim=50,
        encdim=90,
        mode="cell",     # "fast" or "cell"
        wreg=0.0001,
        cuda=False,
        gpu=1,
         ):
    if cuda:
        torch.cuda.set_device(gpu)
    usecuda = cuda
    vocsize = 50
    # create data tensor
    tt.tick("loading data")
    sequences = np.random.randint(0, vocsize, (batsize * 100, 16))
    # wrap in dataset
    dataset = q.TensorDataset(sequences[:batsize * 80], sequences[:batsize * 80])
    validdataset = q.TensorDataset(sequences[batsize * 80:], sequences[batsize * 80:])
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batsize,
                            shuffle=True)
    validdataloader = DataLoader(dataset=validdataset,
                            batch_size=batsize,
                            shuffle=False)
    tt.tock("data loaded")
    # model
    tt.tick("building model")
    embedder = nn.Embedding(vocsize, embdim)

    encoder = q.RecurrentStack(
        embedder,
        q.GRULayer(embdim, encdim),
        q.GRULayer(encdim, encdim),
    )
    if mode == "fast":
        decoder = q.AttentionDecoder(attention=q.Attention().forward_gen(encdim, encdim, encdim),
                                     embedder=embedder,
                                     core=q.RecurrentStack(
                                         q.GRULayer(embdim, encdim)
                                        ),
                                     smo=q.Stack(
                                         nn.Linear(encdim+encdim, vocsize),
                                         q.LogSoftmax()
                                        ),
                                     return_att=True
                                     )
    else:
        decoder = q.AttentionDecoderCell(attention=q.Attention().forward_gen(encdim, encdim+embdim, encdim),
                                         embedder=embedder,
                                         core=q.RecStack(
                                             q.GRUCell(embdim+encdim, encdim,
                                                       use_cudnn_cell=False,
                                                       rec_batch_norm=None,
                                                       activation="crelu")
                                         ),
                                         smo=q.Stack(
                                             nn.Linear(encdim+encdim, vocsize),
                                             q.LogSoftmax()
                                         ),
                                         att_after_update=False,
                                         ctx_to_decinp=True,
                                         decinp_to_att=True,
                                         return_att=True,
                                         ).to_decoder()

    m = EncDec(encoder, decoder, mode=mode)

    losses = q.lossarray(q.SeqNLLLoss())
    validlosses = q.lossarray(q.SeqNLLLoss())

    optimizer = torch.optim.Adadelta(m.parameters(), lr=lr, weight_decay=wreg)
    tt.tock("model built")

    q.train(m).cuda(usecuda).train_on(dataloader, losses)\
        .set_batch_transformer(lambda x, y: (x, y[:, :-1], y[:, 1:]))\
        .valid_on(validdataloader, validlosses)\
        .optimizer(optimizer).clip_grad_norm(2.)\
        .train(epochs)

    testdat = np.random.randint(0, vocsize, (batsize, 20))
    testdata = q.var(torch.from_numpy(testdat)).cuda(usecuda).v
    testdata_out = q.var(torch.from_numpy(testdat)).cuda(usecuda).v
    if mode == "cell" and False:
        inv_idx = torch.arange(testdata.size(1) - 1, -1, -1).long()
        testdata = testdata.index_select(1, inv_idx)
    probs, attw = m(testdata, testdata_out[:, :-1])
    def plot(x):
        sns.heatmap(x)
        plt.show()
    embed()


if __name__ == "__main__":
    argprun(main)