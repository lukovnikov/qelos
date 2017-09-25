import qelos as q
import torch
from torch import nn
import numpy as np


class DCB(object):
    def __init__(self, vocsize, dim):
        super(DCB, self).__init__()
        self.weight = nn.Parameter(torch.randn(vocsize, dim))
        self.reset_parameters()
        self.embedder = nn.Embedding(vocsize, dim)
        self.embedder.weight = self.weight
        self.linout = nn.Linear(dim, vocsize, bias=False)
        self.linout.weight = self.weight

    def reset_parameters(self):
        nn.init.xavier_normal(self.weight.data)

    def get_embedder(self):
        return self.embedder

    def get_linout(self):
        return self.linout


class Logger(object):
    def __init__(self, name="", ):
        self.tt = q.ticktock(name)

    def log(self, _iter=None, niter=None, errD=None, errG=None, scoreD_real=None, scoreD_fake=None, lip_loss=None, **kw):
        if (_iter+1) % 2 == 0:
            self.tt.live("[{}/{}] Loss_D: {:.4f} Loss_G: {:.4f} Score Real: {:.4f} Score Fake: {:.4f} Loss Lip: {:.4f}"
                         .format(_iter, niter, errD, errG, scoreD_real, scoreD_fake, lip_loss))



def get_data_gen(vocsize, batsize, seqlen):
    def data_gen():
        while True:
            yield np.random.randint(0, vocsize, (batsize, seqlen)).astype("int64")
    return data_gen


def make_nets_mine():
    dcb = DCB(vocsize, embdim)

    g_rnu = q.GRUCell(embdim+noisedim, gendim)

    # generator
    netG4D = q.ContextDecoderCell(
        dcb.get_embedder(),
        g_rnu,
        dcb.get_linout(),
        nn.LogSoftmax(),
        q.Lambda(lambda x: torch.max(x, 1)[1])
    )
    netG4D.teacher_unforce(seqlen, block=lambda x: x[0], startsymbols=1)
    netG4D = netG4D.to_decoder()

    netG4G = q.ContextDecoderCell(
        dcb.get_embedder(),
        g_rnu,
    )
    netG4G.teacher_unforce(seqlen,
                           q.Stack(
                               dcb.get_linout(),
                               nn.LogSoftmax(),
                               q.Lambda(lambda x: torch.max(x, 1)[1])
                           ),
                           startsymbols=1)
    netG4G = netG4G.to_decoder()

    # discriminator
    d_rnn = q.GRULayer(embdim, discdim)
    d_scorer = nn.Linear(discdim, 1)

    netD4D = q.RecurrentStack(
        dcb.get_embedder(),
        d_rnn,
    ).return_final()
    netD4D = q.Stack(netD4D, d_scorer)

    netD4G = q.RecurrentStack(
        d_rnn,
    ).return_final()
    netD4G = q.Stack(netD4G, d_scorer)

    # real processing
    netR = q.RecurrentStack(
        dcb.get_embedder(),
    )


def make_nets_normal(vocsize, embdim, gendim, discdim, startsym, seqlen, noisedim):
    netG = q.ContextDecoderCell(
        nn.Embedding(vocsize, embdim),
        q.GRUCell(embdim+noisedim, gendim),
        nn.Linear(embdim, vocsize),
        nn.Softmax(),
    )
    netG.teacher_unforce(seqlen,
                         q.Lambda(lambda x: torch.max(x[0], 1)[1]),
                         startsymbols=startsym)
    netG = netG.to_decoder()

    netD = q.RecurrentStack(
        nn.Linear(vocsize, discdim),
        q.GRULayer(discdim, discdim),
    ).return_final()

    netD = q.Stack(
        netD,
        q.Forward(discdim, discdim, activation="relu"),
        q.Forward(discdim, discdim, activation="relu"),
        nn.Linear(discdim, 1),
        q.Lambda(lambda x: x.squeeze(1))
    )

    netR = q.IdxToOnehot(vocsize)

    return (netD, netD), (netG, netG), netR


def run(lr=0.00005,
        vocsize=100,
        embdim=50,
        noisedim=50,
        gendim=50,
        discdim=50,
        batsize=16,
        niter=100,
        seqlen=10,
        cuda=False,
        mode="normal"       # "normal"
        ):

    # get data and dict
    datagen = get_data_gen(vocsize, batsize, seqlen)()

    # build networks
    if mode == "normal":
        (netD4D, netD4G), (netG4D, netG4G), netR \
            = make_nets_normal(vocsize, embdim, gendim, discdim, 1, seqlen, noisedim)
    elif mode == "mine":
        raise NotImplemented()
        (netD4D, netD4G), (netG4D, netG4G), netR = make_nets_mine()
    else:
        raise q.SumTingWongException("unknown mode {}".format(mode))

    # train
    optimD = torch.optim.RMSprop(q.params_of(netD4D), lr=lr)
    optimG = torch.optim.RMSprop(q.params_of(netG4G), lr=lr)
    gantrainer = q.GANTrainer(mode="WGAN-GP", one_sided=True, noise_dim=noisedim,
                              penalty_weight=5,
                 optimizerD=optimD, optimizerG=optimG, logger=Logger("gan"))

    gantrainer.train((netD4D, netD4G), (netG4D, netG4G), niter=niter,
                     data_gen=datagen, cuda=cuda, netR=netR)


if __name__ == "__main__":
    q.argprun(run)