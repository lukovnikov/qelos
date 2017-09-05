import random, os, numpy as np, sklearn as sk, pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec
import IPython

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable

import qelos as q

# DAGA GENERATOR COPIED FROM .IPYNB
# Dataset generator largely form Improved Training of Wasserstein GAN code (see link above)
def inf_train_gen(DATASET='8gaussians', BATCH_SIZE=None):
    np.random.seed(1234)
    if DATASET == '25gaussians':
        dataset = []
        for i in range(100000//25):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2)*0.05
                    point[0] += 2*x
                    point[1] += 2*y
                    dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        np.random.shuffle(dataset)
        dataset /= 2.828 # stdev
        while True:
            for i in range(len(dataset)//BATCH_SIZE):
                yield torch.from_numpy(dataset[i*BATCH_SIZE:(i+1)*BATCH_SIZE])

    elif DATASET == 'swissroll':

        while True:
            data = sk.datasets.make_swiss_roll(
                n_samples=BATCH_SIZE,
                noise=0.25
            )[0]
            data = data.astype('float32')[:, [0, 2]]
            data /= 7.5 # stdev plus a little
            yield torch.from_numpy(data)

    elif DATASET == '8gaussians':

        scale = 2.
        centers = [
            (1,0),
            (-1,0),
            (0,1),
            (0,-1),
            (1./np.sqrt(2), 1./np.sqrt(2)),
            (1./np.sqrt(2), -1./np.sqrt(2)),
            (-1./np.sqrt(2), 1./np.sqrt(2)),
            (-1./np.sqrt(2), -1./np.sqrt(2))
        ]
        centers = [(scale*x,scale*y) for x,y in centers]
        while True:
            dataset = []
            for i in range(BATCH_SIZE):
                point = np.random.randn(2)*.02
                center = random.choice(centers)
                point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)
            dataset = np.array(dataset, dtype='float32')
            dataset /= 1.414 # stdev
            yield torch.from_numpy(dataset)


class ToyGAN_G(nn.Module):
    def __init__(self, dim_hidden=512, dim_out=2, noise_dim=2):
        super(ToyGAN_G, self).__init__()
        self.dim_hidden, self.dim_out, self.noise_dim = dim_hidden, dim_out, noise_dim
        self.net = nn.Sequential(
            nn.Linear(noise_dim, dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden, dim_out)
            )

    def forward(self, x):
        x = self.net(x)
        return x


class ToyGAN_D(nn.Module):
    def __init__(self, dim_hidden=512, dim_gen_out=2):
        super(ToyGAN_D, self).__init__()
        self.dim_hidden, self.dim_gen_out = dim_hidden, dim_gen_out
        self.net = nn.Sequential(
            nn.Linear(dim_gen_out, dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden, 1)
            )

    def forward(self, x): #?
        x = self.net(x)
        return x


class Logger(object):
    def __init__(self, next_=None):
        self.next_ = next_
        self.errD = []
        self.errG = []
        self.scoreD_real = []
        self.scoreD_fake = []
        self.lip_loss = []

    def log(self, errD=None, errG=None, scoreD_real=None, scoreD_fake=None, lip_loss=None, **kw):
        self.errD.append(errD)
        self.errG.append(errG)
        self.scoreD_real.append(scoreD_real)
        self.scoreD_fake.append(scoreD_fake)
        self.lip_loss.append(lip_loss)
        self.next_.log(errD=errD, errG=errG, scoreD_real=scoreD_real, scoreD_fake=scoreD_fake, lip_loss=lip_loss, **kw)


class Plotter(object):
    def __init__(self, name="", ):
        self.tt = q.ticktock(name)

    def log(self, _iter=None, niter=None, errD=None, errG=None, scoreD_real=None, scoreD_fake=None, lip_loss=None, **kw):
        if (_iter+1) % 2 == 0:
            self.tt.live("[{}/{}] Loss_D: {:.4f} Loss_G: {:.4f} Score Real: {:.4f} Score Fake: {:.4f} Loss Lip: {:.4f}"
                         .format(_iter, niter, errD, errG, scoreD_real, scoreD_fake, lip_loss))


def main(
        lr=0.00005,
        batsize=100,
        cuda=False,
        dataset="swissroll",
        mode="PAGAN"
        ):
    netD = ToyGAN_D()
    netG = ToyGAN_G()
    optimizerD = optim.RMSprop(netD.parameters(), lr=lr)
    optimizerG = optim.RMSprop(netG.parameters(), lr=lr)

    plotter = Plotter(mode)
    logger = Logger(plotter)

    trainer = q.GANTrainer(mode=mode,
                         optimizerD=optimizerD,
                         optimizerG=optimizerG,
                         penalty_weight=10,
                         one_sided=True,
                         logger=logger)

    datagen = inf_train_gen(dataset, batsize)

    trainer.train(netD, netG, niter=500, niterD=10,
                  batsizeG=batsize, data_gen=datagen, cuda=cuda)


if __name__ == "__main__":
    q.argprun(main)
