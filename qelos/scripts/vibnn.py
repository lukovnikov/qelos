import qelos as q
import torch
from torch.nn import Linear
import torch.nn.functional as F
import numpy as np
import math


class ComputedParameter(torch.nn.Module):
    def forward(self, *x):
        return self.compute(*x)

    def compute(self, *x):
        """ computes weights """
        raise NotImplemented()

    def prepare(self, *input):
        raise NotImplemented()


class StochasticParameter(ComputedParameter):
    def __init__(self, *dim, **kw):
        super(StochasticParameter, self).__init__()
        self.v = None
        self.mu_sigma = q.getkw(kw, "mu_sigma", 0.05)
        self.logsigma_offset = q.getkw(kw, "logsigma_offset", 0.)
        self.logsigma_range = q.getkw(kw, "logsigma_range", 0.02)
        self.mu = torch.nn.Parameter(torch.Tensor(*dim).normal_(0, self.mu_sigma))
        self.logsigma = torch.nn.Parameter(torch.Tensor(*dim).uniform_(-self.logsigma_range/2+self.logsigma_offset,
                                                                       self.logsigma_range/2+self.logsigma_offset))
        self.eps_sigma = q.getkw(kw, "eps_sigma", 1.0)
        self.penalties = []

    def compute(self, *x):
        if len(x) > 0:
            self.prepare(*x)
        eps = q.var(torch.zeros(self.mu.size()).normal_(0, self.eps_sigma)).cuda(self.mu).v
        v = self.mu + torch.log(1 + torch.exp(self.logsigma)) * eps
        self.v = v
        for penalty in self.penalties:
            penalty(self)
        return self.v

    def prepare(self, *x):
        pass

    def reset(self):
        for penalty in self.penalties:
            penalty.reset()


def log_gauss(x, mu, sigma):
    return -0.5 * np.log(2 * np.pi) - np.log(np.abs(sigma)) - (x - mu) ** 2 / (2 * sigma ** 2)


def log_gauss_logsigma(x, mu, logsigma):
    return -0.5 * np.log(2 * np.pi) - logsigma - (x - mu) ** 2 / (2 * torch.exp(logsigma) ** 2)


class GaussianPriorKLPenalty(torch.nn.Module):
    """ Simple gaussian penalty from KL divergence between variational approximation and prior"""
    def __init__(self, priorsigma, weight=1.):
        super(GaussianPriorKLPenalty, self).__init__()
        self.priorsigma = priorsigma
        self.weight = weight
        self.acc = 0.

    def reset(self):
        self.acc = 0.

    def forward(self, x):
        log_pw = log_gauss(x.v, 0, self.priorsigma).sum()
        log_qw = log_gauss_logsigma(x.v, x.mu, x.logsigma).sum()
        ret = log_qw - log_pw
        self.acc += ret * self.weight


class SWLinear(torch.nn.Module):
    def __init__(self, indim, outdim, eps_sigma=1., priorsigma=1., **kw):
        super(SWLinear, self).__init__(**kw)
        self.indim, self.outdim = indim, outdim
        self.W = StochasticParameter(outdim, indim, eps_sigma=eps_sigma)
        self.b = StochasticParameter(outdim, eps_sigma=eps_sigma)
        self.W.penalties.append(GaussianPriorKLPenalty(priorsigma))
        self.b.penalties.append(GaussianPriorKLPenalty(priorsigma))

    def reset(self):
        self.W.reset()
        self.b.reset()

    def forward(self, x):
        self.W.prepare(x); self.b.prepare(x)
        out = F.linear(x, self.W(), self.b())
        return out


class DummyPenalty(torch.nn.Module):
    def __init__(self):
        super(DummyPenalty, self).__init__()
        self.acc = 0.

    def reset(self):
        self.acc = 0


class ReparamSWLinear(torch.nn.Module):
    def __init__(self, indim, outdim, eps_sigma=1., priorsigma=1., **kw):
        super(ReparamSWLinear, self).__init__(**kw)
        self.indim, self.outdim, self.eps_sigma = indim, outdim, eps_sigma
        self.W = StochasticParameter(outdim, indim, eps_sigma=eps_sigma)
        self.b = StochasticParameter(outdim, eps_sigma=eps_sigma)
        self.priorsigma = priorsigma
        self.penalties = [DummyPenalty()]

    def reset(self):
        for penalty in self.penalties:
            penalty.reset()

    def forward(self, x):
        out_mu = F.linear(x, self.W.mu, self.b.mu)
        out_sigma = F.linear(x, self.W.logsigma.exp(), self.b.logsigma.exp())
        # reparameterize
        eps = q.var(torch.zeros(out_mu.size()).normal_(0, self.eps_sigma)).cuda(out_mu).v
        y = out_mu + out_sigma * eps
        # kl div penalty    !! unnormalized !!
        W_kl_penalty = 0.5 * torch.sum(
            -1 + 2 * (math.log(self.priorsigma) - self.W.logsigma) + (self.W.mu ** 2 + self.W.logsigma.exp() ** 2) / (
            self.priorsigma ** 2))
        b_kl_penalty = 0.5 * torch.sum(
            -1 + 2 * (math.log(self.priorsigma) - self.b.logsigma) + (self.b.mu ** 2 + self.b.logsigma.exp() ** 2) / (
            self.priorsigma ** 2))
        self.penalties[0].acc += W_kl_penalty + b_kl_penalty
        return y


def collect_penalties(m):
    acc = 0
    for mmod in m.modules():
        if hasattr(mmod, "penalties"):
            acc += sum([penalty.acc for penalty in mmod.penalties])
    return acc


def test_random_linear_classifier(lr=0.1):
    # TODO: test with some targets that make sense, random doesn't work 
    x = np.random.random((500, 20)).astype("float32")
    y = np.random.randint(0, 5, (500,)).astype("int64")

    for i in range(len(x)):
        y[i] = np.argmax(x[i].reshape((5, 4)).sum(axis=1))

    m = ReparamSWLinear(20, 5)

    batsize = 10
    dataloader = q.dataload(x, y, batch_size=batsize)
    optimizer = torch.optim.RMSprop(q.params_of(m), lr=0.001)

    def howmuchvar():
        return torch.exp(m.W.logsigma).sum()

    i = 0
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(2000):
        for xb, yb in dataloader:
            m.reset()
            print(i)
            _xb, _yb = q.var(xb).v, q.var(yb).v
            o = m(_xb)
            # actual loss:
            l = loss(o, _yb)
            cost = l * 1.
            # penalties:
            penalties = collect_penalties(m)
            penalties = 1. * penalties / (1. * batsize * len(dataloader))
            cost = cost + penalties
            cost.backward()
            optimizer.step()
            print(l.data[0], howmuchvar().data[0], penalties.data[0])
            i += 1


def run(lr=0.1):
    test_random_linear_classifier()


if __name__ == "__main__":
    q.argprun(run)