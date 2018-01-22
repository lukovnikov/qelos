import qelos as q
import torch
import numpy as np


class GumbelSoftmax(torch.nn.Module):
    def __init__(self, dim=None, temp=1.):
        super(GumbelSoftmax, self).__init__()
        self.EPS = 1e-16
        self.sm = torch.nn.Softmax(dim)
        self.temp = temp

    def forward(self, x):
        u = q.var(x.data.new(x.size()).random_(0, 1)).v
        g_sample = -torch.log(-torch.log(u + self.EPS) + self.EPS)
        y = x + g_sample
        ret = self.sm(y / q.v(self.temp))
        return ret


def run(lr=0.1, epochs=2000):
    x = np.arange(0, 5, dtype="int64")
    y = np.arange(0, 5, dtype="int64")
    y2 = np.arange(0, 5, dtype="int64")+1
    y = np.stack([y, y2], 1)
    d = dict(zip([chr(i) for i in range(7)], range(7)))
    emb = q.WordEmb(10, worddic=d)
    out = q.WordLinout(20, worddic=d, bias=False)

    z_dist = torch.distributions.Normal(torch.zeros(10), torch.ones(10))

    print(z_dist)

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.emb = emb
            self.out = out
            self.z_dist = z_dist
            self.sm = torch.nn.Softmax(1)
            # self.sm = GumbelSoftmax(1)

        def forward(self, x):
            embs, _ = self.emb(x)
            z = q.var(z_dist.sample_n(x.size(0))).v
            predvec = torch.cat([embs, z], 1)
            # predvec = embs + z
            outvec = self.out(predvec)
            ret = self.sm(outvec)
            return ret

    class Loss(torch.nn.Module):
        def __init__(self):
            super(Loss, self).__init__()

        def forward(self, probs, gold):
            goldprobs = torch.gather(probs, 1, gold)
            # bestprobs, _ = torch.max(goldprobs, 1)
            # bestprobs = torch.sum(goldprobs, 1)
            interp = q.var((torch.rand(goldprobs.size(0)) > 0.5).long().unsqueeze(1)).v
            bestprobs = torch.gather(goldprobs, 1, interp)
            # sample = goldprobs.multinomial(1)
            # bestprobs = torch.gather(goldprobs, 1, sample)
            return bestprobs

    loss = Loss()

    model = Model()

    optim = torch.optim.Adadelta(q.params_of(model), lr=lr)
    xvar = q.var(x).v
    yvar = q.var(y).v

    for i in range(epochs):
        optim.zero_grad()
        pred_y = model(xvar)
        print(pred_y)
        l = loss(pred_y, yvar)
        l = torch.mean(-torch.log(l))
        l.backward()
        optim.step()


def run_cvae(lr=0.001, epochs=10000):
    x = np.arange(0, 5, dtype="int64")
    # x = np.ones((5,), dtype="int64")
    y = np.arange(0, 5, dtype="int64")
    y2 = np.arange(0, 5, dtype="int64") + 1
    y3 = np.arange(0, 5, dtype="int64") + 2
    y = np.stack([y, y2, y3], 1)
    d = dict(zip([chr(i) for i in range(7)], range(7)))
    xemb = q.WordEmb(5, worddic=d)
    yemb = q.WordEmb(5, worddic=d)  # first 10 for means, second 10 for variances
    ff_out = q.Forward(15, 5)
    out = q.WordLinout(5, worddic=d, bias=False)

    temp = q.hyperparam(1.)

    z_dist = torch.distributions.Normal(torch.zeros(10), torch.ones(10))

    class CVAE(torch.nn.Module):
        def __init__(self):
            super(CVAE, self).__init__()
            self.xemb = xemb
            self.yemb = yemb
            self.out = out
            self.z_dist = z_dist
            self.sm = torch.nn.Softmax(1)
            self.ff_fromx = q.Forward(5, 5)
            self.ff_fromy = q.Forward(5, 5)
            self.ff_mean = torch.nn.Linear(5, 10)
            self.ff_sigma = torch.nn.Linear(5, 10)     # predicts log of var
            self.ff_out = ff_out

        def forward(self, x, y):
            # conditional encoder (x is input)
            yembs, _ = self.yemb(y)
            xembs, _ = self.xemb(x)
            enco = self.ff_fromx(xembs) + self.ff_fromy(yembs)
            means, logvar = self.ff_mean(enco), self.ff_sigma(enco)
            std = torch.exp(logvar * 0.5)
            z = q.var(self.z_dist.sample_n(x.size(0))).v
            predvec = means + std * z
            predvec = torch.cat([predvec, xembs], 1)
            outvec = self.out(self.ff_out(predvec))
            ret = self.sm(outvec/q.v(temp))
            return ret, means, logvar

    class CVAE_Predict(torch.nn.Module):
        def __init__(self):
            super(CVAE_Predict, self).__init__()
            self.xemb = xemb
            self.out = out
            self.ff_out = ff_out
            self.z_dist = z_dist
            self.sm = torch.nn.Softmax(1)

        def forward(self, x):
            xembs, _ = self.xemb(x)
            z = q.var(self.z_dist.sample_n(x.size(0))).v
            predvec = torch.cat([z, xembs], 1)
            outvec = self.out(self.ff_out(predvec))
            ret = self.sm(outvec)
            return ret

    class CVAE_Loss(torch.nn.Module):
        def forward(self, probs, gold, means, logvar):
            goldprobs = torch.gather(probs, 1, gold)
            reconstr_loss = -torch.log(goldprobs)
            kld = -0.5 * torch.sum(1 + logvar - (means ** 2) - logvar.exp(), 1)
            # kldiv = 1./2 * (sigmas.sum(1) + (means ** 2).sum(1)
            #                 - sigmas.size(1) - torch.log(sigmas.prod(1)))
            return reconstr_loss.squeeze(1), kld

    loss = CVAE_Loss()
    model = CVAE()
    predmodel = CVAE_Predict()

    optim = torch.optim.Adam(q.params_of(model), lr=lr)
    xvar = q.var(x).v
    yvar = q.var(y).v

    epochs = 10000

    kl_weight = q.hyperparam(1.)

    for i in range(epochs):
        # kl_weight.v = float(np.tanh(i/1000.-7.)/2.+.5)
        kl_weight.v = float(-np.cos(i/500.)/2.+.5)
        optim.zero_grad()
        # sample a y
        goldvecs = q.var(torch.zeros(xvar.size(0), 7).scatter_(1, yvar.data, 1)).v
        samplegold = goldvecs.multinomial(1).squeeze(1)
        prob_y, means, logvar = model(xvar, samplegold)
        l_r, l_kl = loss(prob_y, samplegold.unsqueeze(1), means, logvar)
        print(i, l_r.data[0], l_kl.data[0], q.v(kl_weight))
        l = torch.mean(l_r + q.v(kl_weight) * l_kl)
        # print(l)
        l.backward()
        optim.step()
        # temp.v = (1. - 0.99 * i/epochs)
        # print(i, q.v(temp))

    kl_weight.v = 1.
    epochs = 1000
    for i in range(epochs):
        optim.zero_grad()
        # sample a y
        goldvecs = q.var(torch.zeros(xvar.size(0), 7).scatter_(1, yvar.data, 1)).v
        samplegold = goldvecs.multinomial(1).squeeze(1)
        prob_y, means, logvar = model(xvar, samplegold)
        l_r, l_kl = loss(prob_y, samplegold.unsqueeze(1), means, logvar)
        print(i, l_r.data[0], l_kl.data[0], q.v(kl_weight))
        l = torch.mean(l_r + q.v(kl_weight) * l_kl)
        # print(l)
        l.backward()
        optim.step()

    # model.xemb.embedding.weight.requires_grad = False
    # for p in model.ff_fromx.parameters():
    #     p.requires_grad = False
    #
    # optim = torch.optim.Adam(q.params_of(model), lr=lr)
    #
    # for i in range(3000):
    #     optim.zero_grad()
    #     # sample a y
    #     goldvecs = q.var(torch.zeros(xvar.size(0), 7).scatter_(1, yvar.data, 1)).v
    #     samplegold = goldvecs.multinomial(1).squeeze(1)
    #     prob_y, means, logvar = model(xvar, samplegold)
    #     l_r, l_kl = loss(prob_y, samplegold.unsqueeze(1), means, logvar)
    #     print(i, l_r.data[0], l_kl.data[0])
    #     l = torch.mean(l_r + l_kl)
    #     # print(l)
    #     l.backward()
    #     optim.step()
    temp.reset()

    samples = 20
    np.set_printoptions(precision=3, suppress=True)
    print("pred 0")
    for i in range(samples):
        xpred = xvar[0:1]
        ypreds = predmodel(xpred)
        print(ypreds.data.numpy())
    print("pred 1")
    for i in range(samples):
        xpred = xvar[1:2]
        ypreds = predmodel(xpred)
        print(ypreds.data.numpy())


def run_cgan(lr=0.001, epochs=1000):
    x = np.arange(0, 5, dtype="int64")
    # x = np.ones((5,), dtype="int64")
    y = np.arange(0, 5, dtype="int64")
    y2 = np.arange(0, 5, dtype="int64") + 1
    y3 = np.arange(0, 5, dtype="int64") + 2
    y = np.stack([y, y2, y3], 1)
    d = dict(zip([chr(i) for i in range(7)], range(7)))
    xemb = q.WordEmb(5, worddic=d)
    yemb = q.WordEmb(5, worddic=d)  # first 10 for means, second 10 for variances
    ylin = torch.nn.Linear(7, 5)
    ff_out = q.Forward(15, 5)
    out = q.WordLinout(5, worddic=d, bias=False)
    z_dist = torch.distributions.Normal(torch.zeros(10), torch.ones(10))

    class CGen(torch.nn.Module):
        def __init__(self):
            super(CGen, self).__init__()
            self.xemb = xemb
            self.ff_out = ff_out
            self.out = out
            self.sm = torch.nn.Softmax(1)

        def forward(self, x):
            xembs, _ = self.xemb(x)
            z = q.var(z_dist.sample_n(x.size(0))).v
            predvec = torch.cat([xembs, z], 1)
            outvec = self.out(self.ff_out(predvec))
            ret = self.sm(outvec)
            return ret

    class CDisc(torch.nn.Module):
        def __init__(self):
            super(CDisc, self).__init__()
            self.ylin = ylin
            self.xemb = xemb
            self.ff_disc = q.Forward(10, 10)
            self.summ_disc = q.Forward(10, 1)

        def forward(self, y, x):
            xembs, _ = self.xemb(x)
            yvecs = self.ylin(y)
            predvec = torch.cat([xembs, yvecs], 1)
            out = self.summ_disc(self.ff_disc(predvec)).squeeze(1)
            out = torch.nn.Sigmoid()(out)
            return out

    gen = CGen()
    disc = CDisc()

    genoptim = torch.optim.Adam(q.params_of(gen), lr=lr)
    discoptim = torch.optim.Adam(q.params_of(disc), lr=lr)
    xvar = q.var(x).v
    yvar = q.var(y).v

    for i in range(epochs):
        for j in range(5):  # disc updates
            discoptim.zero_grad()
            goldvecs = q.var(torch.zeros(xvar.size(0), 7).scatter_(1, yvar.data, 1)).v
            samplegold = goldvecs.multinomial(1)
            truedata = q.var(torch.zeros(goldvecs.size()).scatter_(1, samplegold.data, 1)).v
            fakedata = gen(xvar)
            discloss = torch.log(disc(truedata, xvar)) + torch.log(1 - disc(fakedata.detach(), xvar))
            discloss = torch.mean(-discloss)
            print("D loss:", discloss.data[0])
            discloss.backward()
            discoptim.step()
        for k in range(5):  # generator updates
            genoptim.zero_grad()
            fakedata = gen(xvar)
            genloss = torch.log(disc(fakedata, xvar))
            genloss = torch.mean(-genloss)
            print("G loss: ", genloss.data[0])
            genloss.backward()
            genoptim.step()






if __name__ == "__main__":
    q.argprun(run_cvae)