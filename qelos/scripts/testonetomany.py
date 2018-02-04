import qelos as q
import torch
import numpy as np

import os, pickle as pkl


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


def run_vgae(lr=0.001, epochs=2000, batsize=50):
    vocsize = 15
    embdim = 5
    innerdim = 7
    xdata = np.random.randint(0, vocsize, (1000,), dtype="int64")
    xloader = q.dataload(xdata, batch_size=batsize)

    d = dict(zip([chr(i) for i in range(vocsize)], range(vocsize)))
    xembs = q.WordEmb(embdim, worddic=d)

    z_dist = torch.distributions.Normal(torch.zeros(2), torch.ones(2))

    class VAE(torch.nn.Module):
        def __init__(self, decoder):
            super(VAE, self).__init__()
            self.xembs = xembs
            self.ff_mean = torch.nn.Linear(embdim, 2)
            self.ff_sigma = torch.nn.Linear(embdim, 2)
            self.decoder = decoder
            self.z_dist = z_dist
            self.ff_inner = torch.nn.Sequential(q.Forward(2, innerdim, "relu"),
                                                #q.Forward(5, 5, "relu"),
                                                torch.nn.Linear(innerdim, 2))
            # self.ff_inner = None

        def forward(self, x):
            xvec, _ = self.xembs(x)
            means, logvar = self.ff_mean(xvec), self.ff_sigma(xvec)
            std = torch.exp(logvar * 0.5)
            z = q.var(self.z_dist.sample_n(x.size(0))).v
            sample = means + std * z        # 2D gaussian
            if False and self.ff_inner is not None:
                sample = self.ff_inner(sample)
            # inner transform comes here
            ret = self.decoder(sample)
            return ret, means, logvar, sample

    class VAE_Decoder(torch.nn.Module):
        def __init__(self):
            super(VAE_Decoder, self).__init__()
            self.ff_out = torch.nn.Linear(2, vocsize)
            self.sm = torch.nn.Softmax(1)

        def forward(self, code):
            ret = self.sm(self.ff_out(code))
            return ret

    class PriorDisc(torch.nn.Module):       # for getting posterior latent closer to prior latent
        def __init__(self):
            super(PriorDisc, self).__init__()
            self.ff_disc = torch.nn.Sequential(q.Forward(2, innerdim, "relu"),
                                               q.Forward(innerdim, innerdim, "relu"),
                                               q.Forward(innerdim, innerdim, "relu"))
            self.summ_disc = torch.nn.Linear(innerdim, 1)

        def forward(self, code):
            out = self.summ_disc(self.ff_disc(code)).squeeze(1)
            return out

    class PriorGen(torch.nn.Module):
        def __init__(self):
            super(PriorGen, self).__init__()
            self.ff_gen = torch.nn.Sequential(q.Forward(2, innerdim, "relu"),
                                              q.Forward(innerdim, innerdim, "relu"),
                                              q.Forward(innerdim, innerdim, "relu"),
                                              torch.nn.Linear(innerdim, 2))

        def forward(self, prior):
            out = self.ff_gen(prior)
            return out

    class EmpiricalKL(torch.nn.Module):
        def __init__(self, prior=None):
            super(EmpiricalKL, self).__init__()
            self.prior_means = q.var(torch.zeros(2)).v
            self.prior_sigmas = q.var(torch.ones(2)).v

        def forward(self, samples):
            samples_mean = samples.mean(0)
            samples_sigmas = torch.mean((samples - samples_mean.unsqueeze(0))**2, 0)
            logvar = torch.log(samples_sigmas)
            kld = -.5 * torch.sum(1 + logvar - (samples_mean ** 2) - logvar.exp())
            return kld

    decoder = VAE_Decoder()
    vae = VAE(decoder)
    disc = PriorDisc()
    empkld = EmpiricalKL()

    optim_vae = torch.optim.Adam(q.paramgroups_of(vae), lr=lr)
    optim_disc = torch.optim.Adam(q.paramgroups_of(disc), lr=lr)

    postdisc = PriorDisc()
    postgen = PriorGen()
    optim_postdisc = torch.optim.Adam(q.paramgroups_of(postdisc), lr=lr)
    optim_postgen = torch.optim.Adam(q.paramgroups_of(postgen), lr=lr)

    disciters = 5
    prioriters = 5
    grad_penalty_weight = 1.

    def infdataiter(loader):
        while True:
            for b in xloader:
                yield b

    infdata = infdataiter(xloader)

    use_emp_kld = False
    disc_loss = q.var(torch.zeros(1)).v

    j = 0
    for i in range(epochs):
        batch = next(infdata)
        if vae.ff_inner is not None and not use_emp_kld:
            for j in range(prioriters):      # train discriminator
                for k in range(disciters):
                    discbatch = q.var(next(infdata)[0]).v
                    prob_y, means, logvar, priorsamplev = vae(discbatch)
                    priorsample = priorsamplev.data
                    optim_disc.zero_grad()
                    realprior = z_dist.sample_n(priorsample.size(0))
                    interp_alpha = priorsample.new(priorsample.size(0), 1).uniform_(0, 1)
                    interp_points = q.var(interp_alpha * realprior + (1 - interp_alpha) * priorsample, requires_grad=True).cuda(priorsample).v
                    disc_interp_grad, = torch.autograd.grad(disc(interp_points).sum(), interp_points, create_graph=True)
                    lip_grad_norm = disc_interp_grad.view(priorsample.size(0), -1).norm(2, 1)
                    lip_loss = grad_penalty_weight * ((lip_grad_norm - 1.).clamp(min=0) ** 2).mean()
                    core_loss = disc(priorsamplev.detach()) - disc(q.var(realprior).cuda(priorsamplev).v)
                    disc_loss = core_loss.mean() + lip_loss.mean()
                    disc_loss.backward()
                    optim_disc.step()
                priorbatch = q.var(next(infdata)[0]).v
                prob_y, means, logvar, priorsamplev = vae(priorbatch)
                optim_vae.zero_grad()
                priorloss = -disc(priorsamplev)
                priorloss.mean().backward()
                optim_vae.step()

        gold = q.var(batch[0]).v
        optim_vae.zero_grad()
        probs, means, logvar, sample = vae(gold)
        loss_recons = - torch.log(torch.gather(probs, 1, gold.unsqueeze(1))).squeeze(1)
        loss_prior = -0.5 * torch.sum(1 + logvar - (means ** 2) - logvar.exp(), 1)
        vae_loss = loss_recons.mean()
        if vae.ff_inner is None:
            loss_prior = loss_prior.mean()
            vae_loss += loss_prior
        elif not use_emp_kld:
            loss_prior = -disc(sample).mean()
            vae_loss += loss_prior
        else:
            loss_prior = empkld(sample)
            vae_loss += loss_prior
        vae_loss.backward()
        optim_vae.step()
        print(i, loss_recons.mean().data[0], disc_loss.data[0], loss_prior.data[0])

        for k in range(prioriters):
            for j in range(disciters):  # train discriminator
                discbatch = q.var(next(infdata)[0]).v
                _, _, _, realv = vae(discbatch)
                # realv = realv.detach()
                real = realv.data
                optim_postdisc.zero_grad()
                z = q.var(z_dist.sample_n(real.size(0))).v
                fakev = postgen(z)
                fake = fakev.data
                interp_alpha = real.new(real.size(0), 1).uniform_(0, 1)
                interp_points = q.var(interp_alpha * real + (1 - interp_alpha) * fake, requires_grad=True).cuda(real).v
                disc_interp_grad, = torch.autograd.grad(postdisc(interp_points).sum(), interp_points, create_graph=True)
                lip_grad_norm = disc_interp_grad.view(real.size(0), -1).norm(2, 1)
                lip_loss = grad_penalty_weight * ((lip_grad_norm - 1.).clamp(min=0) ** 2)
                core_loss = postdisc(fakev.detach()) - postdisc(realv.detach())
                disc_loss = core_loss.mean() + lip_loss.mean()
                disc_loss.backward()
                optim_postdisc.step()
            optim_postgen.zero_grad()
            real = q.var(next(infdata)[0]).v
            z = q.var(z_dist.sample_n(real.size(0))).v
            gen_loss = -postdisc(postgen(z)).mean()
            gen_loss.backward()
            optim_postgen.step()
        print(i, core_loss.mean().data[0])



    posttrain = False
    if posttrain:
        postepochs = 500
        print("training post")
        for i in range(postepochs):
            for j in range(disciters):      # train discriminator
                discbatch = q.var(next(infdata)[0]).v
                _, _, _, realv = vae(discbatch)
                real = realv.data
                optim_postdisc.zero_grad()
                z = q.var(z_dist.sample_n(real.size(0))).v
                fakev = postgen(z)
                fake = fakev.data
                interp_alpha = real.new(real.size(0), 1).uniform_(0, 1)
                interp_points = q.var(interp_alpha * real + (1 - interp_alpha) * fake, requires_grad=True).cuda(real).v
                disc_interp_grad, = torch.autograd.grad(postdisc(interp_points).sum(), interp_points, create_graph=True)
                lip_grad_norm = disc_interp_grad.view(real.size(0), -1).norm(2, 1)
                lip_loss = grad_penalty_weight * ((lip_grad_norm - 1.).clamp(min=0) ** 2)
                core_loss = postdisc(fakev.detach()) - postdisc(realv.detach())
                disc_loss = core_loss.mean() + lip_loss.mean()
                disc_loss.backward()
                optim_postdisc.step()
            optim_postgen.zero_grad()
            real = q.var(next(infdata)[0]).v
            z = q.var(z_dist.sample_n(real.size(0))).v
            gen_loss = -postdisc(postgen(z)).mean()
            gen_loss.backward()
            optim_postgen.step()
            print(i, core_loss.mean().data[0])


    numsam = 30
    np.set_printoptions(precision=3, suppress=True)
    print("sampling {}".format(numsam))
    for i in range(numsam):
        sample = q.var(z_dist.sample_n(1)).v
        if posttrain:
            sample = postgen(sample)
        preds = decoder(sample)
        print(preds.data.numpy()[0])


    # visualize latent space
    toplot = []
    for i in range(vocsize):
        x = q.var(np.ones((100,), dtype="int64") * i).v
        _, _, _, sample = vae(x)
        toplot.append(sample.data.numpy())
    z = q.var(z_dist.sample_n(500)).v
    postsamples = z
    # if posttrain:
    postsamples = postgen(z)

    pkl.dump((toplot, postsamples.data.numpy()), open("otm.npz", "w"))




def plot_space(p="otm.npz"):
    toplot = pkl.load(open(p))
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    for toplote in toplot:
        plt.scatter(toplote)


def run_cvgae(lr=0.001, epochs=1000):
    x = np.arange(0, 100, dtype="int64")
    # x = np.ones((5,), dtype="int64")
    y = np.arange(0, 100, dtype="int64")
    y2 = np.arange(0, 100, dtype="int64") + 1
    y3 = np.arange(0, 100, dtype="int64") + 2
    y = np.stack([y, y2, y3], 1)
    d = dict(zip([chr(i) for i in range(103)], range(103)))
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
            self.ff_sigma = torch.nn.Linear(5, 10)  # predicts log of var
            self.ff_inner = torch.nn.Sequential(q.Forward(10,10,"relu"),
                                                # q.Forward(10,10,"relu"),
                                                q.Forward(10,10,"tanh"))
            self.ff_out = ff_out

        def forward(self, x, y):
            # conditional encoder (x is input)
            yembs, _ = self.yemb(y)
            xembs, _ = self.xemb(x)
            enco = self.ff_fromx(xembs) + self.ff_fromy(yembs)
            means, logvar = self.ff_mean(enco), self.ff_sigma(enco)
            std = torch.exp(logvar * 0.5)
            z = q.var(self.z_dist.sample_n(x.size(0))).v
            sample = means + std * z
            sample = self.ff_inner(sample)
            predvec = torch.cat([sample, xembs], 1)
            outvec = self.out(self.ff_out(predvec))
            ret = self.sm(outvec / q.v(temp))
            return ret, means, logvar, sample

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

    class PriorDisc(torch.nn.Module):
        def __init__(self):
            super(PriorDisc, self).__init__()
            self.ff_disc = torch.nn.Sequential(q.Forward(10, 10, "relu"),
                                               q.Forward(10, 10, "relu"))
            self.summ_disc = q.Forward(10, 1)

        def forward(self, y):   # y is a sample from the prior
            out = self.summ_disc(self.ff_disc(y)).squeeze(1)
            out = torch.nn.Sigmoid()(out)
            return out

    model = CVAE()
    predmodel = CVAE_Predict()
    disc = PriorDisc()
    grad_penalty_weight = 1.

    optim = torch.optim.Adam(q.params_of(model), lr=lr)
    discoptim = torch.optim.Adam(q.paramgroups_of(disc), lr=lr)
    xvar = q.var(x).v
    yvar = q.var(y).v

    epochs = 5000
    disciters = 5
    prioriters = 5

    kl_weight = q.hyperparam(1.)

    for i in range(epochs):
        # kl_weight.v = float(np.tanh(i/1000.-7.)/2.+.5)
        # kl_weight.v = float(-np.cos(i / 500.) / 2. + .5)
        # sample a y
        for j in range(disciters):      # train discriminator
            goldvecs = q.var(torch.zeros(xvar.size(0), 103).scatter_(1, yvar.data, 1)).v
            samplegold = goldvecs.multinomial(1).squeeze(1)
            prob_y, means, logvar, priorsamplev = model(xvar, samplegold)
            priorsample = priorsamplev.data
            discoptim.zero_grad()
            realprior = z_dist.sample_n(priorsample.size(0))
            interp_alpha = priorsample.new(priorsample.size(0), 1).uniform_(0, 1)
            interp_points = q.var(interp_alpha * realprior + (1 - interp_alpha) * priorsample, requires_grad=True).cuda(priorsample).v
            disc_interp_grad, = torch.autograd.grad(disc(interp_points).sum(), interp_points, create_graph=True)
            lip_grad_norm = disc_interp_grad.view(priorsample.size(0), -1).norm(2, 1)
            lip_loss = grad_penalty_weight * ((lip_grad_norm - 1.).clamp(min=0) ** 2).mean()
            core_loss = disc(priorsamplev.detach()) - disc(q.var(realprior).cuda(priorsamplev).v)
            total_loss = core_loss.mean() + lip_loss.mean()
            total_loss.backward()
            discoptim.step()

        for k in range(prioriters):
            goldvecs = q.var(torch.zeros(xvar.size(0), 103).scatter_(1, yvar.data, 1)).v
            samplegold = goldvecs.multinomial(1).squeeze(1)
            prob_y, means, logvar, priorsample = model(xvar, samplegold)
            optim.zero_grad()
            priorloss = -disc(priorsample)
            priorloss.mean().backward()
            optim.step()

        print(total_loss.mean().data.numpy(), core_loss.mean().data.numpy(), lip_loss.mean().data.numpy())

        optim.zero_grad()

        goldvecs = q.var(torch.zeros(xvar.size(0), 103).scatter_(1, yvar.data, 1)).v
        samplegold = goldvecs.multinomial(1).squeeze(1)
        prob_y, means, logvar, priorsample = model(xvar, samplegold)

        priorloss = -disc(priorsample)
        goldprobs = torch.gather(prob_y, 1, samplegold.unsqueeze(1)).squeeze(1)
        core_loss = -torch.log(goldprobs)       # reconstruction loss

        print(i, core_loss.mean().data[0], priorloss.mean().data[0])
        # l = torch.mean(core_loss + q.v(kl_weight) * priorloss)
        l = core_loss.mean()
        # print(l)
        l.backward()
        optim.step()
        # temp.v = (1. - 0.99 * i/epochs)
        # print(i, q.v(temp))

    samples = 20
    np.set_printoptions(precision=3, suppress=True)
    print("pred 0")
    for i in range(samples):
        xpred = xvar[0:1]
        ypreds = predmodel(xpred)
        print(ypreds.data.numpy()[:10])
    print("pred 1")
    for i in range(samples):
        xpred = xvar[1:2]
        ypreds = predmodel(xpred)
        print(ypreds.data.numpy()[:10])
    print("pred 2")
    for i in range(samples):
        xpred = xvar[2:3]
        ypreds = predmodel(xpred)
        print(ypreds.data.numpy()[:10])


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
    q.argprun(run_vgae)