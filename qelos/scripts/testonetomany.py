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






if __name__ == "__main__":
    q.argprun(run)