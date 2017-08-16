import torch
from torch import nn
from torch.nn import functional as F
from qelos.util import name2fn, issequence
from qelos.containers import ModuleList


class Lambda(nn.Module):
    """ Lambda layer. Don't use for any operations introducing parameters (won't get registered maybe). """
    def __init__(self, fn, register_params=None, register_modules=None):
        super(Lambda, self).__init__()
        self.fn = fn
        # optionally registers passed modules and params
        if register_modules is not None:
            if not issequence(register_modules):
                register_modules = [register_modules]
            self.extra_modules = ModuleList(register_modules)
        if register_params is not None:
            if not issequence(register_params):
                register_params = [register_params]
            self.extra_params = nn.ParameterList(register_params)

    def forward(self, *x, **kwargs):
        return self.fn(*x, **kwargs)


class Stack(nn.Module):
    def __init__(self, *layers):
        super(Stack, self).__init__()
        self.layers = ModuleList(list(layers))

    def add(self, *layers):
        self.layers.extend(list(layers))

    def forward(self, *x, **kw):
        for layer in self.layers:
            x = layer(*x, **kw)
            if not issequence(x):
                x = (x,)
        if len(x) == 1:
            x = x[0]
        return x

    # TODO: stack generator


class Softmax(nn.Module):
    def __init__(self, temperature=1., _log_in_train=False):
        super(Softmax, self).__init__()
        self.temperature = temperature
        self._log = False
        self._logafter = False
        self._log_in_train = _log_in_train

    def logsumexp(self, x, mask=None):
        #x = torch.add(x, torch.log(mask))
        x_max, _ = torch.max(torch.add(x, torch.log(mask)), 1, keepdim=True)
        x = x.mul(mask)
        x = x.sub(x_max)
        x.exp_()
        if mask is not None:
            x = torch.mul(x, mask)
        x_sum = torch.sum(x, 1, keepdim=True)
        x_sum = x_sum.log()
        x_sum.add_(x_max)
        return x_sum

    def forward(self, x, mask=None, temperature=None):
        temperature = temperature if temperature is not None else self.temperature
        mask = mask if mask is not None else x.mask if hasattr(x, "mask") else None
        xndim = x.dim()
        s = x.size()
        seqmask = None
        if mask is not None:
            if mask.dim() == x.dim():
                pass
            elif mask.dim() == x.dim() - 1:
                seqmask = mask
                mask = None
            else:
                raise Exception("bad mask shape")
        if xndim > 2:
            x = x.view(-1, x.size(-1))
            if mask is not None:
                maskshape = mask.size()
                mask = mask.view(-1, mask.size(-1))
        x = x / temperature
        if mask is None:
            if self._log:
                o_exp = F.log_softmax(x)
            else:
                o_exp = F.softmax(x)
        else:
            if self._log:
                o_exp_sum = self.logsumexp(x, mask)
                masklog = mask.log()
                o_exp = x.add(masklog)
                o_exp.sub_(o_exp_sum)
            else:
                x_mins, _ = torch.min(x, 1, keepdim=True)
                x.sub_(x_mins)
                x = torch.mul(x, mask)
                x.add_(x_mins)
                o_exp, _ = torch.max(x, 1, keepdim=True)
                o_exp = x - o_exp
                o_exp.exp_()
                o_exp = torch.mul(o_exp, mask)
                o_exp_sum = torch.sum(o_exp, 1, keepdim=True)
                o_exp = torch.div(o_exp, o_exp_sum)
                if self._logafter:
                    o_exp = torch.log(o_exp)
        retmask = None
        if xndim > 2:
            o_exp = o_exp.view(s)
            if mask is not None:
                mask = mask.view(maskshape)
                retmask = mask
        if seqmask is not None:
            retmask = seqmask
        if retmask is not None:
            return o_exp, retmask
        else:
            return o_exp


class LogSoftmax(Softmax):      # TODO: some numerical instability
    def __init__(self, temperature=1.):
        super(LogSoftmax, self).__init__(temperature=temperature)
        self._log = True


class SoftmaxLog(Softmax):      # TODO: might need something extra for use in training
    def __init__(self, temperature=1.):
        super(SoftmaxLog, self).__init__(temperature=temperature)
        self._logafter = True


class Forward(nn.Module):
    def __init__(self, indim, outdim, activation="tanh", use_bias=True):
        super(Forward, self).__init__()
        self.lin = nn.Linear(indim, outdim, bias=use_bias)
        self.activation = name2fn(activation)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x):
        x = self.lin(x)
        x = self.activation(x)
        return x


class Distance(nn.Module):          # TODO: all distances must work with 2D/2D, 3D/2D and 3D/3D
    pass


class DotDistance(Distance):
    def forward(self, data, crit):        # (batsize, seqlen, dim), (batsize, dim)
        if data.dim() == 2:               # if data is (batsize, dim),
            data = data.unsqueeze(1)      #     make (batsize, 1, dim)
        if crit.dim() == 2:               # if crit is (batsize, dim)
            crit = crit.unsqueeze(-1)     #     make crit (batsize, dim, 1)
        else:                             # else crit must be (batsize, seqlen, dim)
            crit = crit.permute(0, 2, 1)  #     but we need (batsize, dim, seqlen)
        dist = torch.bmm(data, crit)      # batched mat dot --> (batsize,1,1) or (batsize, lseqlen,1) or (batsize, lseqlen, rseqlen)
        return dist.squeeze()


class CosineDistance(DotDistance):
    def forward(self, data, crit):        # (batsize, [lseqlen,] dim), (batsize, [rseqlen,] dim) 
        dots = super(CosineDistance, self).forward(data, crit)
        lnorms = data.norm(2, -1)         # (batsize, [lseqlen])
        rnorms = crit.norm(2, -1)         # (batsize, [rseqlen])
        if data.dim() == 3:               # if data is (batsize, lseqlen, dim)
            rnorms = rnorms.unsqueeze(1)  #     make crit norms (batsize, 1) or (batsize, 1, rseqlen)
            if crit.dim() == 3:                         # (batsize, rseqlen, dim)
                lnorms = lnorms.unsqueeze(2)  # make data norms (batsize, lseqlen, 1)
        dots = dots.div(lnorms)
        dots = dots.div(rnorms)
        return dots

# TODO: Euclidean and LNorm distances


class ForwardDistance(Distance):
    memsave = False
    def __init__(self, ldim, rdim, aggdim, activation="tanh", use_bias=True):
        super(ForwardDistance, self).__init__()
        self.lblock = nn.Linear(indim=ldim, outdim=aggdim, use_bias=use_bias)
        self.rblock = nn.Linear(indim=rdim, outdim=aggdim, use_bias=use_bias)
        self.activation = name2fn(activation)
        self.agg = nn.Parameter(torch.FloatTensor(aggdim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform(self.agg, -0.01, 0.01)
        self.lblock.reset_parameters()
        self.rblock.reset_parameters()

    def forward(self, data, crit):      # (batsize, [lseqlen,] dim), (batsize, [rseqlen,] dim) 
        datalin = self.lblock(data)
        critlin = self.rblock(crit)
        if data.dim() == 3:             # (batsize, lseqlen, dim)
            if crit.dim() == 2:         # (batsize, dim)
                critlin = critlin.unsqueeze(1)      # --> (batsize, 1, dim)
            else:                       # (batsize, rseqlen, dim)
                if not self.memsave:
                    datalin = datalin.unsqueeze(2)      # --> (batsize, lseqlen, 1, dim)
                    critlin = critlin.unsqueeze(1)      # --> (batsize, 1, rseqlen, dim)
                else:
                    pass
                    # TODO: memsave --> loop over 2D slices of 3D crit, do we need memsave?
        linsum = datalin + critlin      # (batsize, dim) or (batsize, lseqlen, dim) or (batsize, lseqlen, rseqlen, dim)
        linsum = self.activation(linsum)
        dists = torch.matmul(linsum, self.agg)  # TODO: check if this works for 3D x 1D and 4D x 1D
        return dists


class BilinearDistance(Distance):       
    def __init__(self, ldim, rdim):
        super(BilinearDistance, self).__init__()
        self.block = nn.Bilinear(ldim, rdim, 1, bias=False)

    def forward(self, data, crit):      # (batsize, [lseqlen,] dim), (batsize, [rseqlen,] dim) 
        if data.dim() == 3:
            l = data.view(-1, data.size(-1))        # (batsize * lseqlen, dim)
            r = crit.unsqueeze(1).expand_as(data).contiguous().view(-1, crit.size(-1))  # (batsize * lseqlen, dim)
            # TODO: 3D crit and memsave
        bilinsum = self.block(l, r)
        dists = bilinsum.squeeze()
        dists = dists.view(*data.size()[:-1])
        return dists


class TrilinearDistance(Distance):
    def __init__(self, ldim, rdim, aggdim, activation="tanh", use_bias=True):
        super(TrilinearDistance, self).__init__()
        self.block = nn.Bilinear(ldim, rdim, aggdim, bias=use_bias)
        self.activation = name2fn(activation)
        self.agg = nn.Parameter(torch.FloatTensor(aggdim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform(self.agg, -0.01, 0.01)

    def forward(self, data, crit):
        if data.dim() == 3:
            l = data.view(-1, data.size(-1))
            r = crit.unsqueeze(1).expand_as(data).contiguous().view(-1, crit.size(-1))
        bilinsum = self.block(l, r)
        bilinsum = self.activation(bilinsum)
        dists = torch.matmul(bilinsum, self.agg)
        dists = dists.view(*data.shape()[:-1])
        return dists
