import torch
from torch import nn
from torch.nn import functional as F
import qelos as q
# from qelos.util import issequence
# from qelos import name2fn
# from qelos.containers import ModuleList
import numpy as np


class Lambda(nn.Module):
    """ Lambda layer. Don't use for any operations introducing parameters (won't get registered maybe). """
    def __init__(self, fn, register_params=None, register_modules=None):
        super(Lambda, self).__init__()
        self.fn = fn
        # optionally registers passed modules and params
        if register_modules is not None:
            if not q.issequence(register_modules):
                register_modules = [register_modules]
            self.extra_modules = q.ModuleList(register_modules)
        if register_params is not None:
            if not q.issequence(register_params):
                register_params = [register_params]
            self.extra_params = nn.ParameterList(register_params)

    def forward(self, *x, **kwargs):
        ret = self.fn(*x, **kwargs)
        return ret


class StackSpecFunction(Lambda):
    """ Only used inside stacks."""
    @staticmethod
    def get_(spece, a, k, g):
        if isinstance(spece, set):
            assert (len(spece) == 1)
            return k[list(spece)[0]]
        elif isinstance(spece, list):
            assert (len(spece) == 1)
            return g[spece[0]]
        else:
            return a[spece]


class argmap(StackSpecFunction):
    """
    Only used inside stacks.
    Provided function must follow .forward()'s specifications
    """
    def forward(self, args, kwargs, saved_slots):
        """
        :param args:    sequence outputs of previous layer
        :param kwargs:  stack's current **kw to be fed to next layer
        :param globols: stack's current saved slots
        :return:        tuple of (*args, **kwargs) to feed to next layer
        """
        return self.fn(args, kwargs, saved_slots)

    @classmethod
    def spec(cls, *argspec, **kwargspec):
        """
        Specs for args and kwargs for next layer
        Values from the sequence output of previous layer (args) can
        be accessed by integers.
        Values from the dictionary output of previous layer (kwargs), which can occur if this is preceded by another argmap
        can be accessed by keys wrapped in a set (e.g. {1} or {"mask"}.
        Values from the saved stack slots can be accessed
        by keys wrapped in a list (e.g. [1] or ["mask"].

        For example, if layer A outputs a tuple (x, y, z),
        and the stack function-wide **kw contains a "t" key,
        and a different key "t" was saved in stack,

        argmap.from_spec(0, 1, {"t"}, z=2, t=["t"])

        will map x to first input to next layer B,
        map y to second input to B,
        map the **kw's "t" to the third input to B,
        map z to the keyword input "z" of B,
        and map stack's saved "t" to keyword input "t" of B.

        """
        this = cls

        def dict_arg_map(args, kwargs, saved_slots):
            outargs = []
            outkwargs = {}
            for argspec_e in argspec:
                outargs.append(this.get_(argspec_e, args, kwargs, saved_slots))
            for kwargspec_k, kwargspec_v in kwargspec.items():
                outkwargs[kwargspec_k] = this.get_(kwargspec_v, args, kwargs, saved_slots)
            return outargs, outkwargs
        return cls(dict_arg_map)


class argsave(StackSpecFunction):
    """
    Only used inside stacks.
    Provided function must accept .forward()'s argument specification
    and return an update dictionary to apply to stack's saved slots (globols).
    Can be used to implement shortcuts within a Stack.
    """
    def forward(self, args, kwargs, saved_slots):
        """
        Updates provided globols.
        :param args:    sequence outputs of previous layer
        :param kwargs:  stack's current **kw to be fed to next layer
        :param globols: stack's current saved slots
        :return:        globols
        """
        upd = self.fn(args, kwargs, saved_slots)
        saved_slots.update(upd)
        return saved_slots

    @classmethod
    def spec(cls, **savespec):
        """
        Specs for saving some current stack variables
        (previous layer sequence outputs (args), kwargs, and saved stack globals)
        to stack-global saved variables (_globals).
        Follows the same specification format as argmap,
        except only keyword-based storage is possible.

        For example, if layer A outputs a tuple (x, y, z),
        and the stack function-wide **kw contains a "t" key,
        and a different key "t" was saved in stack,

        cls.from_spec(t={"t"}, z=2, prev_t=["t"])

        will save the **kw "t" to stack's "t" slot,
        save z to to stack's "z" slot, and
        save stack's saved "t" to stack's "prev_t" slot.
        The update to stack's saved slots is carried out at once, so
        the stack's saved "t" will be backed up
        in stack's saved slot "prev_t" and saved slot "t"
        will be overwritten by **kw's "t".

        """
        this = cls

        def dict_save_map(args, kwargs, saved_slots):
            update = {}
            for spec_key, spec_val in savespec.items():
                update[spec_key] = this.get_(spec_val, args, kwargs, saved_slots)
            return update
        return cls(dict_save_map)


class Stack(nn.Module):
    def __init__(self, *layers):
        super(Stack, self).__init__()
        self.layers = q.ModuleList()
        self.add(*layers)
        self._saved_slots = {}      # not for params

    def add(self, *layers):
        self._add(*layers)

    def _add(self, *layers):
        self.layers.extend(list(layers))

    def forward(self, *x, **kw):
        y_l = x
        args, kwargs = None, None
        argmapped = False
        for layer in self.layers:
            if argmapped:
                rargs = args
                rkw = {}
                rkw.update(kw)
                rkw.update(kwargs)
            else:
                rargs = y_l
                rkw = kw
            if isinstance(layer, q.argmap):
                args, kwargs = layer(rargs, rkw, self._saved_slots)
                argmapped = True
            elif isinstance(layer, argsave):
                globols = layer(rargs, rkw, self._saved_slots)
            else:
                y_l = layer(*rargs, **rkw)
                argmapped = False
            if not q.issequence(y_l) and not argmapped:
                y_l = tuple([y_l])
        if argmapped:
            ret = args
        else:
            ret = y_l
        if len(ret) == 1:
            ret = ret[0]
        return ret

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
        mask = mask.float() if mask is not None else mask
        xndim = x.dim()
        s = x.size()
        seqmask = None
        x = x.contiguous()
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
        self.activation = q.name2fn(activation)()
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x):
        x = self.lin(x)
        x = self.activation(x)
        return x


class Distance(nn.Module):
    pass


class DotDistance(Distance):
    def forward(self, data, crit):        # (batsize, seqlen, dim), (batsize, dim)
        datadim, critdim = data.dim(), crit.dim()
        if data.dim() == 2:               # if datasets is (batsize, dim),
            data = data.unsqueeze(1)      #     make (batsize, 1, dim)
        if crit.dim() == 2:               # if crit is (batsize, dim)
            crit = crit.unsqueeze(-1)     #     make crit (batsize, dim, 1)
        else:                             # else crit must be (batsize, seqlen, dim)
            crit = crit.permute(0, 2, 1)  #     but we need (batsize, dim, seqlen)
        dist = torch.bmm(data, crit)      # batched mat dot --> (batsize,1,1) or (batsize, lseqlen,1) or (batsize, lseqlen, rseqlen)
        ret = dist.squeeze(1) if datadim == 2 else dist
        ret = ret.squeeze(-1) if critdim == 2 else ret
        return ret


class CosineDistance(DotDistance):
    def forward(self, data, crit):        # (batsize, [lseqlen,] dim), (batsize, [rseqlen,] dim) 
        dots = super(CosineDistance, self).forward(data, crit)
        lnorms = data.norm(2, -1)         # (batsize, [lseqlen])
        rnorms = crit.norm(2, -1)         # (batsize, [rseqlen])
        if data.dim() == 3:               # if datasets is (batsize, lseqlen, dim)
            rnorms = rnorms.unsqueeze(1)  #     make crit norms (batsize, 1) or (batsize, 1, rseqlen)
            if crit.dim() == 3:                         # (batsize, rseqlen, dim)
                lnorms = lnorms.unsqueeze(2)  # make datasets norms (batsize, lseqlen, 1)
        dots = dots.div(torch.mul(lnorms, rnorms).clamp(1e-6))
        return dots


class LNormDistance(Distance):
    def __init__(self, L=2, **kw):
        super(LNormDistance, self).__init__(**kw)
        self.L = L

    def forward(self, data, crit):      # (batsize, [lseqlen,] dim), (batsize, [rseqlen,], dim)
        if data.dim() == 3 and crit.dim() == 2:
            crit = crit.unsqueeze(1)
        elif data.dim() == 3 and crit.dim() == 3:
            data = data.unsqueeze(2)
            crit = crit.unsqueeze(1)
        temp = torch.abs(data - crit)
        maximum, _ = torch.max(temp, -1, keepdim=True)
        maximum_red, _ = torch.max(temp, -1, keepdim=False)
        temp = (temp / maximum) ** self.L
        #lognorm = T.logsumexp(temp * self.L, axis=-1) / self.L
        ret = maximum_red * (torch.sum(temp, -1).clamp(min=1e-6) ** (1./self.L))
        return ret


class ForwardDistance(Distance):
    memsave = False
    def __init__(self, ldim, rdim, aggdim, activation="tanh", use_bias=True):
        super(ForwardDistance, self).__init__()
        self.lblock = nn.Linear(ldim, aggdim, bias=use_bias)
        self.rblock = nn.Linear(rdim, aggdim, bias=use_bias)
        self.activation = q.name2fn(activation)()
        self.agg = nn.Parameter(torch.FloatTensor(aggdim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform(self.agg, -0.1, 0.1)
        self.lblock.reset_parameters()
        self.rblock.reset_parameters()

    def forward(self, data, crit):      # (batsize, [lseqlen,] dim), (batsize, [rseqlen,] dim)
        if crit.dim() == 3 and self.memsave:
            acc = []
            for i in range(crit.size(1)):
                acc.append(self.forward(data, crit[:, i]))
            acc = torch.stack(acc, 2)
            return acc
        datalin = self.lblock(data)
        critlin = self.rblock(crit)
        if data.dim() == 3:             # (batsize, lseqlen, dim)
            if crit.dim() == 2:         # (batsize, dim)
                critlin = critlin.unsqueeze(1)      # --> (batsize, 1, dim)
            else:                       # (batsize, rseqlen, dim)
                datalin = datalin.unsqueeze(2)      # --> (batsize, lseqlen, 1, dim)
                critlin = critlin.unsqueeze(1)      # --> (batsize, 1, rseqlen, dim)
        linsum = datalin + critlin      # (batsize, dim) or (batsize, lseqlen, dim) or (batsize, lseqlen, rseqlen, dim)
        linsum = self.activation(linsum)
        dists = torch.matmul(linsum, self.agg)  # TODO: check if this works for 3D x 1D and 4D x 1D
        return dists


class BilinearDistance(Distance):
    memsave = False
    def __init__(self, ldim, rdim):
        super(BilinearDistance, self).__init__()
        self.block = nn.Bilinear(ldim, rdim, 1, bias=False)

    def forward(self, data, crit):      # (batsize, [lseqlen,] dim), (batsize, [rseqlen,] dim)
        if crit.dim() == 3 and self.memsave:
            acc = []
            for i in range(crit.size(1)):
                acc.append(self.forward(data, crit[:, i]))
            acc = torch.stack(acc, 2)
            return acc
        l = data
        r = crit
        if data.dim() == 3:
            l = data.view(-1, data.size(-1))        # (batsize * lseqlen, dim)
            if crit.dim() == 2:
                r = crit.unsqueeze(1).repeat(1, data.size(1), 1).contiguous().view(-1, crit.size(-1))  # (batsize * lseqlen, dim)
            else:       # crit.dim() == 3
                l = l.unsqueeze(1).expand(l.size(0), r.size(1), data.size(-1)).contiguous().view(-1, data.size(-1))     # (batsize * lseqlen * rseqlen, dim)
                r = r.unsqueeze(1).repeat(1, data.size(1), 1, 1).view(-1, crit.size(-1))                                # (batsize * rseqlen * lseqlen, dim)
        bilinsum = self.block(l, r)
        dists = bilinsum.squeeze()
        if data.dim() == 3 and crit.dim() == 3:
            dists = dists.view(data.size(0), data.size(1), crit.size(1))
        else:
            dists = dists.view(*data.size()[:-1])
        return dists


class TrilinearDistance(Distance):
    memsave = False
    def __init__(self, ldim, rdim, aggdim, activation="tanh", use_bias=False):
        super(TrilinearDistance, self).__init__()
        self.block = nn.Bilinear(ldim, rdim, aggdim, bias=use_bias)
        self.activation = q.name2fn(activation)()
        self.agg = nn.Parameter(torch.FloatTensor(aggdim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform(self.agg, -0.01, 0.01)

    def forward(self, data, crit):
        if crit.dim() == 3 and self.memsave:
            acc = []
            for i in range(crit.size(1)):
                acc.append(self.forward(data, crit[:, i]))
            acc = torch.stack(acc, 2)
            return acc
        l = data
        r = crit
        if data.dim() == 3:
            l = data.view(-1, data.size(-1))
            if crit.dim() == 2:
                r = crit.unsqueeze(1).expand_as(data).contiguous().view(-1, crit.size(-1))  # (batsize * lseqlen, dim)
            else:       # crit.dim() == 3
                l = l.unsqueeze(1).repeat(1, r.size(1), 1).contiguous().view(-1, data.size(-1))     # (batsize * lseqlen * rseqlen, dim)
                r = r.unsqueeze(1).repeat(1, data.size(1), 1, 1).view(-1, crit.size(-1))                                # (batsize * rseqlen * lseqlen, dim)
        bilinsum = self.block(l, r)
        bilinsum = self.activation(bilinsum)
        dists = torch.matmul(bilinsum, self.agg)
        if data.dim() == 3 and crit.dim() == 3:
            dists = dists.view(data.size(0), data.size(1), crit.size(1))
        else:
            dists = dists.view(*data.size()[:-1])
        return dists


class SeqBatchNorm1d(nn.Module):

    """
    A batch normalization module which keeps its running mean
    and variance separately per timestep.
    """

    def __init__(self, num_features, max_length=None, eps=1e-5, momentum=0.1,
                 affine=True):
        """
        Most parts are copied from
        torch.nn.modules.batchnorm._BatchNorm.
        """

        super(SeqBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.max_length = max_length
        self.max_time = 0
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        if self.affine:
            self.weight = nn.Parameter(torch.FloatTensor(num_features))
            self.bias = nn.Parameter(torch.FloatTensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.max_time):
            running_mean_i = getattr(self, 'running_mean_{}'.format(i))
            running_var_i = getattr(self, 'running_var_{}'.format(i))
            running_mean_i.zero_()
            running_var_i.fill_(1)
        if self.affine:
            self.weight.data.fill_(0.1)
            self.bias.data.fill_(0.)

    def _check_input_dim(self, input_):
        if input_.size(1) != self.num_features:
            raise ValueError('got {}-feature tensor, expected {}'
                             .format(input_.size(1), self.num_features))

    def get_running_stat(self, which, time):
        if not hasattr(self, "running_{}_{}".format(which, time)):
            self.register_buffer("running_{}_{}".format(which, time),
                                 torch.zeros(self.num_features)
                                 if which == "mean"
                                 else torch.ones(self.num_features))
        return getattr(self, "running_{}_{}".format(which, time))

    def forward(self, input_, time):
        self._check_input_dim(input_)
        maxtime = self.max_length if self.training else self.max_time
        if maxtime is not None and time >= maxtime:
            time = maxtime - 1
        running_mean = self.get_running_stat("mean", time)
        running_var = self.get_running_stat("var", time)
        ret = F.batch_norm(
            input=input_, running_mean=running_mean, running_var=running_var,
            weight=self.weight, bias=self.bias, training=self.training,
            momentum=self.momentum, eps=self.eps)
        return ret

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' max_length={max_length}, affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))


class CReLU(nn.Module):
    def __init__(self):
        super(CReLU, self).__init__()
        self.pos_relu = F.relu
        self.neg_relu = F.relu

    def forward(self, x):
        left = self.pos_relu(x)
        right = self.neg_relu(-x)
        ret = torch.cat([left, right], -1)
        return ret


class Identity(nn.Module):
    def __init__(self, *a, **kw):
        super(Identity, self).__init__(*a, **kw)

    def forward(self, *x):
        if len(x) == 1:
            x = x[0]
        return x


class LayerNormalization(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, dim, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(dim), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out