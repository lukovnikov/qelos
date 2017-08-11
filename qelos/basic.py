import torch
from torch import nn
from torch.nn import functional as F


class Softmax(nn.Module):
    def __init__(self, temperature=1.):
        super(Softmax, self).__init__()
        self.temperature = temperature
        self._log = False
        self._logafter = False

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