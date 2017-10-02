import torch, qelos as q
from torch import nn
import numpy as np

EPS = 1e-6


class SeqNLLLoss(nn.NLLLoss):
    def __init__(self, weight=None, size_average=True, time_average=True, ignore_index=0):
        if ignore_index is not None:
            if not q.issequence(ignore_index):
                ignore_index = [ignore_index]
        else:
            ignore_index = None
        super(SeqNLLLoss, self).__init__(weight=weight, size_average=size_average, ignore_index=ignore_index)
        self.time_average = time_average
        self.EPS = 1e-6

    def forward(self, probs, gold):
        """
        :param probs: (batsize, seqlen, vocsize) log-probabilities for each timestep
        :param gold:  (batsize, seqlen) correct values for each timestep
        :return:
        """
        batsize, seqlen, vocsize = probs.size()
        x = probs.view(batsize * seqlen, vocsize)
        y = gold.contiguous().view(batsize * seqlen)
        mask = None
        if self.ignore_index is not None:
            for ignore in self.ignore_index:
                mask_i = (y != ignore)   # zero for ignored ones
                if mask is None:
                    mask = mask_i
                else:
                    mask = mask * mask_i
            mask = mask.float()

        logprobs = -torch.gather(x, 1, y.unsqueeze(1)).squeeze()
        if self.weight is not None:
            weights = self.weight[y]
            logprobs = logprobs * weights

        if mask is not None:
            logprobs = logprobs * mask

        logprobs = logprobs.view(batsize, seqlen)
        if mask is not None:
            mask = mask.view(batsize, seqlen)
            totals = mask.sum(1).clamp(min=self.EPS)
        else:
            totals = logprobs.size(1)
        logprobsum = logprobs.sum(1)
        if self.time_average:
            logprobsum = logprobsum / totals
        t = logprobsum.size(0)
        loss = logprobsum.sum()
        if self.size_average:
            loss /= t
        return loss


class SeqAccuracy(nn.Module):       # TODO test
    def __init__(self, size_average=True, ignore_index=0):
        super(SeqAccuracy, self).__init__()
        self.size_average = size_average
        if ignore_index is not None:
            if not q.issequence(ignore_index):
                ignore_index = [ignore_index]
            self.ignore_index = ignore_index
        else:
            self.ignore_index = None
        self.EPS = 1e-6

    def forward(self, probs, gold):     # (batsize, seqlen, vocsize), (batsize, seqlen)-idx
        mask = None
        if self.ignore_index is not None:
            for ignore in self.ignore_index:
                mask_i = (gold != ignore)   # zero for ignored ones
                if mask is None:
                    mask = mask_i
                else:
                    mask = mask * mask_i
        maxes, argmaxes = torch.max(probs, dim=2)
        diff = argmaxes != gold
        if mask is not None:
            diff = diff * mask
        diffsums = torch.sum(diff.long(), dim=1)
        total = gold.size(0)
        acc = torch.sum((diffsums == 0).long()).float()
        if self.size_average:
            acc = acc / total
        return acc


class SeqElemAccuracy(SeqAccuracy):     # TODO: test
    def forward(self, probs, gold):
        mask = None
        if self.ignore_index is not None:
            for ignore in self.ignore_index:
                mask_i = (gold != ignore)  # zero for ignored ones
                if mask is None:
                    mask = mask_i
                else:
                    mask = mask * mask_i
        maxes, argmaxes = torch.max(probs, dim=2)
        diff = argmaxes == gold
        if mask is not None:
            diff = diff * mask
            total = torch.sum(mask.long()).data[0]
        else:
            total = gold.size(0) * gold.size(1)
        acc = torch.sum(diff.float())
        if self.size_average:
            acc = acc / total
        return acc, total


class RankingLoss(nn.Module):
    def __init__(self, size_average=True, ignore_index=0,
                 negbest=False,
                 ignore_minimum=False,
                 margin=None, ignore_below_margin=True, **kw):
        super(RankingLoss, self).__init__(**kw)
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.ignore_minimum = ignore_minimum
        self.margin = margin
        self.ignore_below_margin = ignore_below_margin
        self.negmode = "best" if negbest else "random"      # "random" or "best"

    def forward(self, scores, gold, _noagg=False):    # (batsize, numvoc), idx^(batsize,)
        scores = scores - scores.min()
        goldscores = torch.gather(scores, 1, gold.unsqueeze(1)).squeeze()

        if self.negmode == "random":
            sampledist = scores.data.new(scores.size())
            sampledist.fill_(1.)
            sampledist.scatter_(1, gold.data.unsqueeze(1), 0)
            sampledist_orig = sampledist
            if self.margin is not None and self.ignore_below_margin:
                cutoffs = goldscores.data - self.margin
                cutoffmask = scores.data > cutoffs.unsqueeze(1)
                sampledist = sampledist * cutoffmask.float()
            if self.ignore_minimum:
                minmask = scores.data == scores.min().data
                sampledist = sampledist * minmask.float()
            if (sampledist.sum(1) > 0).long().sum() < gold.size(0):
                examplemask = (sampledist.sum(1) == 0)
                addtosampledist = sampledist_orig * examplemask.float().unsqueeze(1)
                sampledist = sampledist + addtosampledist
            sample = torch.multinomial(sampledist, 1)
            sample = q.var(sample).cuda(scores).v
            negscores = torch.gather(scores, 1, sample).squeeze()
        elif self.negmode == "best":
            bestscores, best = torch.max(scores, 1)
            secondscores = scores + 0
            secondscores.scatter_(1, best.unsqueeze(1), 0)
            secondbestscores, secondbest = torch.max(secondscores, 1)
            switchmask = best == gold
            negscores = secondbestscores * switchmask.float() + bestscores * (1 - switchmask.float())
        else:
            raise q.SumTingWongException("unknown mode: {}".format(self.negmode))

        loss = negscores - goldscores
        if self.margin is not None:
            loss = torch.clamp(self.margin + loss, min=0)

        mask = None
        if self.ignore_index is not None:
            mask = gold != self.ignore_index
            loss = loss * mask.float()

        if _noagg:
            return loss, mask

        totalloss = loss.sum()
        if self.size_average:
            totalnumber = loss.size(0)
            totalloss = totalloss / totalnumber
        return totalloss


class SeqLoss(nn.Module):
    def __init__(self, time_average=False, **kw):
        super(SeqLoss, self).__init__(**kw)
        self.time_average = time_average

    def forward(self, probs, gold):     # (batsize, seqlen, dim), idx^(batsize, seqlen)
        batsize, seqlen, vocsize = probs.size()
        x = probs.view(batsize * seqlen, vocsize)
        y = gold.contiguous().view(batsize * seqlen)

        l, mask = super(SeqLoss, self).forward(x, y, _noagg=True)

        l = l.view(batsize, seqlen)
        if mask is not None:
            mask = mask.view(batsize, seqlen)
            totals = mask.float().sum(1).clamp(min=EPS)
        else:
            totals = l.size(1)
        ltotal = l.sum(1)
        if self.time_average:
            ltotal = ltotal / totals
        t = ltotal.size(0)
        loss = ltotal.sum()
        if self.size_average:
            loss /= t
        return loss


class SeqRankingLoss(SeqLoss, RankingLoss):
    pass


