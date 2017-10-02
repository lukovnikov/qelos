import torch, qelos as q
from torch import nn
import numpy as np

EPS = 1e-6


class Loss(nn.Module):
    def __init__(self, size_average=True, _size_avg_ignore_mask=False, **kw):
        super(Loss, self).__init__(**kw)
        self.size_average = size_average
        self._size_avg_ignore_mask = _size_avg_ignore_mask

    def forward(self, x, gold, mask=None, _noagg=False):
        y, ignoremask = self._forward(x, gold, mask=mask)

        if _noagg:
            return y, ignoremask

        if ignoremask is not None:
            y = y * ignoremask.float()
        if ignoremask is not None and self._size_avg_ignore_mask:
            total = ignoremask.long().sum().data[0]
        else:
            total = y.size(0)

        loss = y.sum()
        if self.size_average:
            loss /= total
        return loss


class DiscreteLoss(Loss):
    def __init__(self, size_average=True, ignore_index=None, **kw):
        super(DiscreteLoss, self).__init__(size_average=size_average, **kw)
        if ignore_index is not None:
            if not q.issequence(ignore_index):
                self.ignore_indices = [ignore_index]
        else:
            self.ignore_indices = None

    def _get_ignore_mask(self, gold):
        mask = None     # (batsize,)
        if self.ignore_indices is not None:
            for ignore in self.ignore_indices:
                mask_i = (gold != ignore)   # zero for ignored ones
                if mask is None:
                    mask = mask_i
                else:
                    mask = mask * mask_i
        return mask


class SeqLoss(nn.Module):
    def __init__(self, time_agg="sum", **kw):
        super(SeqLoss, self).__init__(**kw)
        self.time_agg = time_agg

    def _forward(self, probs, gold, mask=None):     # (batsize, seqlen, dim), idx^(batsize, seqlen)
        batsize, seqlen, vocsize = probs.size()
        x = probs.view(batsize * seqlen, vocsize)
        y = gold.contiguous().view(batsize * seqlen)
        if mask is not None:
            mask = mask.view(batsize * seqlen, vocsize)

        l, ignoremask = super(SeqLoss, self)._forward(x, y, mask=mask)

        l = l.view(batsize, seqlen)

        outmask = None
        if ignoremask is not None:
            ignoremask = ignoremask.view(batsize, seqlen)
            outmask = ignoremask.long().sum(1) > 0
            totals = ignoremask.long().sum(1)
        else:
            totals = l.size(1)

        ltotal = l.sum(1)
        if self.time_agg == "sum":
            pass
        elif self.time_agg == "avg":
            totals = totals.float().clamp(min=EPS)
            ltotal = ltotal / totals
        elif self.time_agg == "eq":
            ltotal = (ltotal == totals).float()

        return ltotal, outmask


class NLLLoss(DiscreteLoss):
    def __init__(self, weight=None, size_average=True, ignore_index=None, **kw):
        super(NLLLoss, self).__init__(size_average=size_average, ignore_index=ignore_index, **kw)
        self.weight = weight

    def _forward(self, x, gold, mask=None):     # (batsize, vocsize)
        ignoremask = self._get_ignore_mask(gold)

        logprobs = -torch.gather(x, 1, gold.unsqueeze(1)).squeeze()

        if self.weight is not None:
            weights = self.weight[gold]
            logprobs = logprobs * weights

        if ignoremask is not None:
            logprobs = logprobs * ignoremask.float()

        return logprobs, ignoremask


class SeqNLLLoss(SeqLoss, NLLLoss):
    def __init__(self, size_average=True, time_average=False, weight=None, ignore_index=None, **kw):
        super(SeqNLLLoss, self).__init__(size_average=size_average, time_agg="avg" if time_average else "sum",
                                         weight=weight, ignore_index=ignore_index, **kw)


class CrossEntropyLoss(NLLLoss):
    def __init__(self, weight=None, size_average=True, ignore_index=None, temperature=1., **kw):
        super(CrossEntropyLoss, self).__init__(size_average=size_average, ignore_index=ignore_index, weight=weight, **kw)
        self.softmax = q.LogSoftmax(temperature=temperature)

    def _forward(self, scores, gold, mask=None):
        probs = self.softmax(scores, mask=mask)
        logprobs, ignoremask = super(CrossEntropyLoss, self)._forward(probs, gold, mask=mask)
        return logprobs, ignoremask


class SeqCrossEntropyLoss(SeqLoss, CrossEntropyLoss):
    def __init__(self, size_average=True, time_average=False, weight=None, ignore_index=None, temperature=1., **kw):
        super(SeqCrossEntropyLoss, self).__init__(size_average=size_average, time_agg="avg" if time_average else "sum",
                                         weight=weight, ignore_index=ignore_index, temperature=temperature, **kw)


class RankingLoss(DiscreteLoss):
    def __init__(self, size_average=True, ignore_index=None,
                 negbest=False,
                 ignore_minimum=False,
                 margin=None, ignore_below_margin=True, **kw):
        super(RankingLoss, self).__init__(size_average=size_average, ignore_index=ignore_index,
                                          **kw)
        self.ignore_minimum = ignore_minimum
        self.margin = margin
        self.ignore_below_margin = ignore_below_margin
        self.negmode = "best" if negbest else "random"      # "random" or "best"

    def _forward(self, scores, gold, mask=None):    # (batsize, numvoc), idx^(batsize,)
        scores = scores - scores.min()
        goldscores = torch.gather(scores, 1, gold.unsqueeze(1)).squeeze()

        if self.negmode == "random":
            sampledist = scores.data.new(scores.size())
            sampledist.fill_(1.)
            sampledist.scatter_(1, gold.data.unsqueeze(1), 0)
            if mask is not None:
                sampledist = sampledist * mask.float()
            sampledist_orig = sampledist
            if self.margin is not None and self.ignore_below_margin:
                cutoffs = goldscores.data - self.margin
                cutoffmask = scores.data > cutoffs.unsqueeze(1)
                sampledist = sampledist * cutoffmask.float()
            if self.ignore_minimum:
                minmask = scores.data > scores.min().data
                sampledist = sampledist * minmask.float()
            if (sampledist.sum(1) > 0).long().sum() < gold.size(0):
                # force to sample gold
                gold_onehot = sampledist.new(sampledist.size())
                gold_onehot.fill_(0.)
                gold_onehot.scatter_(1, gold.data.unsqueeze(1), 1.)
                examplemask = (sampledist.sum(1) == 0)
                # addtosampledist = sampledist_orig * examplemask.float().unsqueeze(1)
                addtosampledist = gold_onehot * examplemask.float().unsqueeze(1)
                sampledist = sampledist + addtosampledist
            sample = torch.multinomial(sampledist, 1)
            sample = q.var(sample).cuda(scores).v
            negscores = torch.gather(scores, 1, sample).squeeze()
        elif self.negmode == "best":
            scores = scores * mask.float() if mask else scores
            bestscores, best = torch.max(scores, 1)
            secondscores = scores + 0
            secondscores.scatter_(1, best.unsqueeze(1), 0)
            secondbestscores, secondbest = torch.max(secondscores, 1)
            switchmask = best == gold
            sample = secondbest * switchmask.long() + best * (1 + (-1) * switchmask.long())
            negscores = secondbestscores * switchmask.float() + bestscores * (1 - switchmask.float())
            # raise NotImplemented("some issues regarding implementation not resolved")
        else:
            raise q.SumTingWongException("unknown mode: {}".format(self.negmode))

        loss = negscores - goldscores
        if self.margin is not None:
            loss = torch.clamp(self.margin + loss, min=0)

        examplemask = sample.squeeze() != gold
        loss = examplemask.float() * loss

        ignoremask = self._get_ignore_mask(gold)
        if ignoremask is not None:
            loss = loss * ignoremask.float()

        return loss, ignoremask


class SeqRankingLoss(SeqLoss, RankingLoss):
    def __init__(self, size_average=True, time_average=False,
                 ignore_index=None,
                 negbest=False,
                 ignore_minimum=False,
                 margin=None, ignore_below_margin=True, **kw):
        super(SeqRankingLoss, self).__init__(size_average=size_average,
                                             time_agg="avg" if time_average else "sum",
                                             ignore_index=ignore_index,
                                             negbest=negbest,
                                             ignore_minimum=ignore_minimum,
                                             margin=margin,
                                             ignore_below_margin=ignore_below_margin,
                                             **kw)


class Accuracy(DiscreteLoss):
    def _forward(self, x, gold, mask=None):
        x = x * mask.float() if mask is not None else x
        ignoremask = self._get_ignore_mask(gold)
        maxes, best = torch.max(x, 1)
        same = best == gold
        if ignoremask is not None:
            same = same * ignoremask
        return same.long(), ignoremask


class SeqAccuracy(SeqLoss, Accuracy):
    def __init__(self, size_average=True, ignore_index=None):
        super(SeqAccuracy, self).__init__(size_average=size_average,
                                          ignore_index=ignore_index,
                                          time_agg="eq")


class SeqElemAccuracy(DiscreteLoss):
    def forward(self, probs, gold, mask=None):
        ignoremask = self._get_ignore_mask(gold)
        maxes, argmaxes = torch.max(probs, dim=2)
        diff = argmaxes == gold
        if ignoremask is not None:
            diff = diff * ignoremask
            total = torch.sum(ignoremask.long()).data[0]
        else:
            total = gold.size(0) * gold.size(1)
        acc = torch.sum(diff.float())
        if self.size_average:
            acc = acc / total
        return acc, total


class OldSeqNLLLoss(nn.NLLLoss):
    def __init__(self, weight=None, size_average=True, time_average=True, ignore_index=0):
        if ignore_index is not None:
            if not q.issequence(ignore_index):
                ignore_index = [ignore_index]
        else:
            ignore_index = None
        super(SeqNLLLoss, self).__init__(weight=weight, size_average=size_average, ignore_index=ignore_index)
        self.time_average = time_average

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
            totals = mask.sum(1).clamp(min=EPS)
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


class OldSeqAccuracy(nn.Module):
    def __init__(self, size_average=True, ignore_index=0):
        super(OldSeqAccuracy, self).__init__()
        self.size_average = size_average
        if ignore_index is not None:
            if not q.issequence(ignore_index):
                ignore_index = [ignore_index]
            self.ignore_index = ignore_index
        else:
            self.ignore_index = None
        self.EPS = 1e-6

    def forward(self, probs, gold, mask=None):     # (batsize, seqlen, vocsize), (batsize, seqlen)-idx
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




