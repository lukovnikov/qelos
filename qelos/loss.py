import torch, qelos as q
from torch import nn
import numpy as np

EPS = 1e-6


class Loss(nn.Module):
    def __init__(self, size_average=True, _size_avg_ignore_mask=False, **kw):
        super(Loss, self).__init__(**kw)
        self.size_average = size_average
        self._size_avg_ignore_mask = _size_avg_ignore_mask

    def forward(self, x, gold, mask=None, _noagg=False, **kw):
        y, ignoremask = self._forward(x, gold, mask=mask, **kw)
        y = y.float()
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


class PairRankingLoss(Loss):
    def __init__(self, size_average=True, margin=None, scale=1., **kw):
        super(PairRankingLoss, self).__init__(size_average=size_average, **kw)
        self.margin = margin
        self.scale = scale

    def _forward(self, x, gold, **kw):
        """ x is the difference in scores. optionally, gold is margin
            if x.dim() == 1, assuming margin loss
            if x.dim() == 2, assuming hinge loss of the two separately
        """
        if self.margin is None:     # use gold as margins
            margin = gold
        else:
            margin = self.margin

        zeros = q.var(torch.zeros(x.size(0))).cuda(x).v
        if x.dim() == 1:
            loss = torch.max(zeros, margin * self.scale - x)
        elif x.dim() == 2:
            assert(x.size(1) == 2)
            lossA = torch.max(zeros, margin * self.scale - x[:, 0])
            lossB = torch.max(zeros, margin * self.scale + x[:, 1])
            loss = lossA + lossB
        return loss, None


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
                    mask = mask & mask_i
        return mask


class SeqLoss(nn.Module):       # TODO: take end of sequence token into account
    def __init__(self, time_agg="sum", **kw):
        super(SeqLoss, self).__init__(**kw)
        self.time_agg = time_agg

    def _forward(self, probs, gold, mask=None):     # (batsize, seqlen, dim), idx^(batsize, seqlen)
        if probs.size(1) > gold.size(1):
            probs = probs[:, :gold.size(1)]
        batsize, seqlen, vocsize = probs.size()
        x = probs.contiguous().view(batsize * seqlen, vocsize)
        try:
            y = gold.contiguous().view(batsize * seqlen)
        except Exception as e:
            print(batsize, seqlen, gold.size())
        if mask is not None:
            mask = mask.contiguous().view(batsize * seqlen, -1)

        l, ignoremask = super(SeqLoss, self)._forward(x, y, mask=mask)

        l = l.view(batsize, seqlen)

        outmask = None
        if ignoremask is not None:
            ignoremask = ignoremask.view(batsize, seqlen)
            outmask = ignoremask.long().sum(1) > 0
            totals = ignoremask.float().sum(1)
        else:
            totals = q.var(torch.FloatTensor(l.size(0))).cuda(l).v
            totals.data.fill_(l.size(1))

        if self.time_agg == "sum":
            ltotal = l.float().sum(1)
        elif self.time_agg == "avg":
            ltotal = l.float().sum(1)
            totals = totals.clamp(min=EPS)
            ltotal = ltotal / totals
        elif self.time_agg == "all":
            if ignoremask is not None:
                l = torch.autograd.Variable(l.byte().data | ~ ignoremask.data)
            ltotal = l.float().sum(1)
            ltotal = ltotal == l.size(1)
        elif self.time_agg == "eqtotal":
            ltotal = l.float().sum(1)
            print("DEPRECATED for 'all'")
            ltotal = (ltotal == totals)
        elif self.time_agg == "allone":
            ltotal = l.float().sum(1)
            print("DEPRECATED for 'all'")
            ltotal = (ltotal == l.size(1))

        return ltotal, outmask


class NLLLoss(DiscreteLoss):
    def __init__(self, weight=None, size_average=True, ignore_index=None, **kw):
        super(NLLLoss, self).__init__(size_average=size_average, ignore_index=ignore_index, **kw)
        self.weight = weight

    def _forward(self, x, gold, mask=None):     # (batsize, vocsize)
        # probs for masked elements must have been zero by softmax
        ignoremask = self._get_ignore_mask(gold)

        if mask is not None:
            x = x + torch.log(mask.float())

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
        # softmax zeroes/mininfinites the masked symbols
        probs = self.softmax(scores, mask=mask)
        if isinstance(probs, tuple):
            probs = probs[0]
        logprobs, ignoremask = super(CrossEntropyLoss, self)._forward(probs, gold, mask=mask)
        return logprobs, ignoremask


class SeqCrossEntropyLoss(SeqLoss, CrossEntropyLoss):
    def __init__(self, size_average=True, time_average=False, weight=None, ignore_index=None, temperature=1., **kw):
        super(SeqCrossEntropyLoss, self).__init__(size_average=size_average, time_agg="avg" if time_average else "sum",
                                         weight=weight, ignore_index=ignore_index, temperature=temperature, **kw)


class RankingLoss(DiscreteLoss):
    def __init__(self, size_average=True, ignore_index=None,
                 negmode="random",
                 margin=None, ignore_below_margin=True, **kw):
        super(RankingLoss, self).__init__(size_average=size_average, ignore_index=ignore_index,
                                          **kw)
        self.margin = margin
        self.ignore_below_margin = ignore_below_margin
        self.negmode = negmode      # "random" or "best" or "negall"
        self._average_negall = True

    def _forward(self, scores, gold, mask=None):    # (batsize, numvoc), idx^(batsize,)
        # scores = scores - scores.min()
        goldscores = torch.gather(scores, 1, gold.unsqueeze(1)).squeeze()

        if mask is not None and mask.data[0, 1] > 1:
            mask = q.batchablesparse2densemask(mask)

        goldexamplemask = None

        if self.negmode == "random" or self.negmode == "negall":
            sampledist = q.var(scores.data.new(scores.size())).cuda(scores).v
            sampledist.data.fill_(1.)
            sampledist.data.scatter_(1, gold.data.unsqueeze(1), 0)
            filtermask = scores > -np.infty
            if mask is not None:
                filtermask.data = filtermask.data & mask.byte().data
            sampledist = sampledist * filtermask.float()
            sampledist_orig = sampledist
            if self.margin is not None and self.ignore_below_margin:
                cutoffs = goldscores - self.margin
                cutoffmask = scores > cutoffs.unsqueeze(1)
                sampledist = sampledist * cutoffmask.float()
            if (sampledist.data.sum(1) > 0).long().sum() < gold.size(0):
                # force to sample gold
                gold_onehot = q.var(torch.ByteTensor(sampledist.size())).cuda(sampledist).v
                gold_onehot.data.fill_(0)
                gold_onehot.data.scatter_(1, gold.data.unsqueeze(1), 1)
                goldexamplemask = (sampledist.sum(1) != 0)
                # addtosampledist = sampledist_orig * examplemask.float().unsqueeze(1)
                addtosampledist = gold_onehot.data * (~goldexamplemask.data).unsqueeze(1)
                sampledist.data.masked_fill_(addtosampledist, 1)
            if self.negmode == "random":
                sample = torch.multinomial(sampledist, 1)
                negscores = torch.gather(scores, 1, sample).squeeze()
            elif self.negmode == "negall":
                negscores = scores * sampledist
                numnegs = sampledist.sum(1)
        elif self.negmode == "best":
            # scores = scores * mask.float() if mask else scores
            scores = scores + torch.log(mask.float()) if mask else scores
            bestscores, best = torch.max(scores, 1)
            secondscores = scores + 0
            secondscores.data.scatter_(1, best.data.unsqueeze(1), 0)
            secondbestscores, secondbest = torch.max(secondscores, 1)
            switchmask = best == gold
            sample = secondbest * switchmask.long() + best * (1 + (-1) * switchmask.long())
            negscores = secondbestscores * switchmask.float() + bestscores * (1 - switchmask.float())
            goldexamplemask = sample.squeeze() != gold
            # raise NotImplemented("some issues regarding implementation not resolved")
        else:
            raise q.SumTingWongException("unknown mode: {}".format(self.negmode))

        if self.negmode == "best" or self.negmode == "random":
            loss = negscores - goldscores
            if self.margin is not None:
                loss = torch.clamp(self.margin + loss, min=0)
            if goldexamplemask is not None:
                loss = goldexamplemask.float() * loss
        elif self.negmode == "negall":
            # negscores are 2D
            loss = negscores - goldscores.unsqueeze(1)
            if self.margin is not None:
                loss = torch.clamp(self.margin + loss, min=0)
            loss = loss * sampledist
            loss = loss.sum(1)
            if self._average_negall:
                loss = loss / numnegs
            if goldexamplemask is not None:
                loss = loss * goldexamplemask.float()

        ignoremask = self._get_ignore_mask(gold)
        if ignoremask is not None:
            loss = loss * ignoremask.float()

        return loss, ignoremask


class SeqRankingLoss(SeqLoss, RankingLoss):
    def __init__(self, size_average=True, time_average=False,
                 ignore_index=None, negmode="random",
                 margin=None, ignore_below_margin=True, **kw):
        super(SeqRankingLoss, self).__init__(size_average=size_average,
                                             time_agg="avg" if time_average else "sum",
                                             ignore_index=ignore_index,
                                             negmode=negmode,
                                             margin=margin,
                                             ignore_below_margin=ignore_below_margin,
                                             **kw)


class Accuracy(DiscreteLoss):
    def _forward(self, x, gold, mask=None):
        if mask is not None and mask.data[0, 1] > 1:     # batchable sparse
            mask = q.batchablesparse2densemask(mask)
        if mask is not None:
            x = x + torch.log(mask.float())
        ignoremask = self._get_ignore_mask(gold)
        maxes, best = torch.max(x, 1)
        same = best == gold
        if ignoremask is not None:
            same.data = same.data | ~ ignoremask.data
        return same.float(), ignoremask


class SeqAccuracy(SeqLoss, Accuracy):
    def __init__(self, size_average=True, ignore_index=None):
        super(SeqAccuracy, self).__init__(size_average=size_average,
                                          ignore_index=ignore_index,
                                          time_agg="all")


class SeqElemAccuracy(DiscreteLoss):    # TODO take end of sequence token into account
    def forward(self, x, gold, mask=None):
        if x.size(1) > gold.size(1):
            x = x[:, :gold.size(1)]
        if mask is not None and mask.data[0, 0, 1] > 1:     # batchable sparse
            mask = q.batchablesparse2densemask(mask)
        if mask is not None:
            x = x + torch.log(mask.float())
        ignoremask = self._get_ignore_mask(gold)
        maxes, argmaxes = torch.max(x, dim=2)
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


from nltk.translate.bleu_score import sentence_bleu
import warnings


class MacroBLEU(DiscreteLoss):      # TODO take end of sequence token into account
    """ macro-averaged BLEU over sequences """
    def __init__(self, order=4, predcut=None, ignore_index=None, **kw):
        """
        :param order:           n-gram order of BLEU
        :param predcut:         function to cut prediction. Gets the argmax over prediction and ignore_index kwarg.
                                Must fill all elements after end of sequence with provided ignore_index
        """
        super(MacroBLEU, self).__init__(ignore_index=ignore_index, **kw)
        self.order = order
        self.weights = tuple([1. / self.order for _ in range(self.order)])
        self.predcut = predcut
        warnings.filterwarnings("ignore", module="nltk")

    def forward(self, x, gold, mask=None):
        if x.size(1) > gold.size(1):
            x = x[:, :gold.size(1)]
        if mask is not None and mask.data[0, 0, 1] > 1:     # batchable sparse
            mask = q.batchablesparse2densemask(mask)
        if mask is not None:
            x = x + torch.log(mask.float())
        ignoremask = self._get_ignore_mask(gold)
        maxes, argmaxes = torch.max(x, dim=2)
        ignore_id = None
        if self.ignore_indices is not None:
            ignore_id = self.ignore_indices[0]
        argmaxes = argmaxes.data.cpu()
        if self.predcut is not None:
            argmaxes = self.predcut(argmaxes, ignore_index=ignore_id)
        gold = gold.data.cpu()
        bleus = 0.
        for i in range(gold.size(0)):
            predseq = [str(a) for a in list(argmaxes[i]) if a != ignore_id]
            goldseq = [str(a) for a in list(gold[i]) if a not in self.ignore_indices]
            bleu = sentence_bleu([goldseq], predseq, weights=self.weights)
            bleus += bleu

        total = gold.size(0)
        if self.size_average:
            bleus = bleus / total
        return bleus, total


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




