import torch, qelos as q
from torch import nn
import numpy as np


class SeqNLLLoss(nn.NLLLoss):
    def __init__(self, weight=None, size_average=True, time_average=True, ignore_index=0):
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
            mask = (y != self.ignore_index).float()      # ByteTensor
        # mask = mask.type(torch.FloatTensor)
        logprobs = -torch.gather(x, 1, y.unsqueeze(1)).squeeze()
        if self.weight is not None:
            weights = self.weight[y]
            logprobs = logprobs * weights
        if mask is not None:
            logprobs = logprobs * mask
            mask = mask.view(batsize, seqlen)
        logprobs = logprobs.view(batsize, seqlen)
        if mask is not None:
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
            mask = (gold == self.ignore_index)
        maxes, argmaxes = torch.max(probs, dim=2)
        diff = argmaxes != gold
        if mask is not None:
            diff = diff + mask
            total = torch.sum((mask == 0).long()).data[0]
        else:
            total = gold.size(0) * gold.size(1)
        acc = torch.sum((diff == 0).float())
        if self.size_average:
            acc = acc / total
        return acc, total



