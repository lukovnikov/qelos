import torch, qelos
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
        if self.ignore_index >= 0:
            mask = y.ne(self.ignore_index)      # ByteTensor
            mask = qelos.core.var(mask.data.type(torch.FloatTensor)).cuda(crit=mask).v
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
            totals = mask.sum(1)
        else:
            totals = logprobs.size(1)
        logprobsum = logprobs.sum(1)
        if self.time_average:
            logprobsum = logprobsum / (totals + self.EPS)
        t = logprobsum.size(0)
        loss = logprobsum.sum()
        if self.size_average:
            loss /= t
        return loss




