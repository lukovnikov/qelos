import torch
from torch import nn
from qelos.rnn import Recurrent


class SeqConv(nn.Module, Recurrent):
    def __init__(self, indim, outdim, window=2, bias=True):
        super(SeqConv, self).__init__()
        self.conv = nn.Conv1d(indim, outdim, window*2+1, padding=window, bias=bias)

    def forward(self, x, mask=None):
        if mask is not None:
            xmask = mask.unsqueeze(2).float()
            x = x * xmask
        y = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        if mask is not None:
            y = y * xmask
        return y


