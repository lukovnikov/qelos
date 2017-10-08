import torch
from torch import nn
import qelos as q
from qelos.rnn import GRUCell, Recurrent, Reccable, RNUBase
from qelos.qutils import name2fn
from qelos.basic import Forward


class MemGRUCell(GRUCell):
    """ Very simple memory-augmented GRU """
    def __init__(self, indim, outdim, memsize=5, use_bias=True,
                 dropout_in=None, dropout_rec=None, zoneout=None,
                 shared_dropout_rec=None, shared_zoneout=None,
                 use_cudnn_cell=True, activation="tanh",
                 gate_activation="sigmoid", rec_batch_norm=None):
        super(MemGRUCell, self).__init__(indim, outdim * 2, use_bias=use_bias,
                dropout_in=dropout_in, dropout_rec=dropout_rec,
                zoneout=zoneout, shared_dropout_rec=shared_dropout_rec,
                shared_zoneout=shared_zoneout, use_cudnn_cell=use_cudnn_cell,
                activation=activation, gate_activation=gate_activation,
                rec_batch_norm=rec_batch_norm)

        self.memsize = memsize
        self.selector = nn.Linear(self.outdim, self.memsize)
        self.realoutdim = outdim

    @property
    def state_spec(self):
        return self.realoutdim * (self.memsize + 2),

    def _forward(self, x_t, M_tm1, t=None):
        # unpack
        M_tm1 = M_tm1.contiguous().view(x_t.size(0), self.memsize + 2, self.realoutdim)
        c_tm1 = M_tm1[:, 0, :]
        m_tm1 = M_tm1[:, 1, :]
        M_tm1 = M_tm1[:, 2:, :]     # (batsize, memsize, dim)

        h_tm1 = torch.cat([c_tm1, m_tm1], 1)

        # select and read
        select_t = self.selector(h_tm1)
        select_t = nn.Softmax()(select_t)   # (batsize, memsize)
        r_t = M_tm1 * select_t.unsqueeze(2)
        r_t = r_t.sum(1)

        h_tm1 = torch.cat([c_tm1, r_t], 1)

        # update
        y_t, h_t = super(MemGRUCell, self)._forward(x_t, h_tm1, t=t)

        c_t = h_t[:, :h_t.size(1) // 2]
        m_t = h_t[:, h_t.size(1) // 2:]     # (batsize, dim)

        # write
        mem_update = m_t.unsqueeze(1).repeat(1, self.memsize, 1)
        M_t = mem_update * select_t.unsqueeze(2) + M_tm1 * (1 - select_t.unsqueeze(2))

        # pack
        M_t = torch.cat([c_t.unsqueeze(1), m_t.unsqueeze(1), M_t], 1)
        M_t = M_t.view(x_t.size(0), -1)

        return c_t, M_t

