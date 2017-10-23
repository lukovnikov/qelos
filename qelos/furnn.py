import torch
from torch import nn
from qelos.rnn import GRUCell, Recurrent, Reccable, RNUBase
from qelos.qutils import name2fn
from qelos.basic import Forward
import qelos as q
import numpy as np


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


class maxfwd_sumbwd(torch.autograd.Function):
    def __init__(self, axis, **kw):
        super(maxfwd_sumbwd, self).__init__(**kw)
        self.axis = axis

    def forward(self, input):
        self.save_for_backward(input)
        ret = torch.max(input, self.axis)
        return ret[0]

    def backward(self, gradout):
        input, = self.saved_tensors
        gradinp = gradout.clone()
        repeats = [1] * input.dim()
        repeats[self.axis] = input.size(self.axis)
        gradinp = gradinp.unsqueeze(self.axis).repeat(*repeats)
        return gradinp


class SimpleDGTN(nn.Module):
    """ Single-hop hop-only DGTN """

    def __init__(self, tensor, indim, **kw):
        super(SimpleDGTN, self).__init__(**kw)
        self.tensor = q.val(tensor).v
        self.outdim = self.tensor.size(0)
        self.numents = self.tensor.size(1)
        self.linout = nn.Linear(indim, self.outdim, bias=False)
        self.softmax = q.Softmax()

    def forward(self, ids, travvec):  # (batsize, dim)
        # select by id
        tensorslice = self.tensor.index_select(1, ids)
        tensorslice = tensorslice.transpose(1, 0)
        mask = tensorslice.sum(2) > 0
        y = self.linout(travvec)
        y, _ = self.softmax(y, mask=mask)
        # y = self.softmax(y)
        z = tensorslice.float() * y.unsqueeze(2)
        z, _ = z.max(1)
        # z = maxfwd_sumbwd(1)(z)
        return z
        # q.embed()
        # # summarize tensor
        # flattensor = self.tensor.view(self.outdim, -1)
        # z = torch.mm(y, flattensor.float())
        # sumtensor = z.view(z.size(0), self.numents, self.numents)
        # # traverse
        # outptr = torch.bmm(ptr.unsqueeze(1), sumtensor).squeeze(1)
        # return outptr


class SimpleDGTNDenseStart(nn.Module):
    """ Single-hop hop-only DGTN """

    def __init__(self, tensor, indim, **kw):
        super(SimpleDGTNDenseStart, self).__init__(**kw)
        self.tensor = q.val(tensor).v
        self.numrels = self.tensor.size(0)
        self.numents = self.tensor.size(1)
        self.linout = nn.Linear(indim, self.numrels, bias=False)
        self.softmax = q.Softmax()

    def forward(self, ptr, travvec):  # (batsize, dim)
        # ptr: (batsize, numents), tensor: (numrels, numents, numents)
        # weigh tensor by ptr
        ttensor = self.tensor.unsqueeze(0)
        tptr = ptr.unsqueeze(1).unsqueeze(3)
        ptrweighted_tensor = ttensor.float() * tptr     # (batsize, numrels, numents, numents)
        # get relation mask
        relmask = ptrweighted_tensor.sum(3).sum(2) > 0  # (batsize, numrels)
        # weigh tensor by rel
        y = self.linout(travvec)
        y, _ = self.softmax(y, mask=relmask)
        relweights = y.unsqueeze(2).unsqueeze(3)    # (batsize, numrels, 1, 1)
        weighted_tensor = ptrweighted_tensor * relweights

        # weighted tensor: (batsize, numrels, numents, numents)
        z, _ = weighted_tensor.max(1)
        z, _ = z.max(1)     # (batsize, numents)
        return z


class SimpleDGTNSparse(nn.Module):
    """ Single-hop hop-only DGTN """

    def __init__(self, tensor, indim, **kw):
        super(SimpleDGTNSparse, self).__init__(**kw)
        relmask = np.sum(tensor, axis=2).astype("uint8")
        tensor = np.argwhere(tensor)
        tensorsort = tensor[:, 2].argsort()
        tensor = tensor[tensorsort]

        numobjectgroupbysplits = 3
        binmass = np.ceil(tensor.shape[0] * 1. / numobjectgroupbysplits)

        uniqueobjects, fanins = np.unique(tensor[:, 2], return_counts=True)

        # faninargsort = np.argsort(fanins)[::-1]
        # _, inversefaninargsort = np.unique(faninargsort, return_index=True)
        # sortedobjects = uniqueobjects[faninargsort]
        # sortedfanins = fanins[faninargsort]

        bins = []
        curmass = 0
        previ = 0
        faninbins = []
        for i in range(uniqueobjects.shape[0]):
            curmass += fanins[i]
            if curmass > binmass:
                bins.append(uniqueobjects[previ:i])
                faninbins.append(fanins[previ:i])
                curmass = 0
                previ = i
        bins.append(uniqueobjects[previ:])
        faninbins.append(fanins[previ:])

        self.groupbys = torch.nn.ParameterList()
        self.groupbymasks = torch.nn.ParameterList()

        acc = 0
        groupbys = []
        groupbymasks = []
        for bin, binfanin in zip(bins, faninbins):
            groupby = np.zeros((bin.shape[0], np.max(binfanin)), dtype="int64")
            groupbymask = np.zeros_like(groupby, dtype="uint8")
            for j in range(groupby.shape[0]):
                objfanin = binfanin[j]
                groupby[j, :objfanin] = np.arange(0, objfanin) + acc
                groupbymask[j, :objfanin] = 1
                acc += objfanin
            self.groupbys.append(q.val(groupby).v)
            self.groupbymasks.append(q.val(groupbymask).v)

        self.tensor = q.val(tensor).v

        self.relmask = q.val(relmask).v

        self.numrels = torch.max(self.tensor[:, 0]).data[0] + 1
        self.numents = torch.max(self.tensor[:, 1]).data[0] + 1
        self.linout = nn.Linear(indim, self.numrels, bias=False)
        self.softmax = q.Softmax()

    def forward(self, ptr, travvec):  # (batsize, dim)
        # ptr: (batsize, numents), tensor: (numtrip, 3) - (R, E, E)
        startweights = torch.index_select(ptr, 1, self.tensor[:, 1])

        # relmask
        relmask = ptr.unsqueeze(1) * self.relmask.unsqueeze(0).float()
        relmask = relmask.sum(2) > 0
        y = self.linout(travvec)
        y, _ = self.softmax(y, mask=relmask)
        # y = self.softmax(y)

        # y: (batsize, numrels)

        relweights = torch.index_select(y, 1, self.tensor[:, 0])

        prodweights = startweights * relweights

        # do maxes using bins
        tocat = []
        for i in range(len(self.groupbys)):
            groupby = self.groupbys[i]
            groupbymask = self.groupbymasks[i].unsqueeze(0).float()
            shep = groupby.size()
            groupby = groupby.view(-1)
            binprodweights = torch.index_select(prodweights, 1, groupby)
            # (batsize, numtrip)
            binprodweights = binprodweights.view((binprodweights.size(0),) + tuple(shep))
            binprodweights = binprodweights * groupbymask
            # tocate, _ = torch.max(binprodweights, 2)
            tocate = maxfwd_sumbwd(2)(binprodweights)
            tocat.append(tocate)

        z = torch.cat(tocat, 1)

        return z


class TwoStackCell(RNUBase):
    """ works with two or one single-state cells (GRU or CatLSTM) """
    def __init__(self, cell, init_fraternal_to_ancestor=True, **kw):
        super(TwoStackCell, self).__init__(**kw)
        self.ance_stacks = None
        self.frat_stacks = None   # per-example stacks?

        self.init_fraternal_to_ancestor = init_fraternal_to_ancestor

        if isinstance(cell, tuple):     # dual cell
            assert(len(cell) == 2)
            cells = cell
        else:
            cells = (cell,)
        self.cells = q.ModuleList(list(cells))

    @property
    def state_spec(self):
        if len(self.cells) == 1:
            return self.cells[0].outdim // 2, self.cells[0].outdim // 2
        else:
            return self.cells[0].outdim, self.cells[1].outdim

    def reset_state(self):
        super(TwoStackCell, self).reset_state()
        for cell in self.cells:
            cell.reset_state()
        self.ance_stacks = None
        self.frat_stacks = None

    def set_stack_states(self, anc_state, frat_state):    # push states
        l = torch.split(anc_state, 1, 0)
        if self.ance_stacks is None:
            self.ance_stacks = tuple([[le] for le in l])
        else:
            pass        # running state mgmt already done in forward
            raise q.SumTingWongException("states already set")
            # for le, ance_stack in zip(l, self.ance_stacks):
            #     ance_stack.append(le)
        l = torch.split(frat_state, 1, 0)
        if self.frat_stacks is None:
            self.frat_stacks = tuple([[le] for le in l])
        else:
            pass        # running state mgmt already done in forward
            # for le, frat_stack in zip(l, self.frat_stacks):
            #     frat_stack.append(le)

    def init_stacks_from_cells(self, batsize):
        if len(self.cells) == 2:
            ance_state = self.cells[0].get_states(batsize)[0]
            frat_state = self.cells[1].get_states(batsize)[0]
        else:
            all_state = self.cells[0].get_states(batsize)[0]
            ac, fc, ay, fy = torch.chunk(all_state, 4, 1)
            ance_state, frat_state = torch.cat([ac, ay], 1), torch.cat([fc, fy], 1)
        if self.init_fraternal_to_ancestor:
            frat_state = ance_state
        self.set_stack_states(ance_state, frat_state)

    def get_stack_states(self):
        l = [ance_stack[-1] for ance_stack in self.ance_stacks]
        ance_state = torch.cat(l, 0)
        l = [frat_stack[-1] for frat_stack in self.frat_stacks]
        frat_state = torch.cat(l, 0)
        return ance_state, frat_state

    def _forward(self, x_t, ha_tm1, hf_tm1, t=None, ctrl_tm1=None, **kw):
        batsize = x_t.size(0)
        if self.ance_stacks is None:
            self.init_stacks_from_cells(batsize)
            ha_tm1, hf_tm1 = self.get_stack_states()
            # if TwoStackCell wasn't initialized, use (auto-)init from cell(s)

        ance_stacks_top = torch.split(ha_tm1, 1, 0)
        frat_stacks_top = torch.split(hf_tm1, 1, 0)

        # depending on ctrl action decided in previous time step, manage stacks
        for i in range(ctrl_tm1.size(0)):       # ctrl_tm1 must be (batsize,) int
            ctrl = ctrl_tm1.data[i]
            if ctrl == 0 or ctrl == 4:               # INIT / TERM
                pass
            elif ctrl == 1:               # GO DOWN
                # update frat
                self.frat_stacks[i][-1] = frat_stacks_top[i]
                # push zero frat
                if not self.init_fraternal_to_ancestor:
                    zerofrat = q.var(torch.zeros(frat_stacks_top[i].size())).cuda(x_t).v
                else:
                    zerofrat = ance_stacks_top[i]
                self.frat_stacks[i].append(zerofrat)
                # push anc
                self.ance_stacks[i].append(ance_stacks_top[i])
            elif ctrl == 2:               # GO RIGHT
                # update frat
                self.frat_stacks[i][-1] = frat_stacks_top[i]
                # keep anc
                pass
            elif ctrl == 3:               # GO UP
                # pop frat and anc
                self.ance_stacks[i].pop()
                self.frat_stacks[i].pop()

        # make an update
        stack_states = self.get_stack_states()
        if len(self.cells) == 2:
            ance_cell_out, ance_cell_state = self.cells[0]._forward(x_t, stack_states[0], t=t, **kw)
            frat_cell_out, frat_cell_state = self.cells[1]._forward(x_t, stack_states[1], t=t, **kw)
            cell_out = torch.cat([ance_cell_out, frat_cell_out], 1)
        else:
            ac, ay, fc, fy = torch.chunk(stack_states[0], 2, 1) + torch.chunk(stack_states[1], 2, 1)
            inp_states = torch.cat([ac, fc, ay, fy], 1)
            cell_out, cell_state = self.cells[0]._forward(x_t, inp_states, t=t, **kw)
            ac, fc, ay, fy = torch.chunk(cell_state, 4, 1)
            ance_cell_state, frat_cell_state = torch.cat([ac, ay], 1), torch.cat([fc, fy], 1)
        return cell_out, ance_cell_state, frat_cell_state






