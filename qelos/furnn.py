import torch
from torch import nn
from qelos.rnn import GRUCell, Recurrent, Reccable, RNUBase, RecStateful, RecStatefulContainer
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


class TwoStackCell(RecStatefulContainer):
    """ works with two or one single-state cells (GRU or CatLSTM) """
    def __init__(self, emb, cell, init_fraternal_to_ancestor=True, **kw):
        super(TwoStackCell, self).__init__(**kw)
        # stacks are lists (per example) of tuples (per state) of vars (states)
        self.ance_stacks = None
        self.frat_stacks = None   # per-example stacks?
        self.ance_sym_stacks = None
        self.frat_sym_stacks = None
        self.frat_ctrl_stack = None        # per-example ctrl stacks

        self.init_fraternal_to_ancestor = init_fraternal_to_ancestor

        if isinstance(cell, tuple):     # dual cell, cell contains recstacks
            assert(len(cell) == 2)
            cells = cell
        else:
            cells = (cell,)

        self.cells = q.ModuleList(list(cells))

        if isinstance(emb, tuple):
            assert(len(emb) == 2)
            self.ancemb = emb[0]
            self.fratemb = emb[1]
        else:
            self.ancemb = emb
            self.fratemb = emb

        fratstartsym = self.fratemb.D["<START>"]
        self.y_f_0 = q.val(torch.LongTensor(1,)).v
        self.y_f_0.data.fill_(fratstartsym)

    @property
    def state_spec(self):
        if len(self.cells) == 1:
            return self.cells[0].outdim // 2, self.cells[0].outdim // 2
        else:
            return self.cells[0].outdim, self.cells[1].outdim

    def reset_state(self):
        for cell in self.cells:
            cell.reset_state()
        self.ance_stacks = None
        self.frat_stacks = None
        self.ctrl_stack = None

    def get_states_from_cells(self, batsize):
        if len(self.cells) == 2:
            ance_states = self.cells[0].get_states(batsize)
            frat_states = self.cells[1].get_states(batsize)
        else:
            ance_states, frat_states = [], []
            all_states = self.cells[0].get_states(batsize)
            for all_state in all_states:
                ac, fc, ay, fy = torch.chunk(all_state, 4, 1)
                ance_state, frat_state = torch.cat([ac, ay], 1), torch.cat([fc, fy], 1)
                ance_states.append(ance_state)
                frat_states.append(frat_state)
            ance_states, frat_states = tuple(ance_states), tuple(frat_states)
        return ance_states, frat_states

    def set_states_of_cells(self, ance_states, frat_states):
        if len(self.cells) == 2:
            self.cells[0].set_states(*ance_states)
            self.cells[1].set_states(*frat_states)
        else:
            inp_states = []
            for ance_state, frat_state in zip(ance_states, frat_states):
                ac, ay, fc, fy = torch.chunk(ance_state, 2, 1) + torch.chunk(frat_state, 2, 1)
                inp_state = torch.cat([ac, fc, ay, fy], 1)
                inp_states.append(inp_state)
            self.cells[0].set_states(*inp_states)

    def init_stacks_from_cells(self, batsize):
        ance_states, frat_states = self.get_states_from_cells(batsize)
        if self.init_fraternal_to_ancestor:
            frat_states = ance_states
        self.set_stack_states(ance_states, frat_states)

    @property
    def initialized(self):
        return self.ance_stacks is not None

    def init_all(self, batsize):
        self.ance_stacks = tuple([[] for _ in range(batsize)])
        self.frat_stacks = tuple([[] for _ in range(batsize)])
        self.ance_sym_stacks = tuple([[] for _ in range(batsize)])
        self.frat_sym_stacks = tuple([[] for _ in range(batsize)])
        self.frat_ctrl_stack = tuple([[] for _ in range(batsize)])

    def get_stack_states(self):
        ance_states = [ance_stack[-1] for ance_stack in self.ance_stacks]
        ance_states = zip(*ance_states)
        ance_states = [torch.cat(l, 0) for l in ance_states]
        frat_states = [frat_stack[-1] for frat_stack in self.frat_stacks]
        frat_states = zip(*frat_states)
        frat_states = [torch.cat(l, 0) for l in frat_states]
        return ance_states, frat_states

    def get_stack_syms(self):
        y_a_tm1 = [ance_sym_stack[-1] for ance_sym_stack in self.ance_sym_stacks]
        y_a_tm1 = torch.cat(y_a_tm1, 0)
        y_f_tm1 = [frat_sym_stack[-1] for frat_sym_stack in self.frat_sym_stacks]
        y_f_tm1 = torch.cat(y_f_tm1, 0)
        return y_a_tm1, y_f_tm1

    def set_stack_states(self, anc_states, frat_states):  # tuples of states
        anc_split_states = [torch.split(anc_state, 1, 0) for anc_state in anc_states]
        anc_split_states = zip(*anc_split_states)
        if self.ance_stacks is None:
            self.ance_stacks = tuple([[le] for le in anc_split_states])
        else:
            pass
            raise q.SumTingWongException("states already set")
        frat_split_states = [torch.split(frat_state, 1, 0) for frat_state in frat_states]
        frat_split_states = zip(*frat_split_states)
        if self.frat_stacks is None:
            self.frat_stacks = tuple([[le] for le in frat_split_states])
        else:
            pass

    def forward(self, y_tm1, ctrl_tm1, extravec_t=None, t=None, **kw):
        batsize = y_tm1.size(0)

        ha_tm1, hf_tm1 = self.get_states_from_cells(batsize)

        if self.not_initialized is None:
            self.init_all(batsize)
            # self.set_stack_states(ha_tm1, hf_tm1)

        ha_tm1 = zip(*[torch.split(ha_tm1_e, 1, 0) for ha_tm1_e in ha_tm1])
        hf_tm1 = zip(*[torch.split(hf_tm1_e, 1, 0) for hf_tm1_e in hf_tm1])

        # region 1. get inputs for update
        for i in range(ctrl_tm1.size(0)):
            ctrl = ctrl_tm1.data[i]
            if ctrl == 1 or ctrl == 3:       # has children, has siblings
                # push ancestral stacks
                self.ance_sym_stacks[i].append(y_tm1[i])
                self.ance_stacks[i].append(ha_tm1[i])
                # update frat and control
                if len(self.frat_stacks[i]) > 0:
                    self.frat_sym_stacks[i][-1] = y_tm1[i]
                    self.frat_stacks[i][-1] = hf_tm1[i]
                    self.frat_ctrl_stack[i][-1] = ctrl
                # push frat control
                self.frat_ctrl_stack[i].append(None)
                # push init frat
                self.frat_sym_stacks[i].append(self.y_f_0)
                if not self.init_fraternal_to_ancestor:
                    zerofrat = [q.var(torch.zeros(frat_stacks_top_i_e.size())).cuda(frat_stacks_top_i_e).v
                                for frat_stacks_top_i_e in hf_tm1[i]]
                else:
                    zerofrat = ha_tm1[i]
                self.frat_stacks[i].append(zerofrat)
            elif ctrl == 2:     # no children, has siblings
                # keep ancestral stacks
                pass
                # update fraternal stacks
                self.frat_stacks[i][-1] = hf_tm1[i]
                self.frat_sym_stacks[i][-1] = y_tm1[i]
                # update ctrl stack
                self.frat_ctrl_stack[i][-1] = ctrl
            elif ctrl == 4:     # no children, no siblings
                # pop all stacks until something with siblings or empty
                while True:
                    self.ance_stacks[i].pop()
                    self.frat_stacks[i].pop()
                    self.ance_sym_stacks[i].pop()
                    self.frat_sym_stacks[i].pop()
                    self.frat_ctrl_stack[i].pop()
                    if self.frat_ctrl_stack[i][-1] in (1, 2) or len(self.frat_ctrl_stack[i]) == 0:
                        break


        # make an update
        ance_states, frat_states = self.get_stack_states()
        self.set_states_of_cells(ance_states, frat_states)

        y_a_tm1, y_f_tm1 = self.get_stack_syms()
        y_a_tm1_emb, y_f_tm1_emb = self.ancemb(y_a_tm1), self.fratemb(y_f_tm1)
        tocat = [y_a_tm1_emb, y_f_tm1_emb]
        tocat = tocat + [extravec_t] if extravec_t is not None else tocat
        x_t = torch.cat(tocat, 1)

        if len(self.cells) == 2:
            ance_cell_out = self.cells[0].forward(x_t, t=t, **kw)
            frat_cell_out = self.cells[1].forward(x_t, t=t, **kw)
            cell_out = torch.cat([ance_cell_out, frat_cell_out], 1)
        else:
            cell_out = self.cells[0].forward(x_t, t=t, **kw)
        return cell_out






