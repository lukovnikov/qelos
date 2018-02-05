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
        self.outdim = self.tensor.size(0)       # number of relations
        self.numents = self.tensor.size(1)      # number of entities
        self.linout = nn.Linear(indim, self.outdim, bias=False)
        self.softmax = q.Softmax()

    def forward(self, ids, travvec):  # (batsize, dim) - start entity id
        # select by id
        tensorslice = self.tensor.index_select(1, ids)  # select tensor slice for starting entity
        tensorslice = tensorslice.transpose(1, 0)
        mask = tensorslice.sum(2) > 0       # mask out relation that are applicable for the entity
        # compute weights for tensor summarization (by rel dim) from given vector
        y = self.linout(travvec)
        y, _ = self.softmax(y, mask=mask)
        # y = self.softmax(y)
        #
        z = tensorslice.float() * y.unsqueeze(2)        # multiply each relation fibre by its weight for each example
        z, _ = z.max(1)                                 # do a max across entity fibres instead of sum
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


class TwoStackCell(RecStatefulContainer):   # for depth-first tree decoding
    """ Works with two or one single-state cells (GRU or CatLSTM).
        Can be used as a core in ModularDecoderCell
        ===
        The TwoStackCell maintains separate ancestral and fraternal stacks
        for ancestral resp. fraternal states and previous states and previous symbols,
        as well as a stack of fraternal control symbols that specify if that element
        has a next sibling under the same parent.
    """
    def __init__(self, emb, cell, frat_init="zero",
                 cell_inp_router_mode="default",
                 **kw):
        """
        :param emb: can be a pair of embedders or a single embedder.
            If a pair of embedders is given, irst one is used for ancestral, second for fraternal.
            Embedders used must return a pair of values, conforming to q.WordEmb's signature.
            Fraternal embedder must have "<START>" symbol in the dictionary.
        :param cell: can contain a tuple of two reccables or a single reccable.
        If cell argument is a pair of reccables, first one is ancestral, second one is fraternal,
            and cell_inp_router_mode is set to "separated" by default.
        If cell argument is a single reccable, its states will be split between
            ancestral and fraternal routing (first half resp. second half).
            The cell_inp_router_mode is set to "joined" by default.
        :param cell_inp_router_mode: tells how to route the inputs to the given cells.
            Default is "default" and chosen as described above. Other options are "joined" and "separated".
            If "joined", the ancestral and fraternal embeddings and the context are concatenated
            and fed to the cell/both cells (works with single and dual cell).
            If "separated", the ancestral and fraternal embeddings are *each* concatenated
            with the context and fed to the ancestral resp. fraternal cells (only works with dual cell).
        :param frat_init: tells how to initialize fraternal states of cells.
            Default is "zero", meaning the initial fraternal states are zeros.
            If "ancestor", the initial fraternal states are initialized to the states of the parent.
            If tensor value(s) is given, this value(s) is used as initial fraternal states.
        """
        super(TwoStackCell, self).__init__(**kw)
        # stacks are lists (per example) of tuples (per state) of vars (states)
        self.ance_stacks = None
        self.frat_stacks = None   # per-example stacks
        self.ance_zero_stack_elem = None
        self.frat_zero_stack_elem = None
        self.ance_sym_stacks = None
        self.frat_sym_stacks = None
        self.ance_sym_zero_stack_elem = q.val(np.asarray([0])).v
        self.frat_sym_zero_stack_elem = q.val(np.asarray([0])).v

        self.frat_ctrl_stack = None        # per-example ctrl stacks
        # self.frat_ctrl_zero_stack_elem = q.val(np.asarray([0])).v

        if isinstance(frat_init, basestring):
            self.frat_init_mode = frat_init
        else:
            self.frat_init_mode = "value"
            self.frat_init_value = frat_init

        if isinstance(cell, tuple):     # dual cell, cell contains recstacks
            assert(len(cell) == 2)
            cells = cell
            cell_inp_router_mode = "separated" if cell_inp_router_mode == "default" else cell_inp_router_mode
        else:
            cells = (cell,)
            cell_inp_router_mode = "joined" if cell_inp_router_mode == "default" else cell_inp_router_mode

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

        def cell_inp_router(a, b, ctx):
            if cell_inp_router_mode == "joined":
                tocat = [a, b]
                if q.issequence(ctx):
                    tocat += list(ctx)
                else:
                    tocat += [ctx]
                return torch.cat([xe for xe in tocat if xe is not None], 1)
            elif cell_inp_router_mode == "separated":
                tocat_anc, tocat_frat = [a], [b]
                if q.issequence(ctx):
                    tocat_anc += [ctx[0]]
                    tocat_frat += [ctx[1]]
                else:
                    tocat_anc += [ctx]
                    tocat_frat += [ctx]
                return torch.cat([xe for xe in tocat_anc if xe is not None], 1), \
                       torch.cat([xe for xe in tocat_frat if xe is not None], 1)

        self.cell_inp_router = cell_inp_router

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
        self.frat_ctrl_stack = None
        self.ance_zero_stack_elem = None
        self.frat_zero_stack_elem = None
        self.ance_sym_stacks = None
        self.frat_sym_stacks = None

    def set_init_states(self, *states):
        if len(self.cells) == 1:
            self.cells[0].set_init_states(*states)
        else:   # ancestral first
            anc_states = states[:self.cells[0].numstates]
            frat_states = states[self.cells[0].numstates:]
            self.cells[0].set_init_states(*anc_states)
            self.cells[1].set_init_states(*frat_states)

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
        ance_states = []
        frat_states = []
        for ance_stack, frat_stack in zip(self.ance_stacks, self.frat_stacks):
            if len(ance_stack) == 0:
                ance_stack_l = self.ance_zero_stack_elem
                frat_stack_l = self.frat_zero_stack_elem
            else:
                ance_stack_l = ance_stack[-1]
                frat_stack_l = frat_stack[-1]
                if self.ance_zero_stack_elem is None:    # set zero stacks
                    self.ance_zero_stack_elem = [q.var(torch.zeros(ance_stack_l_e.size())).cuda(ance_stack_l_e).v
                                                 for ance_stack_l_e in ance_stack_l]
                    self.frat_zero_stack_elem = [q.var(torch.zeros(frat_stack_l_e.size())).cuda(frat_stack_l_e).v
                                                 for frat_stack_l_e in frat_stack_l]
            ance_states.append(ance_stack_l)
            frat_states.append(frat_stack_l)
        ance_states = [torch.cat(l, 0) for l in zip(*ance_states)]
        frat_states = [torch.cat(l, 0) for l in zip(*frat_states)]
        return ance_states, frat_states

    def get_stack_syms(self):
        y_a_tm1, y_f_tm1 = [], []
        for ance_sym_stack, frat_sym_stack in zip(self.ance_sym_stacks, self.frat_sym_stacks):
            if len(ance_sym_stack) == 0:
                ance_sym_stack_l = self.ance_sym_zero_stack_elem
                frat_sym_stack_l = self.frat_sym_zero_stack_elem
            else:
                ance_sym_stack_l = ance_sym_stack[-1]
                frat_sym_stack_l = frat_sym_stack[-1]
            y_a_tm1.append(ance_sym_stack_l)
            y_f_tm1.append(frat_sym_stack_l)
        y_f_tm1 = torch.cat(y_f_tm1, 0)
        y_a_tm1 = torch.cat(y_a_tm1, 0)
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

    def forward(self, y_tm1, ctrl_tm1, ctx_t=None, t=None, outmask_t=None, **kw):
        batsize = y_tm1.size(0)

        ha_tm1, hf_tm1 = self.get_states_from_cells(batsize)

        if not self.initialized:
            self.init_all(batsize)
            # self.set_stack_states(ha_tm1, hf_tm1)

        ha_tm1 = zip(*[torch.split(ha_tm1_e, 1, 0) for ha_tm1_e in ha_tm1])
        hf_tm1 = zip(*[torch.split(hf_tm1_e, 1, 0) for hf_tm1_e in hf_tm1])

        # maplist = zip(self.ance_sym_stacks, self.frat_sym_stacks,                      self.ance_stacks,                      self.frat_stacks,                      self.frat_ctrl_stack,                      [a[0] for a in ctrl_tm1.data.split(1, 0)],                      ha_tm1,                      hf_tm1)

        # region 1. update stacks
        for i in range(ctrl_tm1.size(0)):
            ctrl = ctrl_tm1.data[i]
            if ctrl == 0:                    # masked
                pass
            else:
                pass
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
                if self.frat_init_mode == "zero":
                    zerofrat = [q.var(torch.zeros(frat_stacks_top_i_e.size())).cuda(frat_stacks_top_i_e).v
                                for frat_stacks_top_i_e in hf_tm1[i]]
                elif self.frat_init_mode == "ancestor":
                    zerofrat = ha_tm1[i]
                elif self.frat_init_mode == "value":
                    zerofrat = self.frat_init_param
                else:
                    raise q.SumTingWongException()
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
                    if len(self.frat_ctrl_stack[i]) == 0 or self.frat_ctrl_stack[i][-1] in (1, 2):
                        break
        # endregion

        # region 2. make an update
        ance_states, frat_states = self.get_stack_states()
        self.set_states_of_cells(ance_states, frat_states)

        y_a_tm1, y_f_tm1 = self.get_stack_syms()
        y_a_tm1_emb, _ = self.ancemb(y_a_tm1)
        y_f_tm1_emb, _ = self.fratemb(y_f_tm1)

        if len(self.cells) == 2:
            cellinprouterout = self.cell_inp_router(y_a_tm1_emb, y_f_tm1_emb, ctx_t)
            if len(cellinprouterout) == 2:
                x_a_t, x_f_t = cellinprouterout
            else:
                x_a_t, x_f_t = cellinprouterout, cellinprouterout
            ance_cell_out = self.cells[0].forward(x_a_t, t=t, **kw)
            frat_cell_out = self.cells[1].forward(x_f_t, t=t, **kw)
            # cell_out = torch.cat([ance_cell_out, frat_cell_out], 1)
            cell_out = q.intercat([ance_cell_out, frat_cell_out])   # in case split attention is used later
        else:
            x_t = self.cell_inp_router(y_a_tm1_emb, y_f_tm1_emb, ctx_t)
            cell_out = self.cells[0].forward(x_t, t=t, **kw)
        # endregion
        return cell_out, {"t": t, "x_t_emb": torch.cat([y_a_tm1_emb, y_f_tm1_emb], 1), "ctx_t": ctx_t, "mask": outmask_t}


# TODO: test
class ParentStackCell(RecStatefulContainer):      # breadth-first tree decoding
    """ Works with two or one single-state cells (GRU or CatLSTM).
        Can be used as a core in ModularDecoderCell
        ===
        Maintains a stack of stacks of parent states and symbols,
        one outer stack for each parent level
        and one inner stack for parent level parents' states and symbols
    """
    def __init__(self, emb, cell, frat_init="zero",
                 cell_inp_router_mode="default", **kw):
        """ same as TwoStackCell """
        super(ParentStackCell, self).__init__(**kw)
        self.state_stacks = None
        self.symbol_stacks = None

        if isinstance(frat_init, basestring):   # zero or initial
            self.frat_init_mode = frat_init
        else:
            self.frat_init_mode = "value"
            self.frat_init_value = frat_init
        self._init_frat_states = None

        if isinstance(cell, tuple):  # dual cell, cell contains recstacks
            assert (len(cell) == 2)
            cells = cell
            cell_inp_router_mode = "separated" if cell_inp_router_mode == "default" else cell_inp_router_mode
        else:
            cells = (cell,)
            cell_inp_router_mode = "joined" if cell_inp_router_mode == "default" else cell_inp_router_mode

        self.cells = q.ModuleList(list(cells))

        if isinstance(emb, tuple):
            assert (len(emb) == 2)
            self.ancemb = emb[0]
            self.fratemb = emb[1]
        else:
            self.ancemb = emb
            self.fratemb = emb

        fratstartsym = self.fratemb.D["<START>"]
        self.y_f_0 = q.val(torch.LongTensor(1, )).v
        self.y_f_0.data.fill_(fratstartsym)

        def cell_inp_router(a, b, ctx):
            if cell_inp_router_mode == "joined":
                tocat = [a, b]
                if q.issequence(ctx):
                    tocat += list(ctx)
                else:
                    tocat += [ctx]
                return torch.cat([xe for xe in tocat if xe is not None], 1)
            elif cell_inp_router_mode == "separated":
                tocat_anc, tocat_frat = [a], [b]
                if q.issequence(ctx):
                    tocat_anc += [ctx[0]]
                    tocat_frat += [ctx[1]]
                else:
                    tocat_anc += [ctx]
                    tocat_frat += [ctx]
                return torch.cat([xe for xe in tocat_anc if xe is not None], 1), \
                       torch.cat([xe for xe in tocat_frat if xe is not None], 1)

        self.cell_inp_router = cell_inp_router

    @property
    def state_spec(self):
        if len(self.cells) == 1:
            return self.cells[0].outdim // 2, self.cells[0].outdim // 2
        else:
            return self.cells[0].outdim, self.cells[1].outdim

    def reset_state(self):
        for cell in self.cells:
            cell.reset_state()
        self.state_stacks = None
        self.symbol_stacks = None
        self._init_frat_states = None

    def set_init_states(self, *states):
        if len(self.cells) == 1:
            self.cells[0].set_init_states(*states)
        else:   # ancestral first
            anc_states = states[:self.cells[0].numstates]
            frat_states = states[self.cells[0].numstates:]
            self.cells[0].set_init_states(*anc_states)
            self.cells[1].set_init_states(*frat_states)

    def read_states_from_cells(self, batsize):
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

    @property
    def initialized(self):
        return self.state_stacks is not None

    def init_all(self, batsize):
        self.state_stacks =  tuple([[[]] for _ in range(batsize)])
        self.symbol_stacks = tuple([[[]] for _ in range(batsize)])

    def forward(self, y_tm1, ctrl_tm1, ctx_t=None, t=None, outmask_t=None, **kw):
        batsize = y_tm1.size(0)

        ha_tm1, hf_tm1 = self.read_states_from_cells(batsize)
        ance_states = zip(*[torch.split(ha_tm1_e, 1, 0) for ha_tm1_e in ha_tm1])
        ance_symbols = list(torch.split(y_tm1, 1, 0))     # must be overwritten with symbols from stack
        frat_states = zip(*[torch.split(hf_tm1_e, 1, 0) for hf_tm1_e in hf_tm1])
        frat_symbols = list(torch.split(y_tm1, 1, 0))  # when y_tm1 is root, initial frat symbols must be <START>
        # !!: overriding frat line can be more efficient

        if not self.initialized:
            self.init_all(batsize)          # initial parent state will be set because first symbol is a root*LS --> ha_tm1 (initial ancestral part of rec states) is pushed onto stack
            # initialize first ever fraternal state according to frat init mode
            if self.frat_init_mode == "initial":    # save initial frat states
                self._init_frat_states = frat_states + []
            elif self.frat_init_mode == "zero":
                hf_tm1 = [q.var(torch.zeros(hf_tm1_e.size())).cuda(hf_tm1_e).v
                            for hf_tm1_e in hf_tm1]
                frat_states = zip(*[torch.split(hf_tm1_e, 1, 0) for hf_tm1_e in hf_tm1])
            elif self.frat_init_mode == "ancestor":
                frat_states = zip(*[torch.split(ha_tm1_e, 1, 0) for ha_tm1_e in ha_tm1])
            elif self.frat_init_mode == "value":
                frat_states = [self.frat_init_value for _ in frat_states]

            # set previous frat symbols to init frat symbols
            frat_symbols = [self.y_f_0 for _ in frat_symbols]

        # region update stacks
        for i in range(ctrl_tm1.size(0)):
            ctrl = ctrl_tm1.data[i]
            state_stack = self.state_stacks[i]
            symbol_stack = self.symbol_stacks[i]
            if ctrl == 0 or len(state_stack) == 0:       # masked
                continue
            else:
                pass

            previous_was_last = ctrl == 3 or ctrl == 4
            previous_was_leaf = ctrl == 2 or ctrl == 4

            if not previous_was_leaf:   # no children --> ancestral data not queued
                                        # --> queue ancestral data for next level deeper
                state_stack[-1].append(ance_states[i])
                symbol_stack[-1].append(y_tm1[i])

            if previous_was_last:
                # pop parent queue
                if len(state_stack) > 1:    # should only be false at init
                    del state_stack[-2][0]
                    del symbol_stack[-2][0]
                    # pop stack until empty or next non-finished
                    if len(state_stack[-2]) == 0:   # all parents have been consumed
                        del state_stack[-2]
                        del symbol_stack[-2]
                # pop or push last stack
                if len(state_stack) < 2 or len(state_stack[-2]) == 0:
                    if len(state_stack) == 0:
                        print("empty stack")
                    if len(state_stack[-1]) == 0:   # no next depth level needed -> terminate
                        del state_stack[-1]
                        del symbol_stack[-1]
                    else:
                        state_stack.append([])    # new depth level
                        symbol_stack.append([])

                # reset frat
                frat_symbols[i] = self.y_f_0
                if self.frat_init_mode == "zero":
                    zerofrat = [q.var(torch.zeros(frat_stacks_top_i_e.size())).cuda(frat_stacks_top_i_e).v
                                for frat_stacks_top_i_e in frat_states[i]]
                elif self.frat_init_mode == "initial":
                    zerofrat = self._init_frat_states[i]
                elif self.frat_init_mode == "ancestor":
                    zerofrat = ance_states[i]
                elif self.frat_init_mode == "value":
                    zerofrat = self.frat_init_value
                else:
                    raise q.SumTingWongException()
                frat_states[i] = zerofrat
        # endregion

        # region make cell update
        for i, state_stack, symbol_stack in zip(range(len(self.state_stacks)),
                                                self.state_stacks, self.symbol_stacks):
            if len(state_stack) > 0:
                ance_states[i] = state_stack[-2][0]
                ance_symbols[i] = symbol_stack[-2][0]
            else:       # terminated
                ance_states[i] = [q.var(torch.zeros(ance_states_i_e.size())).cuda(ance_states_i_e).v for ance_states_i_e in ance_states[i]]
                frat_states[i] = [q.var(torch.zeros(frat_states_i_e.size())).cuda(frat_states_i_e).v for frat_states_i_e in frat_states[i]]
                ance_symbols[i] = q.var(torch.zeros(ance_symbols[i].size()).long()).cuda(ance_symbols[i]).v
                frat_symbols[i] = q.var(torch.zeros(frat_symbols[i].size()).long()).cuda(frat_symbols[i]).v
        # ance_states = [state_stack[-2][0] for state_stack in self.state_stacks]
        # ance_symbols = [symbol_stack[-2][0] for symbol_stack in self.symbol_stacks]

        ance_states = [torch.cat(l, 0) for l in zip(*ance_states)]
        frat_states = [torch.cat(l, 0) for l in zip(*frat_states)]

        self.set_states_of_cells(ance_states, frat_states)

        y_a_tm1 = torch.cat(ance_symbols, 0)
        y_f_tm1 = torch.cat(frat_symbols, 0)
        y_a_tm1_emb, _ = self.ancemb(y_a_tm1)
        y_f_tm1_emb, _ = self.fratemb(y_f_tm1)

        if len(self.cells) == 2:
            cellinprouterout = self.cell_inp_router(y_a_tm1_emb, y_f_tm1_emb, ctx_t)
            if len(cellinprouterout) == 2:
                x_a_t, x_f_t = cellinprouterout
            else:
                x_a_t, x_f_t = cellinprouterout, cellinprouterout
            ance_cell_out = self.cells[0].forward(x_a_t, t=t, **kw)
            frat_cell_out = self.cells[1].forward(x_f_t, t=t, **kw)
            # cell_out = torch.cat([ance_cell_out, frat_cell_out], 1)
            cell_out = q.intercat([ance_cell_out, frat_cell_out])   # in case split attention is used later
            # TODO: check that intercatted is not fed back in here again
        else:
            x_t = self.cell_inp_router(y_a_tm1_emb, y_f_tm1_emb, ctx_t)
            cell_out = self.cells[0].forward(x_t, t=t, **kw)
        # endregion
        return cell_out, {"t": t, "x_t_emb": torch.cat([y_a_tm1_emb, y_f_tm1_emb], 1), "ctx_t": ctx_t, "mask": outmask_t}


class DynamicOracleRunner(q.DecoderRunner):
    """
    Runs a decoder using the provided dynamic tracker.
    Should be set as the input getter in decoder cells.
    Called at every time step.
    Given full arguments for a timestep, chooses the output token and stores it,
    chooses a gold token and stores it, and return input arguments for next time step
    from the chosen output token.
    ===
    With PAS tracker and exploration, this runner will be feeding decoder
     with randomly sampled (possibly wrong) next symbols, the stored seq
     of this runner will store what was chosen and the stored gold seq
     will contain symbols sampled from acceptable choices (or set
     to the chosen choices if they are in acceptable choices). These gold
     choices will be used by the tracker to track decoding over an acceptable track.
     Thus, exploration with PAS tracker will only go 1 error deep.
    """
    def __init__(self, tracker=None,
                 inparggetter=lambda x: (x, {}),        # tranforms from output symbol to input symbols
                 scores2probs=q.Softmax(),
                 mode="sample",  # "sample" or "argmax" or "uniform" or "esample"
                 eps=0.2,
                 explore=0.,
                 **kw):
        """ sample mode samples a correct token from predicted dist.
            argmax mode takes the most probable correct token (according to predicted dist).
            uniform mode takes any correct token with equal prob """
        super(DynamicOracleRunner, self).__init__(**kw)
        self.inparggetter = inparggetter        # transforms from output symbols to input symbols
        self.scores2probs = scores2probs
        self.mode = mode
        self.eps = eps
        self.explore = explore
        #
        self.tracker = tracker
        self.seqacc = []        # history of what has been fed to next time step
        self.goldacc = []       # use for supervision

        self._argmax_in_eval = True

    def reset(self):
        self.seqacc = []
        self.goldacc = []
        self.tracker.reset()

    def get_sequence(self):
        """ get the chosen sequence """
        ret = torch.stack(self.seqacc, 1)
        return ret

    def get_gold_sequence(self):
        ret = torch.stack(self.goldacc, 1)
        return ret

    def forward(self, t=None, x=None, xkw=None, y_t=None):
        outkwargs = {"t": t}
        eids = xkw["eids"]        # must be given ids of examples
        eids_np = eids.cpu().data.numpy()

        if y_t is None:
            assert(t == 0)
            if q.issequence(x):
                assert(len(x) == 1)
                x = x[0]
            x_t = x
            gold_t = x_t
        else:
            mode = "argmax" if self._argmax_in_eval and not self.training else self.mode
            if q.issequence(y_t):
                assert(len(y_t) == 1)
                y_t = y_t[0]
            # compute prob mask
            ymask_np = np.zeros(y_t.size(), dtype="float32")
            ymask_expl_np = np.ones(y_t.size(), dtype="float32")
            use_expl_mask = False
            for i, eid in enumerate(eids_np):
                avalidnext = None
                validnext = self.tracker.get_valid_next(eid)   # set of ids
                if isinstance(validnext, tuple) and len(validnext) == 2:
                    validnext, avalidnext = validnext
                ymask_np[i, list(validnext)] = 1.
                if avalidnext is not None:
                    ymask_expl_np[i, :] = 0.
                    ymask_expl_np[i, list(avalidnext)] = 1.
                    use_expl_mask = True
            ymask = q.var(ymask_np).cuda(y_t).v
            ymask_expl = q.var(ymask_expl_np).cuda(y_t).v if use_expl_mask else None

            if self.explore > 0:
                _y_t = y_t + torch.log(ymask_expl) if ymask_expl is not None else y_t
                unmaskedprobs = self.scores2probs(_y_t)
                if mode == "sample":
                    x_t = torch.distributions.Categorical(unmaskedprobs).sample()
                    # x_t = torch.multinomial(unmaskedprobs, 1).squeeze(-1).detach()
                elif mode == "uniform":
                    expl_probs = ymask_expl if ymask_expl is not None else q.var(torch.ones(y_t.size())).cuda(y_t).v
                    x_t = torch.distributions.Categorical(expl_probs).sample()
                elif mode == "argmax":
                    _, x_t = torch.max(unmaskedprobs, 1)
                else:
                    raise q.SumTingWongException("unsupported mode: {}".format(mode))

            # get probs
            _y_t = y_t + torch.log(ymask)
            goldprobs, _ = self.scores2probs(_y_t, mask=ymask)     # probs for allowed symbols

            # sample gold from probs
            if mode == "sample":
                gold_t = torch.distributions.Categorical(goldprobs).sample()
                # gold_t = torch.multinomial(goldprobs, 1).squeeze(-1).detach()
            elif mode == "uniform":
                gold_t = torch.distributions.Categorical(ymask).sample()
            elif mode == "esample":
                gold_t = torch.distributions.Categorical(goldprobs).sample()
                alt_gold_t = torch.distributions.Categorical(ymask).sample()
                _epsprobs = (q.var(torch.rand(gold_t.size())).cuda(gold_t).v < self.eps).long()
                gold_t = torch.gather(torch.stack([gold_t, alt_gold_t], 1), 1, _epsprobs.unsqueeze(1)).squeeze(1)
            elif mode == "argmax":
                _, gold_t = torch.max(goldprobs, 1)
            else:
                raise q.SumTingWongException("unsupported mode: {}".format(mode))

            if self.explore == 0:
                x_t = gold_t
            else:
                if self.explore < 1:  # mixture
                    mixmask = q.var(torch.rand(x_t.size())).cuda(x_t).v > self.explore
                    mixidx = mixmask.long()
                    tomix = torch.stack([x_t, gold_t], 1)
                    x_t = torch.gather(tomix, 1, mixidx.unsqueeze(1)).squeeze(-1)
                # if sampled choice is in gold probs, set gold to sampled
                x_t_in_goldprobs = (torch.gather(ymask, 1, x_t.unsqueeze(1)) > 0).long()
                tomix = torch.stack([gold_t, x_t], 1)
                gold_t = torch.gather(tomix, 1, x_t_in_goldprobs).squeeze(-1)

            # store sampled
            self.seqacc.append(x_t)
            self.goldacc.append(gold_t)

            # update tracker
            for x_t_e, eid, gold_t_e in zip(x_t.cpu().data.numpy(), eids_np, gold_t.cpu().data.numpy()):
                # TODO: ?? switch to gold_t here instead of providing to alt_x ??
                self.tracker.update(eid, x_t_e, alt_x=gold_t_e)

        # termination
        _terminates = [self.tracker.is_terminated(eid) for eid in eids_np]
        _terminate = all(_terminates)

        # return
        r = self.inparggetter(x_t)
        if isinstance(r, tuple) and len(r) == 2 and isinstance(r[1], dict):
            inpargs, kwupd = r
            outkwargs.update(kwupd)
        else:
            inpargs = r

        if _terminate:
            outkwargs.update({"_stop": _terminate})
        return inpargs, outkwargs


class MultiTeacherForcer(q.DecoderRunner):
    """ like dynamicoraclerunner except gold is decided before rec step thus can't sample from rec's out dist """
    def __init__(self, tracker=None,
                 inparggetter=lambda x: (x, {}),
                 feed_x_tp1 = False,
                 **kw):
        super(MultiTeacherForcer, self).__init__(**kw)
        self.inparggetter = inparggetter
        self.feed_x_tp1 = feed_x_tp1
        self.tracker = tracker

        self.seqacc = []
        self.goldacc = []

    def reset(self):
        self.seqacc = []
        self.goldacc = []
        self.tracker.reset()

    def get_sequence(self):
        ret = torch.stack(self.seqacc, 1)
        return ret

    def get_gold_sequence(self):
        ret = torch.stack(self.goldacc, 1)

    def forward(self, t=None, x=None, xkw=None, y_t=None):
        outkwargs = {"t": t}
        eids = xkw["eids"]        # must be given ids of examples
        eids_np = eids.cpu().data.numpy()

        if y_t is None:
            assert (t == 0)
            assert (len(self.seqacc) == 0)
            if q.issequence(len(x) == 1):
                assert (len(x) == 1)
                x = x[0]
            x_t = x
        else:
            x_t = self.seqacc[-1]

        ymask_np = np.zeros(y_t.size(), dtype="float32")
        for i, eid in enumerate(eids_np):
            avalidnext = None
            validnext = self.tracker.get_valid_next(eid)
            if isinstance(validnext, tuple) and len(validnext) == 2:
                validnext, avalidnext = validnext
            ymask_np[i, list(validnext)] = 1.
        ymask = q.var(ymask_np).cuda(y_t).v
        gold_t = torch.distributions.Categorical(ymask).sample()

        x_tp1 = gold_t

        # store sampled
        self.seqacc.append(x_tp1)
        self.goldacc.append(gold_t)

        # update tracker
        for x_t_e, eid, gold_t_e in zip(x_tp1.cpu().data.numpy(), eids_np, gold_t.cpu().data.numpy()):
            self.tracker.update(eid, x_t_e, alt_x=gold_t_e)

        # termination
        _terminates = [self.tracker.is_terminated(eid) for eid in eids_np]
        _terminate = all(_terminates)

        # return
        if self.feed_x_tp1:
            r = self.inparggetter(x_t, x_tp1)
        else:
            r = self.inparggetter(x_t)

        if isinstance(r, tuple) and len(r) == 2 and isinstance(r[1], dict):
            inpargs, kwupd = r
            outkwargs.update(kwupd)
        else:
            inpargs = r

        # DONE: if inpargs is a sequence, it will be handled by Decoder's forward (_get_inputs_t() return is inpargs)

        if _terminate:
            outkwargs.update({"_stop": _terminate})
        return inpargs, outkwargs








