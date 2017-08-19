import torch
from torch.autograd import Variable
from torch import nn
from qelos.util import name2fn, issequence, isnumber
from qelos.basic import Stack
import qelos as q


# region I. RNN cells
class RNUGate(nn.Module):

    def __init__(self, indim, outdim, hdim=None, activation="sigmoid", use_bias=True):
        super(RNUGate, self).__init__()
        self.indim, self.outdim, self.activation, self.use_bias = indim, outdim, activation, use_bias
        self.activation_fn = name2fn(self.activation)()
        self.W = nn.Parameter(torch.FloatTensor(self.indim, self.outdim))
        udim = self.outdim if hdim is None else hdim
        self.U = nn.Parameter(torch.FloatTensor(udim, self.outdim))
        if self.use_bias:
            self.b = nn.Parameter(torch.FloatTensor(1, self.outdim))
        else:
            self.register_parameter("b", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform(self.W, gain=nn.init.calculate_gain(self.activation))
        nn.init.xavier_uniform(self.U, gain=nn.init.calculate_gain(self.activation))
        if self.use_bias:
            nn.init.uniform(self.b, -0.01, 0.01)

    def forward(self, x, h, debug=False):
        ret = torch.mm(x, self.W)
        ret.addmm_(h, self.U)
        if self.use_bias:
            ret.add_(self.b)
        if debug: rg = ret.data.numpy()
        ret = self.activation_fn(ret)
        if debug:  return ret, rg
        return ret


class PackedRNUGates(nn.Module):
    def __init__(self, indim, outgatespecs, use_bias=True, rec_bn=False):
        super(PackedRNUGates, self).__init__()
        self.indim = indim
        self.outdim = 0
        self.outgates = []
        self.use_bias = use_bias
        for outgatespec in outgatespecs:
            gateoutdim = outgatespec[0]
            gateact = outgatespec[1]
            gateact_fn = name2fn(gateact)()
            self.outgates.append(((self.outdim, self.outdim+gateoutdim), gateact_fn))
            self.outdim += gateoutdim
        self.W = nn.Parameter(torch.FloatTensor(self.indim, self.outdim))
        if self.use_bias:
            self.b = nn.Parameter(torch.FloatTensor(1, self.outdim,))
        else:
            self.register_parameter("b", None)
        self.rec_bn = rec_bn
        self._rbn = None
        if self.rec_bn is True:
            self._rbn = q.SeqBatchNorm1d(self.outdim)
        self.reset_parameters()

    def reset_parameters(self):     # TODO incorporate gain per activation fn
        nn.init.xavier_uniform(self.W)
        if self.use_bias:
            nn.init.uniform(self.b, -0.01, 0.01)
        if self.rec_bn is True:
            self._rbn.reset_parameters()

    def forward(self, x, h, t=None):
        x = torch.cat([x, h], 1)
        v = torch.mm(x, self.W)
        if self._rbn is not None:
            v = self._rbn(v, t)      # TODO: check masking
        if self.use_bias:
            v.add_(self.b)
        ret = []
        for outgate in self.outgates:
            outgateslice = outgate[0]
            outgateactfn = outgate[1]
            gateval = v[:, outgateslice[0]:outgateslice[1]]
            gateval = outgateactfn(gateval)
            ret.append(gateval)
        ret = tuple(ret)
        if len(ret) == 1:
            ret = ret[0]
        return ret


class Reccable(nn.Module):
    """
    Accepts mask_t and t in its forward kwargs.
    """


class RecStateful(Reccable):
    """
    Stores recurrent state.
    """
    @property
    def state_spec(self):
        raise NotImplementedError("use subclass")

    @property
    def numstates(self):
        return len(self.state_spec)

    def reset_state(self):
        raise NotImplementedError("use subclass. subclasses must implement")

    def set_init_states(self, *states):
        raise NotImplementedError("use subclass. subclasses must implement this method")

    def get_init_states(self, arg):
        raise NotImplementedError("use subclass. subclasses must implement this method")


class RecStatefulContainer(RecStateful):
    """
    Doesn't have own stored states
    """
    def reset_state(self):
        raise NotImplementedError("use subclass. subclasses must implement")

    def set_init_states(self, *states):
        raise NotImplementedError("use subclass. subclasses must implement this method")


class RNUBase(RecStateful):
    def __init__(self, *x, **kw):
        super(RecStateful, self).__init__(*x, **kw)
        self._init_states = None
        self._states = None
        self._y_tm1 = None

    def to_layer(self):
        return RNNLayer(self)

    def reset_state(self):
        """ should be called before every rollout.
        resets states and shared dropout masks.
        """
        self._states = None

    def get_states(self, arg):
        if self._states is None:    # states don't exist yet
            _states = []
            for initstate in self.get_init_states(arg):
                _states.append(initstate)
        else:
            _states = self._states
        return _states

    def set_states(self, *states):
        if self._states is not None:
            assert(len(states) == len(self._states))
            for i, state in enumerate(states):
                assert(state.size() == self._states[i].size())
                self._states[i] = state
        else:
            self._states = []
            for state in states:
                self._states.append(state)

    def get_init_states(self, arg):
        """
        :param arg: batch size (will generate and return compatible init states) or None (will return what is stored)
        :return: initial states, states that have previously been set or newly generated zero states based on given batch size
        """
        if arg is None:
            return self._init_states
        assert(isnumber(arg) or arg is None)       # is batch size
        if self._init_states is None:       # no states have been set using .set_init_states()
            _init_states = [None] * self.numstates
        else:
            _init_states = self._init_states
        # fill up with zeros and expand where necessary
        assert(self.numstates == len(_init_states))
        for i in range(len(_init_states)):
            statespec = self.state_spec[i]
            initstate = _init_states[i]
            if initstate is None:
                state_0 = q.var(torch.zeros(statespec)).cuda(next(self.parameters()).is_cuda).v
                _init_states[i] = state_0
                initstate = state_0
            if initstate.dim() == 2:        # init state set differently for different batches
                if arg > initstate.size(0):
                    raise Exception("asked for a bigger batch size than init states")
                elif arg == initstate.size(0):
                    pass
                else:
                    if arg == 0:
                        return initstate[0]
                    else:
                        return initstate[:arg]
            elif initstate.dim() == 1 and arg > 0:
                _init_states[i] = initstate.unsqueeze(0).expand(arg, initstate.size(-1))
            else:
                raise Exception("initial states set to wrong dimensional values. Must be 1D (will be repeated) or 2D.")
        return _init_states

    def set_init_states(self, *states):
        """
        Sets initial states of this RNU element to provided states
        :param states: list of tensors (batsize, dim) or (dim), can be smaller (will be filled up with zeros)
        :return:
        """
        assert(len(states) <= self.numstates)
        self._init_states = []
        i = 0
        for statespec in self.state_spec:
            if i < len(states):
                assert(statespec == states[i].size(-1))
                self._init_states.append(states[i])
            else:
                self._init_states.append(None)
            i += 1

    def forward(self, x_t, t=None, mask_t=None):
        batsize = x_t.size(0)
        states = self.get_states(batsize)
        ret = self._forward(x_t, *states, t=t)
        y_t = ret[0]
        newstates = ret[1:]
        st = []
        if mask_t is not None:
            for newstate, oldstate in zip(newstates, states):
                newstate = newstate * mask_t + oldstate * (1 - mask_t)
                st.append(newstate)
            if mask_t is not None:  # moved from RNNLayer
                if self._y_tm1 is None:
                    self._y_tm1 = q.var(torch.zeros(y_t.size())).cuda(crit=y_t).v
                y_t = y_t * mask_t + self._y_tm1 * (1 - mask_t)
                self._y_tm1 = y_t
        else:
            st = newstates
        self.set_states(*st)
        return y_t


class RNU(RNUBase):
    debug = False

    def __init__(self, indim, outdim, use_bias=True, activation="tanh",
                 dropout_in=None, dropout_rec=None, zoneout=None):
        super(RNU, self).__init__()
        self.indim, self.outdim, self.use_bias, self.dropout_in, self.dropout_rec, self.zoneout = \
            indim, outdim, use_bias, dropout_in, dropout_rec, zoneout
        self.activation = activation
        self.main_gate = RNUGate(self.indim, self.outdim, use_bias=use_bias, activation=activation)

        if self.dropout_in:
            self.dropout_in = nn.Dropout(p=self.dropout_in)
        if self.dropout_rec:
            self.dropout_rec = nn.Dropout(p=self.dropout_rec)
        if self.zoneout:
            self.zoneout = nn.Dropout(p=self.zoneout)

        self.reset_parameters()

    def reset_parameters(self):
        self.main_gate.reset_parameters()

    @property
    def state_spec(self):
        return self.outdim,

    def _forward(self, x_t, h_tm1, t=None):
        if self.dropout_in:
            x_t = self.dropout_in(x_t)
        if self.dropout_rec:
            h_tm1 = self.dropout_rec(h_tm1)
        h_t = self.main_gate(x_t, h_tm1)
        if self.zoneout:
            zoner = q.var(torch.ones(h_t.size())).cuda(crit=h_t).v
            zoner = self.zoneout(zoner)
            h_t = torch.mul(1 - zoner, h_tm1) + torch.mul(zoner, h_t)
        return h_t, h_t


class _GRUCell(nn.Module):      # TODO: test rbn
    def __init__(self, indim, outdim, bias=True, gate_activation="sigmoid",
                 activation="tanh",
                 recurrent_batch_norm=None):
        super(_GRUCell, self).__init__()
        self.indim, self.outdim, self.use_bias = indim, outdim, bias
        self.gate_activation, self.activation = gate_activation, activation        # sigm, tanh
        self.hdim = self.outdim
        if self.activation == "crelu":
            self.hdim = self.hdim // 2
        self._rbn_on = recurrent_batch_norm
        self._rbn_gates = self._rbn_on == "full" or self._rbn_on == "gates"
        self._rbn_main = self._rbn_on == "full" or self._rbn_on == "main"
        self.gates = PackedRNUGates(self.indim+self.outdim,
                                   [(self.outdim, self.gate_activation),
                                    (self.outdim, self.gate_activation)],
                                   use_bias=self.use_bias,
                                    rec_bn=self._rbn_gates)
        self.main_gate = PackedRNUGates(self.indim + self.outdim,
                                        [(self.hdim, None)],
                                        use_bias=self.use_bias,
                                        rec_bn=self._rbn_main)
        self.activation_fn = name2fn(activation)()
        self.reset_parameters()

    def reset_parameters(self):
        self.gates.reset_parameters()
        self.main_gate.reset_parameters()

    def forward(self, x_t, h_tm1, t=None):
        update_gate, reset_gate = self.gates(x_t, h_tm1, t=t)
        canh = torch.mul(h_tm1, reset_gate)
        canh = self.main_gate(x_t, canh, t=t)
        canh = self.activation_fn(canh)
        h_t = (1 - update_gate) * h_tm1 + update_gate * canh
        return h_t


class GRUCell(RNUBase):
    debug = False

    def __init__(self, indim, outdim, use_bias=True,
                 dropout_in=None, dropout_rec=None, zoneout=None,
                 shared_dropout_rec=None, shared_zoneout=None,
                 use_cudnn_cell=True, activation="tanh",
                 gate_activation="sigmoid", rec_batch_norm=None):    # custom activations have only an effect when not using cudnn cell
        super(GRUCell, self).__init__()
        self.indim, self.outdim, self.use_bias, self.dropout_in, self.dropout_rec, self.zoneout, self.shared_dropout_rec, self.shared_zoneout = \
            indim, outdim, use_bias, dropout_in, dropout_rec, zoneout, shared_dropout_rec, shared_zoneout
        self.use_cudnn_cell = use_cudnn_cell
        self.activation, self.gate_activation, self.recbn = activation, gate_activation, rec_batch_norm
        self.nncell = None
        self.setcell()

        if self.dropout_in:
            self.dropout_in = nn.Dropout(p=self.dropout_in)
        if self.dropout_rec:
            self.dropout_rec = nn.Dropout(p=self.dropout_rec)
        if self.shared_dropout_rec:
            self.shared_dropout_rec = nn.Dropout(p=self.shared_dropout_rec)
            self.shared_dropout_reccer = None
        if self.zoneout:
            self.zoner = None
            self.zoneout = nn.Dropout(p=self.zoneout)
        if self.shared_zoneout:
            self.shared_zoneout = nn.Dropout(p=self.shared_zoneout)
            self.shared_zoneouter = None

    @property
    def h_0(self):
        return self.get_init_states(0)[0]

    @h_0.setter
    def h_0(self, value):
        self.set_init_states(value)

    def setcell(self):
        if self.use_cudnn_cell:
            self.nncell = nn.GRUCell(self.indim, self.outdim, bias=self.use_bias)
        else:
            self.nncell = _GRUCell(self.indim, self.outdim, bias=self.use_bias,
                                   activation=self.activation,
                                   gate_activation=self.gate_activation,
                                   recurrent_batch_norm=self.recbn)

    def apply_nncell(self, *x, **kw):
        t = kw["t"] if "t" in kw else None
        if self.use_cudnn_cell:
            return self.nncell(*x)
        else:
            return self.nncell(*x, t=t)

    def reset_parameters(self):
        # self.gates.reset_parameters()
        # self.update_gate.reset_parameters()
        # self.reset_gate.reset_parameters()
        # self.main_gate.reset_parameters()
        self.nncell.reset_parameters()

    def reset_state(self):
        super(GRUCell, self).reset_state()
        self.shared_dropout_reccer = None
        self.shared_zoneouter = None

    @property
    def state_spec(self):
        return self.outdim,

    def _forward(self, x_t, h_tm1, t=None):      # (batsize, indim), (batsize, outdim)
        if self.dropout_in:
            x_t = self.dropout_in(x_t)
        if self.dropout_rec:
            h_tm1 = self.dropout_rec(h_tm1)
        if self.shared_dropout_rec:
            if self.shared_dropout_reccer is None:
                ones = q.var(torch.ones(h_tm1.size())).cuda(crit=h_tm1).v
                self.shared_dropout_reccer = [self.shared_dropout_rec(ones)]
            h_tm1 = torch.mul(h_tm1, self.shared_dropout_reccer[0])

        h_t = self.apply_nncell(x_t, h_tm1, t=t)

        if self.zoneout:
            if self.zoner is None:
                self.zoner = q.var(torch.ones(h_t.size())).cuda(crit=h_t).v
            zoner = self.zoneout(self.zoner)
            h_t = torch.mul(1 - zoner, h_tm1) + torch.mul(zoner, h_t)
        if self.shared_zoneout:
            if self.shared_zoneouter is None:
                ones = q.var(torch.ones(h_t.size())).cuda(crit=h_t).v
                self.shared_zoneouter = [self.shared_zoneout(ones)]
            h_t = torch.mul(1 - self.shared_zoneouter[0], h_tm1) + torch.mul(self.shared_zoneouter[0], h_t)
        return h_t, h_t


class _LSTMCell(nn.Module):
    def __init__(self, indim, outdim, bias=True, gate_activation="sigmoid",
                 activation="tanh",
                 recurrent_batch_norm=None):
        super(_LSTMCell, self).__init__()
        self.indim, self.outdim, self.use_bias = indim, outdim, bias
        self.gate_activation, self.activation = gate_activation, activation        # sigm, tanh
        self.hdim = self.outdim
        if self.activation == "crelu":
            self.hdim = self.hdim // 2
        self._rbn_on = recurrent_batch_norm
        self._rbn_gates = self._rbn_on == "full"

        self.gates = PackedRNUGates(self.indim + self.outdim,
                                    [(self.outdim, self.gate_activation),
                                     (self.outdim, self.gate_activation),
                                     (self.outdim, self.gate_activation),
                                     (self.hdim, None)],
                                    use_bias=self.use_bias,
                                    rec_bn=self._rbn_gates)
        self.activation_fn = name2fn(activation)()
        self.reset_parameters()

    def reset_parameters(self):
        self.gates.reset_parameters()

    def forward(self, x_t, states, t=None):
        y_tm1, c_tm1 = states
        forget_gate, input_gate, output_gate, main_gate = self.gates(x_t, y_tm1)
        c_t = torch.mul(c_tm1, forget_gate) + torch.mul(main_gate, input_gate)
        c_t = self.activation_fn(c_t)
        y_t = torch.mul(c_t, output_gate)
        return y_t, c_t


class LSTMCell(GRUCell):
    def setcell(self):
        if self.use_cudnn_cell:
            self.nncell = nn.LSTMCell(self.indim, self.outdim, bias=self.use_bias)
        else:
            self.nncell = _LSTMCell(self.indim, self.outdim, bias=self.use_bias,
                                    gate_activation=self.gate_activation, activation=self.activation)

    @property
    def y_0(self):
        return self.get_init_states(0)[1]

    @y_0.setter
    def y_0(self, value):
        self.set_init_states(self.h_0, value)       # TODO: maybe has to be other way around

    @property
    def state_spec(self):
        return self.outdim, self.outdim

    def _forward(self, x_t, c_tm1, y_tm1, t=None):
        # region apply dropouts
        if self.dropout_in:
            x_t = self.dropout_in(x_t)
        if self.dropout_rec:
            y_tm1 = self.dropout_rec(y_tm1)
            c_tm1 = self.dropout_rec(c_tm1)
        if self.shared_dropout_rec:
            if self.shared_dropout_reccer is None:
                ones = q.var(torch.ones(c_tm1.size())).cuda(crit=c_tm1).v
                self.shared_dropout_reccer = [self.shared_dropout_rec(ones), self.shared_dropout_rec(ones)]
            y_tm1 = torch.mul(c_tm1, self.shared_dropout_reccer[0])
            c_tm1 = torch.mul(y_tm1, self.shared_dropout_reccer[1])
        # endregion
        y_t, c_t = self.apply_nncell(x_t, (y_tm1, c_tm1), t=t)
        if self.zoneout:
            if self.zoner is None:
                self.zoner = q.var(torch.ones(c_t.size())).cuda(crit=c_t).v
            zoner = self.zoneout(self.zoner)
            c_t = torch.mul(1 - zoner, c_tm1) + torch.mul(zoner, c_t)
            y_t = torch.mul(1 - zoner, y_tm1) + torch.mul(zoner, y_t)
        if self.shared_zoneout:
            if self.shared_zoneouter is None:
                ones = q.var(torch.ones(c_t.size())).cuda(crit=c_t).v
                self.shared_zoneouter = [self.shared_zoneout(ones), self.shared_zoneout(ones)]
            c_t = torch.mul(1 - self.shared_zoneouter[0], c_tm1) + torch.mul(self.shared_zoneouter[0], c_t)
            y_t = torch.mul(1 - self.shared_zoneouter[1], y_tm1) + torch.mul(self.shared_zoneouter[1], y_t)
        return y_t, c_t, y_t

# endregion

class Recurrent(object):
    pass


class RNNLayer(nn.Module, Recurrent):
    """
    Unrolling an RNN cell over timesteps of a sequence
    """
    def __init__(self, cell):
        super(RNNLayer, self).__init__()
        self.cell = cell
        self.result = set()     # "all", "final"

    def return_all(self):
        self.result.add("all")
        return self

    def return_final(self):
        self.result.add("final")
        return self

    def forward(self, x, mask=None, init_states=None, reverse=False):       # (batsize, seqlen, indim), (batsize, seqlen), [(batsize, hdim)]
        batsize = x.size(0)
        if init_states is not None:
            if not issequence(init_states):
                init_states = (init_states,)
            self.cell.set_init_states(*init_states)
        self.cell.reset_state()
        mask = mask if mask is not None else x.mask if hasattr(x, "mask") else None
        y_list = []
        y_tm1 = None
        y_t = None
        i = x.size(1)
        while i > 0:
            t = i-1 if reverse else x.size(1) - i
            mask_t = mask[:, t].unsqueeze(1) if mask is not None else None
            x_t = x[:, t]
            cellout = self.cell(x_t, mask_t=mask_t, t=t)
            y_t = cellout
            # mask
            # if mask_t is not None:  # moved to cells (recBN is affected here)
            #     if y_tm1 is None:
            #         y_tm1 = q.var(torch.zeros(y_t.size())).cuda(crit=y_t).v
            #         if x.is_cuda: y_tm1 = y_tm1.cuda()
            #     y_t = y_t * mask_t + y_tm1 * (1 - mask_t)
            #     y_tm1 = y_t
            if "all" in self.result:
                y_list.append(y_t)
            i -= 1
        ret = tuple()
        if "final" in self.result:
            ret += (y_t,)
        if "all" in self.result:
            if reverse: y_list.reverse()
            y = torch.stack(y_list, 1)
            ret += (y,)
        if len(ret) == 1:
            return ret[0]
        elif len(ret) == 0:
            print("no output specified")
            return
        else:
            return ret


class GRULayer(RNUBase, Recurrent):
    def __init__(self, indim, outdim, use_bias=True, bidirectional=False):
        super(GRULayer, self).__init__()
        self.indim, self.outdim, self.use_bias, self.bidirectional = indim, outdim, use_bias, bidirectional
        self.nnlayer = self._nn_unit()(indim, outdim, bias=use_bias, batch_first=True, bidirectional=bidirectional, num_layers=1)

    @property
    def h_0(self):
        return self.get_init_states(0)[0]

    @h_0.setter
    def h_0(self, value):
        self.set_init_states(value)

    def _nn_unit(self):
        return nn.GRU

    def forward(self, x):
        self.reset_state()
        h_0 = self._get_init_states(x.size(0))
        y, s_t = self.nnlayer(x, h_0)
        self.set_states(s_t)
        return y

    @property
    def state_spec(self):
        if self.bidirectional:
            return (self.outdim, self.outdim)
        else:
            return (self.outdim,)

    def _get_init_states(self, arg):
        initstates = super(GRULayer, self).get_init_states(arg)
        l = []
        for initstate in initstates:
            initstate = initstate.unsqueeze(0)
            l.append(initstate)
        if len(l) > 1:
            initstates = torch.cat(l, 0)
        else:
            initstates = l[0]
        return initstates

    def set_states(self, *states):
        out = []
        for i in range(states[0].size(0)):
            out.append(states[0][i])
        super(GRULayer, self).set_states(*out)


class LSTMLayer(GRULayer):
    def _nn_unit(self):
        return nn.LSTM

    @property
    def y_0(self):
        return self.get_init_states(0)[1]

    @y_0.setter
    def y_0(self, value):
        self.set_init_states(self.h_0, value)

    @property
    def state_spec(self):
        statespec = (self.outdim, self.outdim)
        if self.bidirectional:
            statespec = statespec + statespec
        return statespec

    def _get_init_states(self, arg):
        initstate = super(LSTMLayer, self)._get_init_states(arg)
        ret = (initstate[:initstate.size(0)/2],
               initstate[initstate.size(0)/2:])
        return ret

    def set_states(self, *states):
        state = torch.cat(list(states[0]), 0)
        super(LSTMLayer, self).set_states(state)


class BiRNNLayer(nn.Module, Recurrent):
    def __init__(self, cell1, cell2, mode="cat"):       # "cat" or "sum" or ...?
        super(BiRNNLayer, self).__init__()
        self.cell1 = cell1
        self.cell2 = cell2
        self.rnnlayer1 = cell1.to_layer()
        self.rnnlayer2 = cell2.to_layer()
        self.mode = mode
        self.returns = set()        # "all", "final"

    def return_all(self):
        self.rnnlayer1.return_all()
        self.rnnlayer2.return_all()
        self.returns.add("all")
        return self

    def return_final(self):
        self.rnnlayer1.return_final()
        self.rnnlayer2.return_final()
        self.returns.add("final")
        return self

    def forward(self, x, mask=None, init_states=None):
        states1, states2 = None, None
        if init_states is not None:
            states1 = init_states[:len(init_states)//2]
            states2 = init_states[len(init_states)//2:]
        rets1 = self.rnnlayer1(x, mask=mask, init_states=states1)
        rets2 = self.rnnlayer2(x, mask=mask, init_states=states2, reverse=True)
        ret = tuple()
        if "final" in self.returns:
            if self.mode == "cat":
                ret += (torch.cat([rets1[0], rets2[0]], 1),)
            else:
                ret += rets1[0] + rets2[0]
            rets1, rets2 = rets1[1:], rets2[1:]
        if "all" in self.returns:
            if self.mode == "cat":
                ret += (torch.cat([rets1[0], rets2[0]], 2),)
            else:
                ret += rets1[0] + rets2[0]
        if len(ret) == 1:
            return ret[0]
        elif len(ret) == 0:
            return
        else:
            return ret


# region II. RNN stacks
class RecStack(RecStatefulContainer, Stack):        # contains rec statefuls, not rec stateful itself
    """
    Module containing multiple rec modules (modules that can operate on a single time step)
    """
    def forward(self, *x_t, **kw):
        y_t = x_t
        for layer in self.layers:
            layerkw = {k: v for k, v in kw.items() if k not in "mask_t t".split()}\
                if not isinstance(layer, Reccable) else kw
            y_t = layer(*y_t, **layerkw)
            if not issequence(y_t):
                y_t = tuple([y_t])
        if len(y_t) == 1:
            y_t = y_t[0]
        return y_t

    @property
    def state_spec(self):
        statespec = tuple()
        for layer in self.layers:
            if isinstance(layer, RecStateful):
                statespec += tuple(layer.state_spec)
        return statespec

    def reset_state(self):
        for layer in self.layers:
            if isinstance(layer, RecStateful):
                layer.reset_state()

    def set_init_states(self, *states):     # bottom layers first
        for layer in self.layers:
            if isinstance(layer, RecStateful):
                if len(states) == 0:        # no states left to set
                    break
                statesforlayer = states[:min(len(states), layer.numstates)]
                states = states[min(len(states), layer.numstates):]
                layer.set_init_states(*statesforlayer)

    def get_init_states(self, batsize):     # bottom layers first
        initstates = []
        for layer in self.layers:
            if isinstance(layer, RecStateful):
                layerinitstates = layer.get_init_states(batsize)
                initstates += layerinitstates
        return initstates

    def set_states(self, *states):
        assert(len(states) == len(self.numstates))
        for layer in self.layers:
            if isinstance(layer, RecStateful):
                statesforlayer = states[:layer.numstates]
                states = states[layer.numstates:]
                layer.set_states(*statesforlayer)

    def get_states(self, batsize):
        states = []
        for layer in self.layers:
            if isinstance(layer, RecStateful):
                states += layer.get_states(batsize)
        return states

    def to_layer(self):
        return RNNLayer(self)

    # TODO: test for visibility of modules and their params


class RecurrentWrapper(Recurrent, nn.Module):
    def __init__(self, block):
        super(RecurrentWrapper, self).__init__()
        self.block = block

    def forward(self, x):       # TODO: multiple inputs and outputs
        x = x.contiguous()
        batsize, seqlen = x.size(0), x.size(1)
        i = x.view(batsize * seqlen, *x.size()[2:])
        y = self.block(i)
        y = y.view(batsize, seqlen, *y.size()[1:])
        return y


class RecurrentStack(RecStack):
    def __init__(self, *layers):
        newlayers = []
        for layer in layers:
            if not isinstance(layer, Recurrent):
                layer = RecurrentWrapper(layer)
            newlayers.append(layer)
        super(RecurrentStack, self).__init__(*newlayers)

    def forward(self, *x):
        y_l = x
        for layer in self.layers:
            # print layer
            y_l = layer(*y_l)
            if not issequence(y_l):
                y_l = [y_l]
        if len(y_l) == 1:
            y_l = y_l[0]
        return y_l
