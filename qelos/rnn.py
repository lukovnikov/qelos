import torch
from torch.autograd import Variable
from torch import nn
from qelos.util import name2fn, issequence, isnumber


# region I. RNN cells
class RNUGate(nn.Module):

    def __init__(self, indim, outdim, hdim=None, activation="sigmoid", use_bias=True):
        super(RNUGate, self).__init__()
        self.indim, self.outdim, self.activation, self.use_bias = indim, outdim, activation, use_bias
        self.activation_fn = name2fn(self.activation)
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


class Reccable(nn.Module):
    @property
    def state_spec(self):
        raise NotImplementedError("use subclass")

    @property
    def numstates(self):
        return len(self.state_spec)


class RNUBase(nn.Module):
    def to_layer(self):
        return RNNLayer(self)

    def get_init_states(self, arg):
        if isnumber(arg):   # batsize --> generate
            ret = []
            for statespec in self.state_spec:
                state_0 = Variable(torch.zeros((arg, statespec)))
                ret.append(state_0)
        else:
            raise NotImplementedError("can't handle other args yet")
        return ret


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

    def forward(self, x_t, h_tm1, t=None):
        if self.dropout_in:
            x_t = self.dropout_in(x_t)
        if self.dropout_rec:
            h_tm1 = self.dropout_rec(h_tm1)
        h_t = self.main_gate(x_t, h_tm1)
        if self.zoneout:
            zoner = Variable(torch.ones(h_t.size()))
            zoner = self.zoneout(zoner)
            h_t = torch.mul(1 - zoner, h_tm1) + torch.mul(zoner, h_t)
        return h_t, h_t


class GRU(RNUBase):
    debug = False

    def __init__(self, indim, outdim, use_bias=True, activation="tanh", gate_activation="sigmoid",
                 dropout_in=None, dropout_rec=None, zoneout=None):
        super(GRU, self).__init__()
        self.indim, self.outdim, self.use_bias, self.dropout_in, self.dropout_rec, self.zoneout = \
            indim, outdim, use_bias, dropout_in, dropout_rec, zoneout
        self.gate_activation, self.activation = gate_activation, activation        # sigm, tanh

        self.update_gate = RNUGate(self.indim, self.outdim, activation=gate_activation, use_bias=use_bias)
        self.reset_gate = RNUGate(self.indim, self.outdim, activation=gate_activation, use_bias=use_bias)
        self.main_gate = RNUGate(self.indim, self.outdim, activation=activation, use_bias=use_bias)

        if self.dropout_in:
            self.dropout_in = nn.Dropout(p=self.dropout_in)
        if self.dropout_rec:
            self.dropout_rec = nn.Dropout(p=self.dropout_rec)
        if self.zoneout:
            self.zoneout = nn.Dropout(p=self.zoneout)

        self.reset_parameters()

    def reset_parameters(self):
        self.update_gate.reset_parameters()
        self.reset_gate.reset_parameters()
        self.main_gate.reset_parameters()

    @property
    def state_spec(self):
        return self.outdim,

    def forward(self, x_t, h_tm1, t=None):      # (batsize, indim), (batsize, outdim)
        if self.dropout_in:
            x_t = self.dropout_in(x_t)
        if self.dropout_rec:
            h_tm1 = self.dropout_rec(h_tm1)
        reset_gate = self.reset_gate(x_t, h_tm1, debug=self.debug)
        update_gate = self.update_gate(x_t, h_tm1, debug=self.debug)
        if self.debug:  reset_gate, rg = reset_gate; update_gate, ug = update_gate
        canh = torch.mul(h_tm1, reset_gate)
        canh = self.main_gate(x_t, canh)
        if self.zoneout:
            update_gate = self.zoneout(update_gate)
        h_t = (1 - update_gate) * h_tm1 + update_gate * canh
        if self.debug: return h_t, rg, ug
        return h_t, h_t


class LSTM(RNUBase):
    debug = False

    def __init__(self, indim, outdim, use_bias=True, activation="tanh", gate_activation="sigmoid",
                 dropout_in=None, dropout_rec=None, zoneout=None):
        super(LSTM, self).__init__()
        self.indim, self.outdim, self.use_bias, self.dropout_in, self.dropout_rec, self.zoneout = \
            indim, outdim, use_bias, dropout_in, dropout_rec, zoneout

        self.gate_activation, self.activation = gate_activation, activation  # sigm, tanh
        self.activation_fn = name2fn(activation)

        self.forget_gate = RNUGate(self.indim, self.outdim, activation=gate_activation, use_bias=use_bias)
        self.input_gate = RNUGate(self.indim, self.outdim, activation=gate_activation, use_bias=use_bias)
        self.output_gate = RNUGate(self.indim, self.outdim, activation=gate_activation, use_bias=use_bias)
        self.main_gate = RNUGate(self.indim, self.outdim, activation=activation, use_bias=use_bias)

        # region dropouts
        if self.dropout_in:
            self.dropout_in = nn.Dropout(p=self.dropout_in)
        if self.dropout_rec:
            self.dropout_rec = nn.Dropout(p=self.dropout_rec)
        if self.zoneout:
            self.zoneout = nn.Dropout(p=self.zoneout)
        # endregion

        self.reset_parameters()

    def reset_parameters(self):
        self.forget_gate.reset_parameters()
        self.input_gate.reset_parameters()
        self.output_gate.reset_parameters()
        self.main_gate.reset_parameters()

    @property
    def state_spec(self):
        return self.outdim, self.outdim

    def forward(self, x_t, c_tm1, y_tm1, t=None):
        # region apply dropouts
        if self.dropout_in:
            x_t = self.dropout_in(x_t)
        if self.dropout_rec:
            y_tm1 = self.dropout_rec(y_tm1)
            c_tm1 = self.dropout_rec(c_tm1)
        # endregion
        input_gate = self.input_gate(x_t, y_tm1)
        output_gate = self.output_gate(x_t, y_tm1)
        forget_gate = self.forget_gate(x_t, y_tm1)
        main_gate = self.main_gate(x_t, y_tm1)
        c_t = torch.mul(c_tm1, forget_gate) + torch.mul(main_gate, input_gate)
        y_t = torch.mul(self.activation_fn(c_t), output_gate)
        if self.zoneout:
            zoner = Variable(torch.ones(c_t.size()))
            zoner = self.zoneout(zoner)
            c_t = torch.mul(1 - zoner, c_tm1) + torch.mul(zoner, c_t)
            y_t = torch.mul(1 - zoner, y_tm1) + torch.mul(zoner, y_t)
        return y_t, c_t, y_t

# endregion

class RNNLayer(nn.Module):
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
        states = init_states if init_states is not None else self.cell.get_init_states(batsize)
        mask = mask if mask is not None else x.mask if hasattr(x, "mask") else None
        y_list = []
        y_tm1 = None
        y_t = None
        i = x.size(1)
        while i > 0:
            t = i-1 if reverse else x.size(1) - i
            x_t = x[:, t]
            cellout = self.cell(x_t, *states, **{"t":t})
            y_t = cellout[0]
            if y_tm1 is None and mask is not None:
                y_tm1 = Variable(torch.zeros(y_t.size()))
                if x.is_cuda: y_tm1 = y_tm1.cuda()
            newstates = cellout[1:]
            # mask
            if mask is not None:
                mask_t = mask[:, t].unsqueeze(1)
                y_t = y_t * mask_t + y_tm1 * (1 - mask_t)
                y_tm1 = y_t
                st = []
                for newstate, oldstate in zip(newstates, states):
                    newstate = newstate * mask_t + oldstate * (1 - mask_t)
                    st.append(newstate)
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


class BiRNNLayer(nn.Module):
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
class RecStack(Reccable):
    """
    Module containing multiple rec modules (modules that can operate on a single time step)
    """
    def __init__(self):
        super(RecStack, self).__init__()
        self.layers = []

    def add_layer(self, x):
        self.layers.append(x)

    def forward(self, x_t, *states, **kwargs):
        y_t = x_t
        newstates = []
        for layer in self.layers:
            if isinstance(layer, Reccable) and layer.numstates > 0:
                layerstates = states[:layer.numstates]
                states = states[layer.numstates:]
                layer_ret = layer(y_t, *layerstates, **kwargs)
                if not issequence(layer_ret):
                    layer_ret = [layer_ret]
                y_t = layer_ret[0]
                newstates += list(layer_ret[1:])
        return tuple([y_t] + newstates)

    @property
    def state_spec(self):
        statespec = tuple()
        for layer in self.layers:
            if hasattr(layer, "state_spec"):
                statespec += tuple(layer.state_spec)
        return statespec

    def get_init_states(self, arg):
        initstates = tuple()
        if isnumber(arg):       # batsize
            for layer in self.layers:
                if hasattr(layer, "get_init_states"):
                    initstates += tuple(layer.get_init_states(arg))
        else:
            raise NotImplementedError("other args not supported yet")
        return initstates

    def to_layer(self):
        return RNNLayer(self)


# TODO port attention, decoder, seq2seq, ptrnet,
