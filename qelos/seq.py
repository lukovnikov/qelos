import torch
from torch import nn
from qelos.basic import DotDistance, CosineDistance, ForwardDistance, BilinearDistance, TrilinearDistance, Softmax
from qelos.rnn import RecStack, Reccable
from qelos.util import issequence
from torch.autograd import Variable


# region attention
class AttentionGenerator(nn.Module):
    def __init__(self, dist=None, normalizer=Softmax(), data_selector=None):
        super(AttentionGenerator, self).__init__()
        self.dist = dist
        self.data_selector = data_selector
        self.normalizer = normalizer

    def forward(self, data, crit, mask=None):
        if self.data_selector is not None:
            data = self.data_selector(data)
        scores = self.dist(data, crit)      # (batsize, seqlen)
        weights = self.normalizer(scores, mask=mask)
        return weights


class AttentionConsumer(nn.Module):
    def __init__(self, data_selector=None):
        super(AttentionConsumer, self).__init__()
        self.data_selector = data_selector

    def forward(self, data, weights):
        if self.data_selector is not None:
            data = self.data_selector(data)
        weights = weights.unsqueeze(2)
        ret = data * weights
        return torch.sum(ret, 1)


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.attgen = AttentionGenerator()
        self.attcon = AttentionConsumer()

    def split_data(self):       # splits data in two along dim axis, one goes to gen, other to cons
        def attgen_ds(data):        # (batsize, seqlen, dim)
            return data[:, :, :data.size(2)//2]
        def attcon_ds(data):
            return data[:, :, data.size(2)//2:]
        self.attgen.data_selector = attgen_ds
        self.attcon.data_selector = attcon_ds
        return self

    def forward(self, data, crit):
        weights = self.attgen(data, crit)
        summary = self.attcon(data, weights)
        return summary

    def dot_gen(self):
        self.attgen.dist = DotDistance()
        return self

    def cosine_gen(self):
        self.attgen.dist = CosineDistance()
        return self

    def forward_gen(self, ldim, rdim, aggdim, activation="tanh", use_bias=True):
        self.attgen.dist = ForwardDistance(ldim, rdim, aggdim, activation=activation, use_bias=use_bias)
        return self

    def bilinear_gen(self, ldim, rdim):
        self.attgen.dist = BilinearDistance(ldim, rdim)
        return self

    def trilinear_gen(self, ldim, rdim, aggdim, activation="tanh", use_bias=True):
        self.attgen.dist = TrilinearDistance(ldim, rdim, aggdim, activation=activation, use_bias=use_bias)
        return self
# endregion

# TODO rnn/cnn encoders, decoders, encdec, seq2seq, ptrnet, teacher forcing etc


class SeqVariable(Variable):
    sequence = True


class Decoder(nn.Module):
    """
    Takes some context and decodes a sequence
    Should support (partial) teacher forcing
    """
    def __init__(self, reccable):
        """
        :param reccable: main block that generates. must include everything
        :param init_state_gen: optional block to generate initial states
        :param teacher_force: fraction of teacher forcing
        """
        super(Decoder, self).__init__()
        assert(isinstance(reccable, Reccable))
        self.block = reccable

    def forward(self, *x, **kw):  # first input must be (batsize, seqlen,...)
        batsize = x[0].size(0)
        maxtime = x[0].size(1) if "maxtime" not in kw else kw["maxtime"]
        init_states = self.block.get_init_states(batsize, *x)
        states_t = init_states
        y_list = []
        y_t = None
        for t in range(maxtime):
            #x_t = [x_e[:, t] if x_e.sequence else x_e for x_e in x]
            x_t = self.block.get_inputs_t(t=t, x=x, y_t=y_t)        # let the Rec definition decide what to input
            if not issequence(x_t):
                x_t = [x_t]
            x_t = tuple(x_t)
            blockret = self.block(*(x_t + states_t), t=t)
            if not issequence(blockret):
                blockret = [blockret]
            y_t = blockret[:self.block.numstates]
            #y_t = [y_t_e.unsqueeze(1) for y_t_e in blockret[:self.block.numstates]]
            y_list.append(y_t)
            states_t = blockret[self.block.numstates:]
        y = []
        for i in range(len(y_list[0])):
            yl_e = [y_list[j][i] for j in range(len(y_list))]
            y.append(torch.stack(yl_e, 1))
        return tuple(y)


# implementing a new decoder just needs subclassing ADecoderCell and calling .to_decoder() on it

class ADecoderCell(Reccable):           # SUBCLASS THIS FOR A NEW DECODER
    """
    The recurrence of the decoder. Subclass this and then call .to_decoder() to get decoder
    """
    _teacher_unforcing_support = False                  # OVERRIDE THIS  to enable teacher unforcing args

    def __init__(self, reccable=None):
        super(ADecoderCell, self).__init__()
        self.block = reccable
        self.teacher_force = 1

    def teacher_force(self, frac=1):        # set teacher forcing
        if not self._teacher_unforcing_support and frac < 1:
            raise NotImplementedError("only teacher forcing supported")
        if frac < 0 or frac > 1:
            raise Exception("bad argument, must be [0, 1]")
        self.teacher_force = frac

    def forward(self, *x, **kw):                        # OVERRIDE THIS
        """
        Must be implemented in all real decoder cells.
        :param x: inputs to this timestep (list of tensors)
        :param kw: more arguments, might include time step as t=
        :return: outputs of one decoding timestep (list of tensors)
        """
        raise NotImplementedError("use subclass")

    def get_init_states(self, batsize, *x):             # OVERRIDE THIS
        """
        Must return initial states for all stateful submodules. Lower layers are first, more internal states are first
        Must be implemented in all real decoder cells.
        :param batsize: batch size (integer)
        :param x: original inputs to decoder (list of tensors)
        :return: initial states for all stateful submodules (list of tensors)
        """
        raise NotImplementedError("use subclass")

    def get_inputs_t(self, t, x, y_t):                  # OVERRIDE THIS
        """
        Make the inputs to cell from timestep, inputs to decoder and previous outputs of cell.
        Called before every call to .forward() and must compute all arguments for .forward() for given timestep.
        Must be implemented in all concrete decoder cells.
        This method is the place to implement teacher forcing (don't forget to override _teacher_unforcing_support to True to
        enable official teacher forcing support).
        This method could also be used for computing dynamic contexts (attention)
        :param t: timestep (integer)
        :param x: original inputs to decoder (list of tensors)
        :param y_t: previous outputs of this cell (list of tensors or None). If None, no previous outputs have been output yet
        :return: actual inputs to .forward() of this decoder cell (list of tensors)
        """
        raise NotImplementedError("use subclass")

    @property
    def state_spec(self):
        """ Reuse internal block. Override if introducing additional stateful components """
        return self.block.state_spec

    def to_decoder(self):
        """ Makes a decoder from this decoder cell """
        return Decoder(self)


class SimpleDecoderCell(ADecoderCell):
    """
    Simple decoder not relying on any context.
    """
    def forward(self, x_t, *states, **kw):
        """
        :param x_t: (batsize, ...), input for current timestep
        :param t: (possible) timestep
        :param kw:
        :return: (batsize, ...), output for current timestep
        """
        return self.block(x_t, *states)

    def get_inputs_t(self, t, x, y_t):
        """
        :param t: timestep
        :param x: (batsize, seqlen, ...) input sequence
        :param y_t: (batsize, ...) output
        :return: x_t, (batsize, ...), sliced from x using t
        """
        return x[0][:, t]

    def get_init_states(self, batsize, *x):
        return self.block.get_init_states(batsize)  # blank initial states, no context


class ContextDecoderCell(ADecoderCell):
    """
    Decoder based on static context     # TODO
    """


class AttentionDecoderCell(ADecoderCell):
    """
    Recurrence of decoder with attention    # TODO
    """

