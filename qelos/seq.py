import torch
from torch import nn
from qelos.basic import DotDistance, CosineDistance, ForwardDistance, BilinearDistance, TrilinearDistance, Softmax, Lambda
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


class Decoder(nn.Module):
    """
    Takes some context and decodes a sequence
    Should support (partial) teacher forcing
    """
    def __init__(self, decodercell):
        """
        :param decodercell: main block that generates. must include everything
        """
        super(Decoder, self).__init__()
        assert(isinstance(decodercell, DecoderCell))
        self.block = decodercell

    def forward(self, *x, **kw):  # first input must be (batsize, seqlen,...)
        self.block.reset_state()
        batsize = x[0].size(0)
        maxtime = x[0].size(1) if "maxtime" not in kw else kw["maxtime"]
        new_init_states = self.block._compute_init_states(*x, **kw)
        if new_init_states is not None:
            self.block.set_init_states(*new_init_states)
        y_list = []
        y_t = None
        for t in range(maxtime):
            #x_t = [x_e[:, t] if x_e.sequence else x_e for x_e in x]
            x_t = self.block._get_inputs_t(t=t, x=x, y_t=y_t)        # let the Rec definition decide what to input
            if not issequence(x_t):
                x_t = [x_t]
            x_t = tuple(x_t)
            blockret = self.block(*x_t, t=t)
            if not issequence(blockret):
                blockret = [blockret]
            y_t = blockret
            #y_t = [y_t_e.unsqueeze(1) for y_t_e in blockret[:self.block.numstates]]
            y_list.append(y_t)
        y = []
        for i in range(len(y_list[0])):
            yl_e = [y_list[j][i] for j in range(len(y_list))]
            y.append(torch.stack(yl_e, 1))
        return tuple(y)


class DecoderCell(Reccable):
    """
    Decoder logic.
    Call .to_decoder() to get decoder.
    Two ways to make a new decoder architecture:
        * subclass this and override forward(), get_inputs_t() and compute_init_states()
        * set modules/functions for the three pieces by using the provided setters (overrides subclassing)
    """
    _teacher_unforcing_support = False                  # OVERRIDE THIS  to enable teacher unforcing args

    def __init__(self, *layers):
        super(DecoderCell, self).__init__()
        if len(layers) == 1:
            self.set_core(layers[0])
        elif len(layers) > 1:
            self._core = RecStack(*layers)
        else:
            self._core = None
        self.teacher_force = 1
        self._init_state_computer = None
        self._inputs_t_getter = None

    # region forward reccable calls to forwarder
    def reset_state(self):
        self._core.reset_state()

    def set_init_states(self, *states):
        self._core.set_init_states(*states)
    # endregion

    def teacher_force(self, frac=1):        # set teacher forcing
        if not self._teacher_unforcing_support and frac < 1:
            raise NotImplementedError("only teacher forcing supported")
        if frac < 0 or frac > 1:
            raise Exception("bad argument, must be [0, 1]")
        self.teacher_force = frac

    def forward(self, *x, **kw):                        # OVERRIDE THIS
        """
        Must be implemented in all real decoder cells.
        :param x: inputs to this timestep (list of tensors) and states
        :param kw: more arguments, might include time step as t=
        :return: outputs of one decoding timestep (list of tensors)
        """
        return self._core(*x, **kw)

    def set_core(self, reccable):
        assert(isinstance(reccable, Reccable))
        self._core = reccable

    def _get_inputs_t(self, t, x, y_t):
        if self._inputs_t_getter is None:
            return self.get_inputs_t(t, x, y_t)
        else:
            return self._inputs_t_getter(t, x, y_t)

    def get_inputs_t(self, t, x, y_t):
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
        return x[0][:, t]

    def set_inputs_t_getter(self, callabla):
        self._inputs_t_getter = callabla

    def _compute_init_states(self, *x, **kw):
        if self._init_state_computer is None:
            return self.compute_init_states(*x, **kw)
        else:
            return self._init_state_computer(*x, **kw)

    def compute_init_states(self, *x, **kw):
        """
        Compute new initial states for the reccable elements in the decoder's rec stack
        :param x: the original inputs
        :return: possibly incomplete list of initial states to set, or None
        """
        return None

    def set_init_states_computer(self, callabla):
        self._init_state_computer = callabla

    def to_decoder(self):
        """ Makes a decoder from this decoder cell """
        return Decoder(self)


class ContextDecoderCell(DecoderCell):
    def __init__(self, embedder=None, *layers):
        super(ContextDecoderCell, self).__init__(*layers)
        self.embedder = embedder

    def forward(self, x, ctx, **kw):
        if self.embedder is not None:
            emb = self.embedder(x)
        else:
            emb = x
        inp = torch.cat([emb, ctx], 1)
        ret = self._core(inp)
        return ret

    def get_inputs_t(self, t, x, y_t):
        return x[0][:, t], x[1]


class AttentionDecoderCell(DecoderCell):
    """
    Recurrence of decoder with attention    # TODO
    """
    def __init__(self, attention,
                 embedder=None,
                 core=None,
                 ):
        super(AttentionDecoderCell, self).__init__()
        self.attention = attention
        self.embedder = embedder
        self.innercore = core

    # region implement DecoderCell signature
    def forward(self, x, ctx, ctxmask, **kw):
        pass

    def get_inputs_t(self, t, x, y_t):
        pass

    def compute_init_states(self, *x, **kw):
        pass

    def reset_state(self):
        pass

    def set_init_states(self, *states):
        pass
    # endregion

