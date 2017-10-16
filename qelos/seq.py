import torch
from torch import nn

from qelos.basic import DotDistance, CosineDistance, ForwardDistance, BilinearDistance, TrilinearDistance, Softmax, Lambda
from qelos.rnn import RecStack, Reccable, RecStatefulContainer, RecStateful, RecurrentStack, RecurrentWrapper
from qelos.util import issequence
from qelos.qutils import var, Hyperparam
from qelos.exceptions import SumTingWongException


# region attention
class AttentionGenerator(nn.Module):
    def __init__(self, dist=None, normalizer=Softmax(),
                 data_selector=None, scale=1., dropout=0.):
        super(AttentionGenerator, self).__init__()
        self.dist = dist
        self.data_selector = data_selector
        self.normalizer = normalizer
        self.dropout = nn.Dropout(p=dropout) if dropout > 0. else None
        self.scale = scale

    def forward(self, data, crit, mask=None):   # should work for 3D/2D and 3D/3D
        if self.data_selector is not None:
            data = self.data_selector(data)
        scores = self.dist(data, crit)      # (batsize, seqlen)
        if scores.dim() == 3:       # (batsize, dseqlen, cseqlen)
            assert(crit.dim() == 3)
            scores = scores.permute(0, 2, 1)        # because scores for 3D3D are given from data to crit, here we need from crit to data
            if mask is not None and mask.dim() == 2:
                mask = mask.unsqueeze(1).repeat(1, scores.size(1), 1)
        if mask is not None:
            assert(mask.size() == scores.size(), "mask should be same size as scores")
            scores.data.masked_fill_((-1*mask+1).byte().data, -float("inf"))
        if self.scale != 1.:
            scores = scores / self.scale
        weights = self.normalizer(scores)
        if self.dropout is not None:
            weights = self.dropout(weights)
        return weights      # (batsize, dseqlen) or (batsize, cseqlen, dseqlen)


class AttentionConsumer(nn.Module):
    def __init__(self, data_selector=None):
        super(AttentionConsumer, self).__init__()
        self.data_selector = data_selector

    def forward(self, data, weights):       # weights can (batsize, seqlen) or (batsize, cseqlen, seqlen)
        if self.data_selector is not None:
            data = self.data_selector(data)
        if weights.dim() == 3:
            data = data.unsqueeze(1)        # (batsize, 1, seqlen, dim)
        weights = weights.unsqueeze(-1)     # (batsize, seqlen, 1) or (batsize, cseqlen, seqlen, 1)
        ret = data * weights
        return torch.sum(ret, -2)


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.attgen = AttentionGenerator()
        self.attcon = AttentionConsumer()

    def split_data(self):       # splits datasets in two along dim axis, one goes to gen, other to cons
        def attgen_ds(data):        # (batsize, seqlen, dim)
            return data[:, :, :data.size(2)//2]
        def attcon_ds(data):
            return data[:, :, data.size(2)//2:]
        self.attgen.data_selector = attgen_ds
        self.attcon.data_selector = attcon_ds
        return self

    def scale(self, scale):
        self.attgen.scale = scale
        return self

    def dropout(self, rate):
        self.attgen.dropout = nn.Dropout(rate)
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
        #assert(isinstance(decodercell, DecoderCell))
        self.block = decodercell

    def reset_state(self):
        self.block.reset_state()

    def _compute_init_states(self, *x, **kw):
        return self.block._compute_init_states(*x, **kw)

    def set_init_states(self, *x):
        self.block.set_init_states(*x)

    def forward(self, *x, **kw):  # first input must be (batsize, seqlen,...)
        self.reset_state()
        batsize = x[0].size(0)
        if "maxtime" in kw:
            maxtime = kw["maxtime"]
        elif not hasattr(self.block, "_max_time") or self.block._max_time is None:
            maxtime = x[0].size(1)
        else:
            maxtime = self.block._max_time
        new_init_states = self._compute_init_states(*x, **kw)
        if new_init_states is not None:
            if not issequence(new_init_states):
                new_init_states = (new_init_states,)
            self.set_init_states(*new_init_states)
        y_list = []
        y_t = None
        for t in range(maxtime):
            #x_t = [x_e[:, t] if x_e.sequence else x_e for x_e in x]
            x_t, x_t_kw = self.block._get_inputs_t(t=t, x=x, xkw=kw, y_t=y_t)        # let the Rec definition decide what to input
            if not issequence(x_t):
                x_t = [x_t]
            x_t = tuple(x_t)
            x_t_kw["t"] = t
            blockret = self.block(*x_t, **x_t_kw)
            if not issequence(blockret):
                blockret = [blockret]
            y_t = blockret
            #y_t = [y_t_e.unsqueeze(1) for y_t_e in blockret[:self.block.numstates]]
            y_list.append(y_t)
        y = tuple()
        for i in range(len(y_list[0])):
            yl_e = [y_list[j][i] for j in range(len(y_list))]
            y += (torch.stack(yl_e, 1),)
        if len(y) == 1:
            y = y[0]
        return y


class ContextDecoder(Decoder):
    """
    Allows to use efficient cudnn RNN unrolled over time
    """
    def __init__(self, embedder=None, core=None, **kw):
        assert(core is not None)
        super(ContextDecoder, self).__init__(core)
        self.embedder = embedder
        self.ctx_to_decinp = kw["ctx_to_decinp"] if "ctx_to_decinp" in kw else True
        self.init_state_gen = kw["ctx_to_h0"] if "ctx_to_h0" in kw else None

    def forward(self, x, ctx):
        """
        :param x:   (batsize, seqlen) of integers or (batsize, seqlen, dim) of vectors if embedder is None
        :param ctx: (batsize, dim) of context vectors
        :return:
        """
        new_init_states = self._compute_init_states(x, ctx)
        if new_init_states is not None:
            if not issequence(new_init_states):
                new_init_states = (new_init_states,)
            self.set_init_states(*new_init_states)
        x_emb = self.embedder(x) if self.embedder is not None else x
        if self.ctx_to_decinp:
            ctx = ctx.unsqueeze(1).repeat(1, x_emb.size(1), 1)
            i = torch.cat([x_emb, ctx], 2)
        else:
            i = x_emb
        y = self.block(i)
        return y

    def _compute_init_states(self, x, ctx):
        if self.init_state_gen is not None:
            h_0 = self.init_state_gen(ctx)
            return h_0
        else:
            return None


class AttentionDecoder(ContextDecoder):
    def __init__(self, attention=None,
                 embedder=None,
                 core=None,                 # RecurrentStack
                 smo=None,                  # non-rec
                 att_transform=None,
                 init_state_gen=None,
                 ctx_to_smo=True,
                 state_to_smo=True,
                 decinp_to_att=False,
                 decinp_to_smo=False,
                 return_out=True,
                 return_att=False):
        super(AttentionDecoder, self).__init__(embedder=embedder, core=core, ctx_to_h0=init_state_gen)
        self.attention = attention
        self.smo = RecurrentWrapper(smo) if smo is not None else None
        self.att_transform = RecurrentWrapper(att_transform) if att_transform is not None else None
        # wiring
        self.att_after_update = True
        self.ctx_to_smo = ctx_to_smo
        self.state_to_smo = state_to_smo
        self.decinp_to_att = decinp_to_att
        self.decinp_to_smo = decinp_to_smo
        # returns
        self.return_out = return_out
        self.return_att = return_att

    def forward(self, x, ctx, ctxmask=None):
        """
        :param x:   (batsize, seqlen) of integers or (batsize, seqlen, dim) of vectors if embedder is None
        :param ctx: (batsize, dim) of context vectors
        :return:
        """
        new_init_states = self._compute_init_states(x, ctx)
        if new_init_states is not None:
            if not issequence(new_init_states):
                new_init_states = (new_init_states,)
            self.set_init_states(*new_init_states)
        x_emb = self.embedder(x) if self.embedder is not None else x
        y = self.block(x_emb)
        toatt = y
        if self.decinp_to_att:
            toatt = torch.cat([y, x_emb], 2)
        dctx, att_weights = self._get_dctx(ctx, ctxmask, toatt)
        cat_to_smo = []
        if self.state_to_smo:   cat_to_smo.append(y)
        if self.ctx_to_smo:     cat_to_smo.append(dctx)
        if self.decinp_to_smo:  cat_to_smo.append(x_emb)
        smoinp = torch.cat(cat_to_smo, 2) if len(cat_to_smo) > 1 else cat_to_smo[0]
        output = self.smo(smoinp) if self.smo is not None else smoinp
        # returns
        ret = tuple()
        if self.return_out:
            ret += (output,)
        if self.return_att:
            ret += (att_weights,)
        if len(ret) == 1:
            ret = ret[0]
        return ret

    def _get_dctx(self, ctx, ctxmask, toatt):
        """
        :param ctx:     (batsize, inpseqlen, dim)
        :param ctxmask: (batsize, inpseqlen)
        :param toatt:   (batsize, outseqlen, dim)
        :return:        (batsize, outseqlen, dim) and (batsize, outseqlen, inpseqlen)
        """
        if self.att_transform is not None:
            toatt = self.att_transform(toatt)
        att_weights = self.attention.attgen(ctx, toatt, mask=ctxmask)
        res = self.attention.attcon(ctx, att_weights)
        return res, att_weights


        pass # TODO


class Argmaxer(nn.Module):
    def forward(self, x):       # (batsize, vocsize)
        maxes, argmaxes = torch.max(x[0], 1)
        return argmaxes


class DecoderCell(RecStatefulContainer):
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
            self.core = RecStack(*layers)
        else:
            self.core = None
        self._teacher_force = 1
        self._teacher_unforced = False
        self._y_tm1_to_x_t = None
        self._max_time = None
        self._start_symbols = None
        self._init_state_computer = None
        self._inputs_t_getter = None

    # region RecStatefulContainer signature
    def reset_state(self):
        self.core.reset_state()

    def set_init_states(self, *states):
        self.core.set_init_states(*states)

    def get_init_states(self, batsize):
        return self.core.get_init_states(batsize)
    # endregion

    def teacher_force(self, frac=1, block=Argmaxer()):        # set teacher forcing
        """
        Can only be called on blocks that support teacher unforcing (_teacher_unforce_support)
        :param frac: mixture between teacher forcing and own decoding
        :param block: how to transform previous outputs to new inputs
        :return:
        """
        if not self._teacher_unforcing_support and frac < 1:
            raise NotImplementedError("only teacher forcing supported")
        if frac < 0 or frac > 1:
            raise Exception("bad argument, must be [0, 1]")
        self._teacher_force = frac
        self._y_tm1_to_x_t = block

    def teacher_unforce(self, maxtime=None, block=Argmaxer(), startsymbols=None):
        self.teacher_force(0, block=block)
        self._teacher_unforced = True
        self._max_time = maxtime
        self._start_symbols = startsymbols

    def forward(self, *x, **kw):                        # OVERRIDE THIS
        """
        Must be implemented in all real decoder cells.
        :param x: inputs to this timestep (list of tensors) and states
        :param kw: more arguments, might include time step as t=
        :return: outputs of one decoding timestep (list of tensors)
        """
        return self.core(*x, **kw)

    def set_core(self, reccable):
        assert(isinstance(reccable, RecStateful))
        self.core = reccable

    def _get_inputs_t(self, t=None, x=None, xkw=None, y_t=None):
        if self._inputs_t_getter is None:
            return self.get_inputs_t(t=t, x=x, xkw=xkw, y_t=y_t)
        else:
            return self._inputs_t_getter(t=t, x=x, xkw=xkw, y_t=y_t)

    def get_inputs_t(self, t=None, x=None, xkw=None, y_t=None):
        """
        Make the inputs to cell from timestep, inputs to decoder and previous outputs of cell.
        Called before every call to .forward() and must compute all arguments for .forward() for given timestep.
        Must be implemented in all concrete decoder cells.
        This method is the place to implement teacher forcing (don't forget to override _teacher_unforcing_support to True to
        enable official teacher forcing support).
        This method could also be used for computing dynamic contexts (attention)
        :param t: timestep (integer)
        :param x: original arguments to decoder (list of tensors)
        :param xkw: original kwargs to decoder
        :param y_t: previous outputs of this cell (list of tensors or None). If None, no previous outputs have been output yet
        :return: actual (inputs, kwinpts) to .forward() of this decoder cell (list of tensors)
        """
        return x[0][:, t], {"t": t}

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
    _teacher_unforcing_support = True

    def __init__(self, embedder=None, *layers):
        super(ContextDecoderCell, self).__init__(*layers)
        self.embedder = embedder
        # teacher unforce after
        self._tua_after = None

    def forward(self, x, ctx, **kw):
        if self.embedder is not None:
            emb = self.embedder(x)
        else:
            emb = x
        inp = torch.cat([emb, ctx], 1)
        ret = self.core(inp)
        return ret

    def teacher_unforce_after(self, maxlen, after, block):
        self._y_tm1_to_x_t = block
        self._max_time = maxlen
        self._tua_after = after     # hyperparam or int !

    def get_inputs_t(self, t=None, x=None, xkw=None, y_t=None):
        if "teacherforce_mask" in xkw:
            teacherforce_mask = xkw["teacherforce_mask"]
            teacherforce_mask_t = teacherforce_mask[:, t].long()
            teacherforced_outs = self.get_inputs_t(t=t, x=x, xkw={"teacher_force": True}, y_t=y_t)
            freex = (x[0][:, 0], x[1])
            freerunning_outs = self.get_inputs_t(t=t, x=freex, xkw={"teacher_force": False}, y_t=y_t)
            x_t = teacherforced_outs[0][0] * teacherforce_mask_t \
                  + (1 + (-1) * teacherforce_mask_t) * freerunning_outs[0][0]
            ctx = teacherforced_outs[0][1]
            return (x_t, ctx), {}
        if "teacher_force" not in xkw:
            if self._tua_after is not None:
                if isinstance(self._tua_after, Hyperparam):
                    tua_after = self._tua_after.v
                else:
                    tua_after = self._tua_after
                tua_after = max(1, self._max_time - tua_after)
                if t < tua_after:
                    teacher_force = 1
                else:
                    teacher_force = 0
            else:
                teacher_force = self._teacher_force
        else:
            teacher_force = xkw["teacher_force"]
        if teacher_force == 0:
            if y_t is None:
                assert(t == 0)
                if self._start_symbols is not None:
                    ctx = x[0]
                    if self._start_symbols.size() == 2:
                        y_t = self._start_symbols.repeat(ctx.size(0), 1)
                    else:
                        y_t = self._start_symbols.repeat(ctx.size(0))
                    # y_t = torch.LongTensor(ctx.size(0))
                    # y_t.fill_(self._start_symbols)
                    # y_t = var(y_t).cuda(ctx).v
                else:
                    y_t = x[0]
                    ctx = x[1]
            else:
                y_t = self._y_tm1_to_x_t(y_t)
                if self._start_symbols is not None:
                    ctx = x[0]
                else:
                    ctx = x[1]
            return (y_t, ctx), {}
        elif teacher_force == 1:
            return (x[0][:, t], x[1]), {}
        else:
            raise NotImplemented("partial teacher forcing not supported")


class AttentionDecoderCell(DecoderCell):
    """
    Recurrence of decoder with attention    # TODO
    """
    def __init__(self, attention=None,
                 embedder=None,
                 core=None,
                 smo=None,
                 init_state_gen=None,
                 attention_transform=None,
                 att_after_update=False,
                 ctx_to_decinp=True,
                 ctx_to_smo=True,
                 state_to_smo=True,
                 decinp_to_att=False,
                 decinp_to_smo=False,
                 return_out=True,
                 return_att=False,
                 state_split=False,
                 **kw):
        """
        Initializes attention-based decoder cell
        :param attention:           attention module
        :param embedder:            embedder module
        :param core:                core recurrent module   (recstateful)
        :param smo:                 module that takes core's output vectors and produces probabilities over output vocabulary
        :param init_state_gen:      module that generates initial states for the decoder and its core (see also .set_init_states())
        :param attention_transform: module that transforms attention vector just before generation of attention weights
        :param att_after_update:    perform recurrent step before attention
        :param ctx_to_decinp:       feed attention context to core
        :param ctx_to_smo:          feed attention context to smo
        :param state_to_smo:        feed output of core to smo
        :param decinp_to_att:       feed embedding to attention generation
        :param decinp_to_smo:       feed embedding to smo
        :param return_out:          return output probabilities
        :param return_att:          return attention weights over input sequence
        :param state_split:       split core's state, first half goes to attention (which might also be split), second half goes to smo
        :param kw:
        """
        super(AttentionDecoderCell, self).__init__(**kw)
        # submodules
        self.attention = attention
        self.embedder = embedder
        self.core = core
        self.smo = smo
        self.set_init_states_computer(init_state_gen) if init_state_gen is not None else None
        self.att_transform = attention_transform
        # wiring
        self.att_after_update = att_after_update
        self.ctx_to_decinp = ctx_to_decinp
        self.ctx_to_smo = ctx_to_smo
        self.state_to_smo = state_to_smo
        self.decinp_to_att = decinp_to_att
        self.decinp_to_smo = decinp_to_smo
        self.state_split = state_split
        # returns
        self.return_out = return_out
        self.return_att = return_att
        # states
        self._state = [None]


    # region implement DecoderCell signature
    def forward(self, x_t, ctx, ctxmask=None, t=None, outmask_t=None, **kw):
        """
        :param x_t:     (batsize,...) input for current timestep
        :param ctx:     (batsize, inpseqlen, dim) whole context
        :param ctxmask: (batsize, inpseqlen) context mask
        :param t:       current timestep
        :param kw:
        :return: output probabilities for current timestep and/or attention weights
        """
        batsize = x_t.size(0)
        x_t_emb = self.embedder(x_t)
        if len(x_t_emb) == 2 and isinstance(x_t_emb, tuple):
            x_t_emb, _ = x_t_emb
        if self.att_after_update:
            ctx_tm1 = self._state[0]
            i_t = torch.cat([x_t_emb, ctx_tm1], 1) if self.ctx_to_decinp else x_t_emb
            o_t = self.core(i_t, t=t)
            ctx_t, att_weights_t = self._get_ctx_t(ctx, ctxmask, o_t, x_t_emb)
        else:
            o_tm1 = self._state[0]
            ctx_t, att_weights_t = self._get_ctx_t(ctx, ctxmask, o_tm1, x_t_emb)
            i_t = torch.cat([x_t_emb, ctx_t], 1) if self.ctx_to_decinp else x_t_emb
            o_t = self.core(i_t, t=t)
        cat_to_smo = []
        o_to_smo = o_t[:, o_t.size(1)//2:] if self.state_split else o_t      # first half is split off in _get_ctx_t()
        if self.state_to_smo:   cat_to_smo.append(o_to_smo)
        if self.ctx_to_smo:     cat_to_smo.append(ctx_t)
        if self.decinp_to_smo:  cat_to_smo.append(x_t_emb)
        smoinp_t = torch.cat(cat_to_smo, 1) if len(cat_to_smo) > 1 else cat_to_smo[0]
        smokw = {}
        smokw.update(kw)
        if outmask_t is not None:
            smokw["mask"] = outmask_t.float()
        y_t = self.smo(smoinp_t, **smokw) if self.smo is not None else smoinp_t
        # returns
        ret = tuple()
        if self.return_out:
            ret += (y_t,)
        if self.return_att:
            ret += (att_weights_t,)
        # store rec state
        if self.att_after_update:
            self._state[0] = ctx_t
        else:
            self._state[0] = o_t
        if len(ret) == 1:
            ret = ret[0]
        return ret

    def _get_ctx_t(self, ctx, ctxmask, h, x_emb):
        """
        :param ctx:     (batsize, inpseqlen, dim) whole context
        :param ctxmask: (batsize, inpseqlen) context mask over time
        :param h:       (batsize, dim) criterion for attention
        :param x_emb:   (batsize, dim) vector of current input, used in attention if decinp_to_att==True
        :return: (summary of ctx based on attention, attention weights)
        """
        assert(ctx.dim() == 3)
        assert(ctxmask is None or ctxmask.dim() == 2)
        if self.state_split:
            h = h[:, :h.size(1)//2]
        if self.decinp_to_att:
            h = torch.cat([h, x_emb], 1)
        if self.att_transform is not None:
            h = self.att_transform(h)
        att_weights = self.attention.attgen(ctx, h, mask=ctxmask)
        res = self.attention.attcon(ctx, att_weights)
        return res, att_weights

    def get_inputs_t(self, t=None, x=None, xkw=None, y_t=None):      # TODO implement teacher forcing
        outargs = (x[0][:, t], x[1])    # (prev_token, ctx)
        outkwargs = {"t": t}
        if "ctxmask" in xkw:        # copy over ctxmask (shared over decoder steps)
            outkwargs["ctxmask"] = xkw["ctxmask"]
        if "outmask" in xkw:        # slice out the time from outmask
            outkwargs["outmask_t"] = xkw["outmask"][:, t]
        return outargs, outkwargs
    # endregion

    # region RecStatefulContainer signature
    def reset_state(self):
        #self._state[0] = None
        self.core.reset_state()

    def set_init_states(self, ownstate, *states):
        """
        :param ownstate:    treated as first context (ctx_0) if att_after_update==True,
                            treated as initial output of core (o_0) otherwise
        :param states:      (optional) states for core
        """
        self._state[0] = ownstate
        self.core.set_init_states(*states)
    # endregion

