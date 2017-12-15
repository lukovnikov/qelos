import torch
from torch import nn
from qelos.basic import DotDistance, CosineDistance, ForwardDistance, BilinearDistance, TrilinearDistance, Softmax, Lambda, Stack
from qelos.rnn import RecStack, Reccable, RecStatefulContainer, RecStateful, RecurrentStack, RecurrentWrapper
from qelos.util import issequence, getkw
from qelos.qutils import var, intercat


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
            data = self.data_selector(data).contiguous()
        scores = self.dist(data, crit)      # (batsize, seqlen)
        if scores.dim() == 3:       # (batsize, dseqlen, cseqlen)
            assert(crit.dim() == 3)
            scores = scores.permute(0, 2, 1)        # because scores for 3D3D are given from data to crit, here we need from crit to data
            if mask is not None and mask.dim() == 2:
                mask = mask.unsqueeze(1).repeat(1, scores.size(1), 1)
        if mask is not None:
            assert(mask.size() == scores.size(), "mask should be same size as scores")
            infmask = var(torch.zeros(mask.size())).cuda(mask).v
            infmask.data.masked_fill_((-1 * mask + 1).byte().data, -float("inf"))
            scores = scores + infmask
            # scores.data.masked_fill_((-1*mask+1).byte().data, -float("inf"))
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
        batsize = x[0].size(0) if "batsize" not in kw else kw["batsize"]
        maxtime = x[0].size(1) if "maxtime" not in kw else kw["maxtime"]
        self.block.initialize(*x, **kw)
        if not isinstance(self.block, ModularDecoderCell):  # ModularDecoderCell must initialize by itself
            new_init_states = self._compute_init_states(*x, **kw)
            if new_init_states is not None:
                if not issequence(new_init_states):
                    new_init_states = (new_init_states,)
                self.set_init_states(*new_init_states)
        y_list = []
        y_t = None
        t = 0
        stopdecoding = False
        while True:
            #x_t = [x_e[:, t] if x_e.sequence else x_e for x_e in x]
            x_t, x_t_kw = self.block._get_inputs_t(t=t, x=x, xkw=kw, y_t=y_t)        # let the Rec definition decide what to input
            stopdecoding |= getkw(x_t_kw, "_stop", default=False)
            if not issequence(x_t):
                x_t = [x_t]
            x_t = tuple(x_t)
            x_t_kw["t"] = t
            blockret = self.block(*x_t, **x_t_kw)
            if isinstance(blockret, tuple) and len(blockret) == 2 and isinstance(blockret[1], dict):
                blockret_kw = blockret[1]
                stopdecoding |= getkw(blockret_kw, "_stop", default=False)
                blockret = blockret[0]
            if not issequence(blockret):
                blockret = [blockret]
            y_t = blockret
            #y_t = [y_t_e.unsqueeze(1) for y_t_e in blockret[:self.block.numstates]]
            y_list.append(y_t)
            t += 1
            if (maxtime is not None and t == maxtime) or stopdecoding:
                break
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
        self.teacher_force = 1
        self._init_state_computer = None
        self._inputs_t_getter = None

        self._sparse_outmask = None

    # region RecStatefulContainer signature
    def reset_state(self):
        self.core.reset_state()
        if self._inputs_t_getter is not None and hasattr(self._inputs_t_getter, "reset"):
            self._inputs_t_getter.reset()

    def set_init_states(self, *states):
        self.core.set_init_states(*states)

    def get_init_states(self, batsize):
        return self.core.get_init_states(batsize)
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

    def set_runner(self, runner):
        self.set_inputs_t_getter(runner)

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

    def initialize(self, *x, **kw):
        pass


class ContextDecoderCell(DecoderCell):
    """ static context decoding cell """
    def __init__(self, embedder=None, *layers):
        super(ContextDecoderCell, self).__init__(*layers)
        self.embedder = embedder

    def forward(self, x, ctx, **kw):
        if self.embedder is not None:
            if isinstance(self.embedder, nn.Embedding):
                emb = self.embedder(x)
            else:
                emb, mask = self.embedder(x)
        else:
            emb = x
        inp = torch.cat([emb, ctx], 1)
        ret = self.core(inp, **kw)
        return ret

    def get_inputs_t(self, t=None, x=None, xkw=None, y_t=None):
        outargs = (x[0][:, t], x[1])  # (prev_token, ctx)
        outkwargs = {"t": t}
        if "outmask" in xkw:  # slice out the time from outmask
            outmask = xkw["outmask"]
            if outmask.dim() == 1:  # get mask from data stored on this object
                assert (self._sparse_outmask is not None)
                outmaskaddrs = list(outmask.cpu().data.numpy())
                sparse_outmask_t = [self._sparse_outmask[a][t] for a in outmaskaddrs]
                sparse_outmask_t = [torch.from_numpy(a.todense()).t() for a in sparse_outmask_t]
                sparse_outmask_t = torch.cat(sparse_outmask_t, 0)
                outmask_t = var(sparse_outmask_t).cuda(outmask).v
            elif outmask.data[0, 0, 1] > 1:  # batchable sparse
                vocsize = outmask.data[0, 0, 1]
                outmask_t = var(torch.ByteTensor(outmask.size(0), vocsize + 1)).cuda(outmask).v
                outmask_t.data.fill_(0)
                outmask_t.data.scatter_(1, outmask.data[:, t, 2:], 1)
                outmask_t.data = outmask_t.data[:, 1:]
                # outmask_t = var(outmask_t).cuda(outmask).v
            else:
                outmask_t = outmask[:, t]
            outkwargs["outmask_t"] = outmask_t
        return outargs, outkwargs


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
        if state_split:
            self.attention.split_data()
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
            if self.state_split and t == 0:
                ctx_tm1 = ctx_tm1[:, ctx_tm1.size(1)//2:]       # equivalent is handled by attention splitter
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
            outmask = xkw["outmask"]
            if outmask.dim() == 1:       # get mask from data stored on this object
                assert(self._sparse_outmask is not None)
                outmaskaddrs = list(outmask.cpu().data.numpy())
                sparse_outmask_t = [self._sparse_outmask[a][t] for a in outmaskaddrs]
                sparse_outmask_t = [torch.from_numpy(a.todense()).t() for a in sparse_outmask_t]
                sparse_outmask_t = torch.cat(sparse_outmask_t, 0)
                outmask_t = var(sparse_outmask_t).cuda(outmask).v
            elif outmask.data[0, 0, 1] > 1:        # batchable sparse
                vocsize = outmask.data[0, 0, 1]
                outmask_t = var(torch.ByteTensor(outmask.size(0), vocsize+1)).cuda(outmask).v
                outmask_t.data.fill_(0)
                outmask_t.data.scatter_(1, outmask.data[:, t, 2:], 1)
                outmask_t.data = outmask_t.data[:, 1:]
                # outmask_t = var(outmask_t).cuda(outmask).v
            else:
                outmask_t = outmask[:, t]
            outkwargs["outmask_t"] = outmask_t
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


class HierarchicalAttentionDecoderCell(AttentionDecoderCell):
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
                 structure_tokens=(0, 1),
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
        :param state_split:         split core's state, first half goes to attention (which might also be split), second half goes to smo
        :param structure_tokens:    TODO
        :param kw:
        """
        super(HierarchicalAttentionDecoderCell, self).__init__(attention=attention, embedder=embedder,
            core=core, smo=smo, init_state_gen=init_state_gen, attention_transform=attention_transform,
            att_after_update=att_after_update, ctx_to_decinp=ctx_to_decinp, ctx_to_smo=ctx_to_smo,
            state_to_smo=state_to_smo, decinp_to_att=decinp_to_att, decinp_to_smo=decinp_to_smo,
            return_att=return_att, return_out=return_out, state_split=state_split, **kw)
        # hierarchical
        self._branch_token = structure_tokens[0]        # actually ID
        self._join_token = structure_tokens[1]
        self._state_stack = None

    def forward(self, x_t, ctx, ctxmask=None, t=None, outmask_t=None, **kw):
        # check where in x_t are branch tokens and save the right states from the core
        batsize = x_t.size(0)

        bt = self._branch_token
        if not issequence(bt):
            bt = [bt]
        branch_mask = None
        for bte in bt:
            branch_mask_i = x_t != bte
            if branch_mask is None:
                branch_mask = branch_mask_i
            else:
                branch_mask = branch_mask * branch_mask_i

        jt = self._join_token
        if not issequence(jt):
            jt = [jt]
        join_mask = None
        for jte in jt:
            join_mask_i = x_t != jte
            if join_mask is None:
                join_mask = join_mask_i
            else:
                join_mask = join_mask * join_mask_i

        corestates = self.core.get_states(batsize)

        self._save_states(corestates, branch_mask)
        restore_states = self._pop_states(corestates, join_mask)

        self.core.set_states(*restore_states)

        ret = super(HierarchicalAttentionDecoderCell, self).forward(x_t, ctx, ctxmask=ctxmask, t=t, outmask_t=outmask_t, **kw)
        return ret

    def reset_state(self):
        super(HierarchicalAttentionDecoderCell, self).reset_state()
        self._state_stack = None      # list per example in batch

    def _save_states(self, states, mask):
        """ saves states in state stack based on mask
            states are saved in lists per example, each example list contains lists of states"""
        if self._state_stack is None:
            self._state_stack = [[] for _ in range(mask.size(0))]
        for i in range(mask.size(0)):
            if mask[i].data[0] == 0:        # save
                statestosave = [state[i] for state in states]
                self._state_stack[i].append(statestosave)

    def _pop_states(self, states, mask):
        """ pops from saved states based on mask
            and merges with provided states based on mask """
        gatheredstates = []
        if torch.sum(mask).data[0] == mask.size(0):
            return states
        for i in range(mask.size(0)):
            if mask[i].data[0] == 0:        # pop saved
                poppedstates = self._state_stack[i].pop()
                gatheredstates.append(poppedstates)
            else:
                statestopreserve = [state[i] for state in states]
                gatheredstates.append(statestopreserve)
        # merge gathered states
        outstates = []
        for statepos in zip(*gatheredstates):
            toout = torch.stack(statepos, 0)
            outstates.append(toout)
        return outstates


class ModularDecoderCell(DecoderCell):
    """
    ModularDecoderCell contains core module and top module.
    Core module takes input as given to ModularDecoderCell and saved top2core_t.
    Top module takes the output from core module and returns output for a timestep and optional new values for top2core_t.
    """
    def __init__(self, core, top, **kw):
        """
        :param core: core recurrent module, gets a number of inputs at each timestep, updates its states, outputs one vector at each timestep
        :param top: top module implementing at least smo, optionally attention
        :param kw:
        """
        super(ModularDecoderCell, self).__init__(core, **kw)
        self.decoder_top = top
        self.top2core_t = None
        self.set_runner(TeacherForcer())        # teacher forcer by default

    def initialize(self, *x, **kw):
        """ called at every call to Decoder's forward() """
        # initialize decoder top
        if hasattr(self.decoder_top, "init_ctx_from_decoder_args"):
            self.decoder_top.init_ctx_from_decoder_args(*x, **kw)
        elif hasattr(self.decoder_top, "set_ctx"):
            topkw = {}
            toparg = []
            if "ctx" in kw:
                toparg += [kw["ctx"]]
                if "ctxmask" in kw:
                    topkw["ctxmask"] = kw["ctxmask"]
            self.decoder_top.set_ctx(*toparg, **topkw)

        # initialize stored top2core
        top2core_0 = None
        if hasattr(self.decoder_top, "get_top2core_0"):
            top2core_0 = self.decoder_top.get_top2core_0()
        self.top2core_t = top2core_0
        if "top2core_0" in kw:
            self.top2core_t = kw["top2core_0"]

        # initialize core states
        new_init_states = self._compute_init_states(*x, **kw)
        if new_init_states is not None:
            if not issequence(new_init_states):
                new_init_states = (new_init_states,)
            self.set_init_states(*new_init_states)

    def forward(self, *x, **kw):
        # get args and kwargs for core
        if self.top2core_t is not None:
            if isinstance(self.top2core_t, tuple) \
                    and len(self.top2core_t) == 2 \
                    and isinstance(self.top2core_t[1], dict):
                x = x + tuple(self.top2core_t[0])
                kwupd = self.top2core_t[1]
            elif isinstance(self.top2core_t, (tuple, list)):
                x = x + tuple(self.top2core_t)
            elif isinstance(self.top2core_t, dict):
                kwupd = self.top2core_t
            else:
                kwupd = {"ctx_t": self.top2core_t}
            _kw = {}
            _kw.update(kw)
            _kw.update(kwupd)
            kw = _kw
        # perform core forward
        coreout = self.core(*x, **kw)
        # routing core forward outputs
        if isinstance(coreout, tuple):
            assert(len(coreout) == 2)
            coreout_arg = coreout[0]
            coreout_kw = coreout[1]
        else:
            coreout_arg = coreout
            coreout_kw = {}
        assert(isinstance(coreout_kw, dict))
        if not issequence(coreout_arg):
            coreout_arg = [coreout_arg]
        # perform top forward
        topout = self.decoder_top(*coreout_arg, **coreout_kw)
        if isinstance(topout, tuple):
            assert(len(topout) == 2)
            y_t, top2core_tp1 = topout[0], topout[1]
        else:
            y_t, top2core_tp1 = topout, None
        # save top forward
        self.top2core_t = top2core_tp1
        # return top output
        return y_t


### RUNNERS #######################

class DecoderRunner(nn.Module):
    def __init__(self, **kw):
        super(DecoderRunner, self).__init__(**kw)

    def reset(self):
        pass

    # def __call__(self, t=None, x=None, xkw=None, y_t=None):
    #     self.forward(t=t, x=x, xkw=xkw, y_t=y_t)

    def forward(self, t=None, x=None, xkw=None, y_t=None):
        raise NotImplemented("use subclass")


class TeacherForcer(DecoderRunner):
    def forward(self, t=None, x=None, xkw=None, y_t=None):
        outargs = tuple([xi[:, t] for xi in x])
        outkwargs = {"t": t}
        if "outmask" in xkw:  # slice out the time from outmask
            outmask = xkw["outmask"]
            if outmask.dim() == 1:  # get mask from data stored on this object
                assert (self._sparse_outmask is not None)
                outmaskaddrs = list(outmask.cpu().data.numpy())
                sparse_outmask_t = [self._sparse_outmask[a][t] for a in outmaskaddrs]
                sparse_outmask_t = [torch.from_numpy(a.todense()).t() for a in sparse_outmask_t]
                sparse_outmask_t = torch.cat(sparse_outmask_t, 0)
                outmask_t = var(sparse_outmask_t).cuda(outmask).v
            elif outmask.data[0, 0, 1] > 1:  # batchable sparse
                vocsize = outmask.data[0, 0, 1]
                outmask_t = var(torch.ByteTensor(outmask.size(0), vocsize + 1)).cuda(outmask).v
                outmask_t.data.fill_(0)
                outmask_t.data.scatter_(1, outmask.data[:, t, 2:], 1)
                outmask_t.data = outmask_t.data[:, 1:]
                # outmask_t = var(outmask_t).cuda(outmask).v
            else:
                outmask_t = outmask[:, t]
            outkwargs["outmask_t"] = outmask_t
        return outargs, outkwargs


class FreeRunner(DecoderRunner):
    """ Basic free runner with argmax """
    def __init__(self, scores2probs=Softmax(), inparggetter=lambda x: (x, {}), **kw):
        super(FreeRunner, self).__init__(**kw)
        self.scores2probs = scores2probs
        self.outsym2insym = inparggetter

    def forward(self, t=None, x=None, xkw=None, y_t=None):
        outkwargs = {"t": t}
        if y_t is None:     # first time step
            assert(t == 0)
            if len(x) == 1:
                x = x[0]
                if x.dim() == 2:
                    x = x[:, 0]
            x_t = x
        else:
            if "outmask" in xkw:
                raise NotImplemented("outmask not supported in free running mode yet")
            if issequence(y_t):
                assert(len(y_t) == 1)
                y_t = y_t[0]
            _y_t = self.scores2probs(y_t)
            _, x_t = torch.max(_y_t, 1)

        # return
        r = self.outsym2insym(x_t)
        if isinstance(r, tuple) and len(r) == 2 and isinstance(r[1], dict):
            inpargs, kwupd = r
            outkwargs.update(kwupd)
        else:
            inpargs = r
        # outargs = (x_t,)
        return inpargs, outkwargs


# DECODER MODULES #################

class DecoderCore(RecStateful):
    def __init__(self, emb, *layers, **kw):
        super(DecoderCore, self).__init__(**kw)
        self.block = RecStack(*layers)
        self.emb = emb

    def forward(self, x_t, ctx_t=None, t=None, outmask_t=None, **kw):
        if self.emb is not None:
            emb, _ = self.emb(x_t)
        else:
            emb = x_t
        i_t = emb if ctx_t is None else torch.cat([emb, ctx_t], 1)
        o_t = self.block(i_t)
        return o_t, {"mask": outmask_t, "t":t, "x_t_emb": emb, "ctx_t": ctx_t}

    def reset_state(self):
        self.block.reset_state()


class DecoderTop(nn.Module):
    """ Decoder Top - default - applies given layers as a stack """
    def __init__(self, *layers, **kw):
        super(DecoderTop, self).__init__(**kw)
        self.layers = Stack(*layers)

    def forward(self, x, **kw):   # x = vector from decoder core
        out = self.layers(x, **kw)
        return out


class ContextDecoderTop(DecoderTop):
    """ Decoder Top with context """
    def __init__(self, *layers, **kw):
        super(ContextDecoderTop, self).__init__(**kw)
        self.layers = Stack(*layers)
        self.stored_ctx = None

    def set_ctx(self, *ctx_arg, **ctx_kw):
        ctx = (ctx_arg, ctx_kw)
        self.stored_ctx = ctx

    def reset(self):
        self.stored_ctx = None

    def forward(self, x, **kw):
        out = self.layers(x)
        return out, self.stored_ctx

    def get_top2core_0(self):
        raise NotImplemented()


class StaticContextDecoderTop(ContextDecoderTop):
    """ Decoder Top with static context """
    def __init__(self, *layers, **kw):
        self.ctx2inp = getkw(kw, "ctx2inp", False)
        self.ctx2out = getkw(kw, "ctx2out", True)
        super(StaticContextDecoderTop, self).__init__(*layers, **kw)

    def set_ctx(self, ctx):     # ctx must be concatenatable to x in forward (along axis 1)
        self.stored_ctx = ctx

    def forward(self, x, mask=None, **kw):
        inp = x
        if self.ctx2inp is True:
            inp = torch.cat([inp, self.stored_ctx], 1)
        out = self.layers(inp, mask=mask)
        if self.ctx2out:
            return out, self.stored_ctx
        else:
            return out

    def get_top2core_0(self):
        if self.ctx2out:
            return self.stored_ctx
        else:
            return None


class AttentionContextDecoderTop(ContextDecoderTop):
    """ Decoder Top with dynamic attention-based context """
    def __init__(self, attention, *layers, **kw):
        self.ctx2inp = getkw(kw, "ctx2inp", True)       # use ctx as input to smo
        self.ctx2out = getkw(kw, "ctx2out", False)      # use ctx as input to core
        self.inp2inp = getkw(kw, "inp2inp", True)       # use input as input to smo
        self.att_after_update = getkw(kw, "att_after_update", True)     # false not supported
        self.split = getkw(kw, "split", False)
        self.return_out = getkw(kw, "return_out", True)
        self.return_att = getkw(kw, "return_att", False)
        self.inpemb2att = getkw(kw, "inpemb2att", False)
        self.inpemb2inp = getkw(kw, "inpemb2inp", False)
        self.attention_transform = getkw(kw, "attention_transform", None)

        if self.att_after_update is False:
            raise NotImplemented("attention before update is not supported yet")

        super(AttentionContextDecoderTop, self).__init__(*layers, **kw)

        self.attention = attention

        if self.split:      # !!! if split, use q.intercat in ctx encoder and core output if they are from separate chunks of networks
            self.attention.split_data()

    def set_ctx(self, ctx, ctx_0, ctxmask=None):
        self.stored_ctx = ([ctx, ctx_0], {"ctxmask": ctxmask})

    def init_ctx_from_decoder_args(self, *x, **kw):
        ctx = kw["ctx"]
        ctx_0 = kw["ctx_0"]
        ctxmask = kw["ctxmask"] if "ctxmask" in kw else None
        self.set_ctx(ctx, ctx_0, ctxmask)

    def get_top2core_0(self):
        ctx_0 = self.stored_ctx[0][1]
        if self.split:
            ctx_0 = ctx_0[:, ctx_0.size(1) // 2:]
        return ctx_0    # ctx_0 provided in set_ctx()

    def forward(self, core_out, x_t_emb=None, ctx_t=None, mask=None, t=None, **kw):
        ctx, ctxmask = self.stored_ctx[0][0], self.stored_ctx[1]["ctxmask"]

        # compute new attention
        gen_vec = core_out
        if self.split:
            gen_vec = core_out[:, :core_out.size(1) // 2]
        if self.inpemb2att:
            assert(x_t_emb is not None)
            gen_vec = torch.cat([gen_vec, x_t_emb], 1)
        if self.attention_transform is not None:
            gen_vec = self.attention_transform(gen_vec)
        att_weights_t = self.attention.attgen(ctx, gen_vec, mask=ctxmask)
        ctx_t = self.attention.attcon(ctx, att_weights_t)

        # execute block
        cat_to_smo = []
        o_to_smo = core_out[:, core_out.size(1)//2:] if self.split else core_out
        if self.inp2inp:    cat_to_smo.append(o_to_smo)
        if self.ctx2inp:    cat_to_smo.append(ctx_t)
        if self.inpemb2inp: cat_to_smo.append(x_t_emb)
        inp_t = torch.cat(cat_to_smo, 1) if len(cat_to_smo) > 1 else cat_to_smo[0]
        inp_t_kw = {}
        inp_t_kw.update(kw)
        if mask is not None:
            inp_t_kw["mask"] = mask

        y_t = self.layers(inp_t, **inp_t_kw)

        # returns
        ret = tuple()
        if self.return_out:
            ret += (y_t,)
        if self.return_att:
            ret += (att_weights_t,)

        return ret, ctx_t




