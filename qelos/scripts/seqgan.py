import qelos as q
import torch
from torch import nn
import numpy as np
import h5py
from IPython import embed
import sparkline


class Logger(object):
    def __init__(self, name="", logiter=50):
        self.tt = q.ticktock(name)
        self.logiter = logiter

    def log(self, _iter=None, niter=None, errD=None, errG=None, scoreD_real=None, scoreD_fake=None, lip_loss=None, **kw):
        if (_iter+1) % self.logiter == 0:
            self.tt.live("[{}/{}] Loss_D: {:.4f} Loss_G: {:.4f} Score Real: {:.4f} Score Fake: {:.4f} Loss Lip: {:.4f}"
                         .format(_iter, niter, errD, errG, scoreD_real, scoreD_fake, lip_loss))



def get_data_gen(vocsize, batsize, seqlen):
    def data_gen():
        while True:
            yield np.random.randint(0, vocsize, (batsize, seqlen)).astype("int64")
    return data_gen


def test_acha_net_g():
    batsize, seqlen, vocsize = 5, 4, 7
    embdim, encdim, outdim, ctxdim = 10, 16, 10, 8
    noisedim = ctxdim
    gendim, discdim = encdim, encdim

    (netD, _), (netG, _), _, _, amortizer = make_nets_normal(vocsize, embdim, gendim, discdim, None, seqlen, noisedim)


    # end model def
    data = np.random.randint(0, vocsize, (batsize, seqlen))
    data = q.var(torch.LongTensor(data)).v
    ctx = q.var(torch.FloatTensor(np.random.random((batsize, ctxdim)))).v

    for iter in (499, 500, 600, 800, 1000):
        amortizer.update(iter=iter)
        decoded = netG(ctx, data)
        decoded = decoded.data.numpy()
        assert(decoded.shape == (batsize, seqlen-1, vocsize))  # shape check
        assert(np.allclose(np.sum(decoded, axis=-1), np.ones_like(np.sum(decoded, axis=-1))))  # prob check


class ContextDecoderACHAWrappper(nn.Module):
    def __init__(self, decoder, amortizer=None, clcutter=None, soft_next=False, vocsize=None, **kw):
        super(ContextDecoderACHAWrappper, self).__init__(**kw)
        self.decoder = decoder
        self.amortizer = amortizer
        self.soft_next = soft_next
        self.vocsize = vocsize
        self.idx2onehot = q.IdxToOnehot(vocsize)
        self.clcutter = clcutter

    def forward(self, noise, sequences, maxtime=None):
        if self.clcutter is not None:
            sequences = self.clcutter(sequences)
        overrideseq = sequences[:, 1:].contiguous()
        sequences = sequences[:, :-1]
        if self.soft_next:
            sequences = self.idx2onehot(sequences)
        kw = {}
        if maxtime is not None:
            kw["maxtime"] = maxtime
        dec = self.decoder(sequences, noise, **kw)        # (batsize, seqlen, vocsize)
        seqlen, vocsize = dec.size(1), dec.size(2)
        override = self.idx2onehot(overrideseq)
        amortizer = min(self.amortizer.v+1, seqlen)
        if amortizer == seqlen:
            out = dec
        else:
            overridelen = override.size(1) - amortizer
            copylen = seqlen - overridelen
            out = torch.cat([override[:, :overridelen], dec[:, -copylen:]], 1)
        return out


def make_nets_ganesh(vocsize, embdim, gendim, discdim, startsym,
                     seqlen, noisedim, soft_next=False, temperature=1.,
                     clrate=200):
    # TODO: variable-length sequences (masks!!!)
    def get_amortizer_update_rule(offset=0, headstart=clrate, interval=clrate):
        def amortizer_update_rule(current_value=0, iter=0, state=None):
            if iter < (offset + headstart):
                return 0
            else:
                ret = 1 + (iter - (offset + headstart)) // interval
                if current_value != ret:
                    print("AMORTIZED FROM {} TO {}".format(current_value, ret))
                return ret
        return amortizer_update_rule
    amortizer_curriculum = q.DynamicHyperparam(update_rule=get_amortizer_update_rule())
    amortizers = [amortizer_curriculum]

    def get_cl_cutter(clamort):
        def cl_cutter(seq):
            if clamort is not None:
                upto = min(clamort.v + 2, seq.size(1))
                out = seq[:, :upto]
                return out
            else:
                return seq

        return cl_cutter

    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            decodercell = q.ContextDecoderCell(
                nn.Linear(vocsize, embdim, bias=False),
                q.GRUCell(embdim+noisedim, gendim),
                nn.Linear(gendim, vocsize),
                q.Softmax(temperature=temperature)
            )

            startsymbols = np.zeros((1, vocsize), dtype="float32")
            startsymbols[0, startsym] = 1
            startsymbols = q.val(startsymbols).v
            self.startsymbols = startsymbols
            decodercell.teacher_unforce(seqlen-1, q.Lambda(lambda x: x[0]), startsymbols=startsymbols)

            self.decoder = decodercell.to_decoder()

        def forward(self, noise):
            decret = self.decoder(noise, maxtime=min(amortizer_curriculum.v+1, seqlen-1))    # (batsize, seqlen, vocsize)
            batsize = noise.size(0)
            ret = torch.cat([self.startsymbols.repeat(batsize, 1).unsqueeze(1),
                             decret], 1)
            return ret

    class Critic(nn.Module):
        def __init__(self):
            super(Critic, self).__init__()
            self.net = q.RecurrentStack(
                nn.Linear(vocsize, discdim, bias=False),
                q.GRUCell(discdim, discdim, use_cudnn_cell=False).to_layer(),
                nn.Linear(discdim, vocsize, bias=False)
            )

        def forward(self, x):   # (batsize, seqlen, vocsize)
            inpx = x[:, :-1, :]
            netouts = self.net(inpx)
            outx = x[:, 1:, :]
            scores = outx * netouts
            vocagg_scores = scores.sum(2)
            seqagg_scores = vocagg_scores.sum(1)
            return seqagg_scores

    netRclcutter = get_cl_cutter(amortizer_curriculum)
    netR = q.Stack(q.Lambda(lambda x: netRclcutter(x)),
                   q.Lambda(lambda x: x.contiguous()),
                   q.IdxToOnehot(vocsize))
    # netR = q.IdxToOnehot(vocsize)
    netD = Critic()
    netG = Generator()

    # amortizers = []

    def sample(noise=None, cuda=False, gen=None, rawlen=None):
        if noise is None:
            noise = q.var(torch.randn(1, noisedim)).cuda(cuda).v
            noise.data.normal_(0, 1)
        if rawlen is not None:
            o = netG.decoder(noise, maxtime=rawlen)
        else:
            o = netG(noise)

        _, y = torch.max(o, 2)
        do = o.clone()

        score = netD(do)
        return y, o[0], score, None, None

    return (netD, netD), (netG, netG), netR, sample, amortizers


def make_nets_wrongfool(vocsize, embdim, gendim, discdim, startsym,
                        seqlen, noisedim, soft_next=False, temperature=1.,
                        clrate=0, mode="single"):     # "dual" or "single"

    print("wrongfool mode: {}".format(mode))

    def get_amortizer_update_rule(offset=0, headstart=clrate, interval=clrate):
        def amortizer_update_rule(current_value=0, iter=0, state=None):
            if iter < (offset + headstart):
                return 0
            else:
                ret = 1 + (iter - (offset + headstart)) // max(interval, 1)
                if current_value != ret:
                    print("AMORTIZED FROM {} TO {}".format(current_value, ret))
                return ret

        return amortizer_update_rule

    amortizer_curriculum = q.DynamicHyperparam(update_rule=get_amortizer_update_rule())
    amortizers = [amortizer_curriculum]

    def get_cl_cutter(clamort):
        def cl_cutter(seq):
            if clamort is not None:
                upto = min(clamort.v + 2, seq.size(1))
                out = seq[:, :upto]
                return out
            else:
                return seq

        return cl_cutter

    decodercell = q.ContextDecoderCell(
        nn.Linear(vocsize, embdim, bias=False),
        q.GRUCell(embdim + noisedim, gendim),
        nn.Linear(gendim, vocsize),
        q.Softmax(temperature=temperature)
    )
    startsymbols = np.zeros((1, vocsize), dtype="float32")
    startsymbols[0, startsym] = 1
    startsymbols = q.val(startsymbols).v
    decodercell.teacher_unforce(seqlen - 1, q.Lambda(lambda x: x[0]), startsymbols=startsymbols)

    decoder = decodercell.to_decoder()
    idxtoonehot = q.IdxToOnehot(vocsize)

    argmax_in_d_mode = True

    class Generator(nn.Module):
        def __init__(self, gmode=False):
            super(Generator, self).__init__()
            self.gmode = gmode
            self.idxtoonehot = idxtoonehot
            self.decoder = decoder
            self.argmax_in_d_mode = argmax_in_d_mode

        def forward(self, noise):
            if clrate is not None and clrate > 0:
                decret = self.decoder(noise, maxtime=min(amortizer_curriculum.v+1, seqlen-1))  # (batsize, seqlen, vocsize)
            else:
                decret = self.decoder(noise)
            batsize = noise.size(0)
            ret = torch.cat([startsymbols.repeat(batsize, 1).unsqueeze(1),
                             decret], 1)
            if not self.gmode and self.argmax_in_d_mode:
                _, ret = torch.max(ret, 2)
                ret = self.idxtoonehot(ret)
            return ret


    specialcriticemb = nn.Linear(vocsize, discdim, bias=False)


    class SpecialCriticCell(q.rnn.GRUCell):
        """ embeds, does one-hot for states, no one-hot for outputs"""
        def __init__(self, *args, **kwargs):
            super(SpecialCriticCell, self).__init__(*args, **kwargs)
            self.idxtoonehot = idxtoonehot
            self.specialcriticemb = specialcriticemb
            self.gmode = False

        def _forward(self, x_t, h_tm1, t=None): #x_t: dist
            if self.gmode:  #
                _, x_t_onehot = torch.max(x_t, 1)
                x_t_onehot = self.idxtoonehot(x_t_onehot)
                x_t_onehot_emb = self.specialcriticemb(x_t_onehot)
                _, h_t = super(SpecialCriticCell, self)._forward(x_t_onehot_emb, h_tm1, t=t)

                x_t_emb = self.specialcriticemb(x_t)
                y_t, _ = super(SpecialCriticCell, self)._forward(x_t_emb, h_tm1, t=t)
            else:   # not in g-mode --> generator outputting sharp if argmax_in_g
                x_t_emb = self.specialcriticemb(x_t)
                y_t, h_t = super(SpecialCriticCell, self)._forward(x_t_emb, h_tm1, t=t)
            return y_t, h_t


    class DualCritic(nn.Module):
        def __init__(self, netpast, netpresent, cellpresent, gmode=False):
            super(DualCritic, self).__init__()
            self.net_past = netpast
            self.net_present = netpresent
            self.cell_present = cellpresent
            self.distance = q.DotDistance()
            # self.distact = q.Lambda(lambda x: x)
            self.distact = nn.Tanh()
            self.gmode = gmode
            self.aggparam = None
            self.idxtoonehot = q.IdxToOnehot(vocsize)

        def forward(self, x):   # (batsize, seqlen, ...), seqlen >= 3
            x = torch.cat([x[:, :1, :], x], 1)  # HACK: to have something for first real output from present
            # if not self.gmode:
            #     _, x = torch.max(x.detach(), 2)
            #     x = self.idxtoonehot(x)
            self.cell_present._detach_states = self.gmode
            inppast = x[:, :-1, :]
            if self.gmode and argmax_in_d_mode:
                _, inppast = torch.max(inppast, 2)
                inppast = self.idxtoonehot(inppast)
            pastouts = self.net_past(inppast)
            inppresent = x[:, :, :]
            if isinstance(self.cell_present, SpecialCriticCell):
                self.cell_present.gmode = self.gmode
            presentouts = self.net_present(inppresent)
            presentouts = presentouts[:, 1:, :]     # what about first present output?

            pastouts = pastouts.detach() if self.gmode else pastouts

            batsize, seqlen, _ = pastouts.size()
            pastouts = pastouts.contiguous().view(batsize * seqlen, -1)
            presentouts = presentouts.contiguous().view(batsize * seqlen, -1)
            distances = self.distance(pastouts, presentouts)
            distances = distances.view(batsize, seqlen)
            distances = self.distact(distances)
            scores = distances.sum(1)
            return scores

    singlecriticdist = q.BilinearDistance(discdim, discdim)

    class SingleCritic(nn.Module):
        def __init__(self, net, cell, gmode=False):
            super(SingleCritic, self).__init__()
            self.net = net
            self.cell = cell
            self.distance = singlecriticdist
            self.distact = nn.Tanh()
            # self.distance = q.DotDistance()
            self.gmode = gmode
            self.aggparam = None
            self.idxtoonehot = q.IdxToOnehot(vocsize)

        def forward(self, x):  # (batsize, seqlen, ...), seqlen >= 3
            x = torch.cat([x[:, :1, :], x], 1)  # HACK: to have something for first real output from present
            # if not self.gmode:
            #     _, x = torch.max(x, 2)
            #     x = self.idxtoonehot(x)
            self.cell._detach_states = self.gmode
            y = self.net(x)
            pastouts = y[:, :-1, :]
            presentouts = y[:, 1:, :]  # what about first present output?

            pastouts = pastouts.detach() if self.gmode else pastouts

            batsize, seqlen, _ = pastouts.size()
            pastouts = pastouts.contiguous().view(batsize * seqlen, -1)
            presentouts = presentouts.contiguous().view(batsize * seqlen, -1)
            distances = self.distance(pastouts, presentouts)
            distances = distances.view(batsize, seqlen)
            distances = self.distact(distances)
            scores = distances.sum(1)
            return scores

    def make_critics():
        net_past = q.RecurrentStack(
                nn.Linear(vocsize, discdim, bias=False),
                q.GRUCell(discdim, discdim, use_cudnn_cell=False).to_layer(),
                # nn.Linear(discdim, discdim, bias=False)
            )

        cell_present = q.GRUCell(discdim, discdim, use_cudnn_cell=False)
        if argmax_in_d_mode:
            cell_present = SpecialCriticCell(discdim, discdim, use_cudnn_cell=False)
            net_present = cell_present.to_layer()
        else:
            net_present = q.RecurrentStack(
                nn.Linear(vocsize, discdim, bias=False),
                cell_present.to_layer(),
                # nn.Linear(discdim, discdim, bias=False)
            )
        if mode == "dual":
            criticd = DualCritic(net_past, net_present, cell_present, gmode=False)
            criticg = DualCritic(net_past, net_present, cell_present, gmode=True)
        elif mode == "single":
            criticd = SingleCritic(net_present, cell_present, gmode=False)
            criticg = SingleCritic(net_present, cell_present, gmode=True)

        return criticd, criticg

    netG4D = Generator(gmode=False)
    netG4G = Generator(gmode=True)
    netD4D, netD4G = make_critics()

    if clrate is not None and clrate > 0:
        netRclcutter = get_cl_cutter(amortizer_curriculum)
        netR = q.Stack(q.Lambda(lambda x: netRclcutter(x)),
                       q.Lambda(lambda x: x.contiguous()),
                       q.IdxToOnehot(vocsize))
    else:
        netR = q.IdxToOnehot(vocsize)

    def sample(noise=None, cuda=False, gen=None, rawlen=None):
        if noise is None:
            noise = q.var(torch.randn(1, noisedim)).cuda(cuda).v
            noise.data.normal_(0, 1)
        if rawlen is not None:
            o = netG4G.decoder(noise, maxtime=rawlen)
        else:
            o = netG4G(noise)

        _, y = torch.max(o, 2)
        do = o.clone()

        score = netD4D(do)
        return y, o[0], score, None, None

    return (netD4D, netD4G), (netG4D, netG4G), netR, sample, [amortizer_curriculum]



def make_nets_wrongfool_acha(vocsize, embdim, gendim, discdim, startsym,
                        seqlen, noisedim, soft_next=False, temperature=1.,
                        clrate=0, mode="acha-sync", g_sharp=True):     # "dual" or "single"

    def get_amortizer_update_rule(offset=0, headstart=clrate, interval=clrate):
        def amortizer_update_rule(current_value=0, iter=0, state=None):
            if iter < (offset + headstart):
                return 0
            else:
                ret = 1 + (iter - (offset + headstart)) // max(1, interval)
                if current_value != ret:
                    print("AMORTIZED FROM {} TO {}".format(current_value, ret))
                return ret

        return amortizer_update_rule

    if mode == "acha-sync":
        amortizer_override = q.DynamicHyperparam(update_rule=get_amortizer_update_rule())
        amortizer_teacherforce = amortizer_override
        amortizer_curriculum = None
        amortizers = [amortizer_override]

    def get_cl_cutter(clamort):
        def cl_cutter(seq):
            if clamort is not None:
                upto = min(clamort.v + 2, seq.size(1))
                out = seq[:, :upto]
                return out
            else:
                return seq

        return cl_cutter

    decodercell = q.ContextDecoderCell(
        nn.Linear(vocsize, embdim, bias=False),
        q.GRUCell(embdim + noisedim, gendim),
        nn.Linear(gendim, vocsize),
        q.Softmax(temperature=temperature)
    )
    # startsymbols = np.zeros((1, vocsize), dtype="float32")
    # startsymbols[0, startsym] = 1
    # startsymbols = q.val(startsymbols).v
    # decodercell.teacher_unforce(seqlen - 1, q.Lambda(lambda x: x[0]),
    #                             startsymbols=startsymbols)

    # decoder = decodercell.to_decoder()
    idxtoonehot = q.IdxToOnehot(vocsize)

    # g_sharp = True
    netR_sharp = g_sharp

    if netR_sharp:
        netR = q.Stack(q.Lambda(lambda x: clcutter(x)),
                       q.Lambda(lambda x: x[:, 1:].contiguous()),
                       q.IdxToOnehot(vocsize)
                       )
    else:
        netR = q.Stack(q.Lambda(lambda x: clcutter(x)),
                       q.Lambda(lambda x: x[:, 1:].contiguous()),
                       q.IdxToOnehot(vocsize),
                       q.Softmax(temperature=0.3),
                       )

    clcutter = get_cl_cutter(amortizer_curriculum)

    class Generator(nn.Module):
        def __init__(self, sharp=False, sharp_free=False, **kw):
            super(Generator, self).__init__(**kw)
            self.decodercell = decodercell
            self.idx2onehot = idxtoonehot
            if sharp_free:
                freeblock = q.Stack(q.Lambda(lambda x: torch.max(x[0], 1)[1]),
                                    self.idx2onehot)
            else:
                freeblock = q.Lambda(lambda x: x[0])
            self.decodercell.teacher_unforce_after(seqlen-1,
                                                   amortizer_teacherforce,
                                                   freeblock)
            self.decoder = self.decodercell.to_decoder()
            self.sharp = sharp
            self.override_softmax = q.Softmax(temperature=0.3)

        def forward(self, noise, sequences, maxtime=None):
            if clcutter is not None:
                sequences = clcutter(sequences)
            sequences = self.idx2onehot(sequences)
            overrideseq = sequences[:, 1:].contiguous()
            sequences = sequences[:, :-1].contiguous()
            if not netR_sharp:
                overrideseq = self.override_softmax(overrideseq)
            kw = {}
            if maxtime is not None:
                kw["maxtime"] = maxtime
            dec = self.decoder(sequences, noise, **kw)  # (batsize, seqlen, vocsize)
            if self.sharp:
                _, dec = torch.max(dec, 2)
                dec = self.idx2onehot(dec)
            decseqlen, decvocsize = dec.size(1), dec.size(2)
            amortizer = min(amortizer_override.v + 1, decseqlen)
            if amortizer == decseqlen:
                out = dec
            else:
                overridelen = overrideseq.size(1) - amortizer
                copylen = decseqlen - overridelen
                out = torch.cat([overrideseq[:, :overridelen], dec[:, -copylen:]], 1)
            return out

    disccell = q.RecStack(
        nn.Linear(vocsize, embdim, bias=False),
        q.GRUCell(embdim, discdim, use_cudnn_cell=False),
    )

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.cell = disccell
            self.recnet = self.cell.to_layer()
            self.summ = q.Stack(
                nn.Linear(discdim, 1),
            )

        def forward(self, x):   # sequence of onehot
            recout = self.recnet(x)
            sum = self.summ(recout[:, -1, :])
            return sum.unsqueeze(1)

    netG4D = Generator(sharp=g_sharp, sharp_free=True)
    netG4G = Generator(sharp=False, sharp_free=True)
    netD4D = Discriminator()
    netD4G = Discriminator()

    def sample(noise=None, cuda=False, gen=None, rawlen=None):
        gold = None
        if noise is None:
            noise = q.var(torch.randn(1, noisedim)).cuda(cuda).v
            noise.data.normal_(0, 1)
        if rawlen is not None and gen is None:  # start form startsyms
            data = q.var(np.ones((1, seqlen), dtype="int64") * startsym).cuda(cuda).v
        else:  # start from data
            data = next(gen)[0:1]
            gold = data
            data = q.var(data).cuda(cuda).v
        if rawlen is not None:
            if gen is None:  # completely unforce, generate rawlen
                netG4G.decoder.block.teacher_unforce(rawlen, q.Lambda(lambda x: torch.max(x[0], 1)[1]), data.data[0][0])
                o = netG4G.decoder(noise, maxtime=rawlen)
            else:  # generate from data up to rawlen
                o = netG4G(noise, data, maxtime=rawlen)
        else:
            o = netG4G(noise, data)

        _, y = torch.max(o, 2)

        score = netD4D(o)
        return y, o[0], score, None, None

    return (netD4D, netD4G), (netG4D, netG4G), netR, sample, amortizers


def make_nets_normal(vocsize, embdim, gendim, discdim, startsym,
                     seqlen, noisedim, soft_next=False, temperature=1.,
                     mode="acha-sync"):

    def get_amortizer_update_rule(offset=0, headstart=500, interval=200):
        def amortizer_update_rule(current_value=0, iter=0, state=None):
            if iter < (offset + headstart):
                return 0
            else:
                ret = 1 + (iter - (offset + headstart)) // interval
                if current_value != ret:
                    print("AMORTIZED FROM {} TO {}".format(current_value, ret))
                return ret
        return amortizer_update_rule
    if mode == "acha-async":
        amortizer_override = q.DynamicHyperparam(update_rule=get_amortizer_update_rule())
        tf_amor_offset = seqlen * 200 + 500
        amortizer_teacherforce = q.DynamicHyperparam(update_rule=get_amortizer_update_rule(offset=tf_amor_offset))
        amortizer_curriculum = None
        amortizers = [amortizer_override, amortizer_teacherforce]
    elif mode == "acha-sync":
        amortizer_override = q.DynamicHyperparam(update_rule=get_amortizer_update_rule())
        amortizer_teacherforce = amortizer_override
        amortizer_curriculum = None
        amortizers = [amortizer_override]
    elif mode == "clth":
        amortizer_override = q.Hyperparam(0)
        amortizer_teacherforce = q.Hyperparam(0)    # no unteacherforce
        amortizer_curriculum = q.DynamicHyperparam(update_rule=get_amortizer_update_rule(0))
        amortizers = [amortizer_curriculum]

    def get_cl_cutter(clamort):
        def cl_cutter(seq):
            if clamort is not None:
                upto = min(clamort.v + 2, seq.size(1))
                out = seq[:, :upto]
                return out
            else:
                return seq

        return cl_cutter

    netG = q.ContextDecoderCell(
        nn.Embedding(vocsize, embdim) if not soft_next else nn.Linear(vocsize, embdim),
        q.GRUCell(embdim+noisedim, gendim),
        nn.Linear(gendim, vocsize),
        q.Softmax(temperature=temperature),
    )
    if amortizer_curriculum is None:
        netG.teacher_unforce_after(seqlen-1, amortizer_teacherforce,
                             q.Lambda(lambda x: torch.max(x[0], 1)[1])
                             if not soft_next else q.Lambda(lambda x: x[0])
        )
    netG = netG.to_decoder()
    netG = ContextDecoderACHAWrappper(netG, amortizer=amortizer_override,
            clcutter=get_cl_cutter(amortizer_curriculum),
            soft_next=soft_next, vocsize=vocsize)

    # DISCRIMINATOR
    netD = q.RecurrentStack(
        nn.Linear(vocsize, discdim, bias=False),
        q.GRUCell(discdim, discdim, use_cudnn_cell=False).to_layer(),
    ).return_final()

    netD = q.Stack(
        netD,
        # q.Forward(discdim, discdim, activation="relu"),
        nn.Linear(discdim, 1),
        q.Lambda(lambda x: x.squeeze(1))
    )

    netRclcutter = get_cl_cutter(amortizer_curriculum)
    netR = q.Stack(q.Lambda(lambda x: netRclcutter(x)),
                   q.Lambda(lambda x: x[:, 1:].contiguous()),
                   q.IdxToOnehot(vocsize))

    def sample(noise=None, cuda=False, gen=None, rawlen=None):
        gold = None
        if noise is None:
            noise = q.var(torch.randn(1, noisedim)).cuda(cuda).v
            noise.data.normal_(0, 1)
        if rawlen is not None and gen is None:      # start form startsyms
            data = q.var(np.ones((1, seqlen), dtype="int64") * startsym).cuda(cuda).v
        else:   # start from data
            data = next(gen)[0:1]
            gold = data
            data = q.var(data).cuda(cuda).v
        if rawlen is not None:
            if gen is None:     # completely unforce, generate rawlen
                netG.decoder.block.teacher_unforce(rawlen, q.Lambda(lambda x: torch.max(x[0], 1)[1]), data.data[0][0])
                o = netG.decoder(noise, maxtime=rawlen)
            else:   # generate from data up to rawlen
                o = netG(noise, data, maxtime=rawlen)
        else:
            o = netG(noise, data)

        do = q.var(o.data.new(o.size())).cuda(o).v
        do.data = o.data + 0
        do.requires_grad = True

        netDparamswithgrad = []
        for param in netD.parameters():
            if param.requires_grad:
                netDparamswithgrad.append(param)
                param.requires_grad = False

        _, y = torch.max(o, 2)

        score = netD(do)

        if rawlen is None:
            loss = -score.sum()
            loss.backward()
            dograds = do.grad

            for param in netDparamswithgrad:
                param.requires_grad = True
            return y, o[0], score, dograds[0], gold[0]
        else:
            return y, o[0], score, None, None

    return (netD, netD), (netG, netG), netR, sample, amortizers


# region data loading
def makemat(data, window, subsample, startid=None):
    startpositions = np.arange(0, data.shape[0] - window)
    np.random.shuffle(startpositions)
    numex = startpositions.shape[0] // subsample
    startpositions = startpositions[:numex]
    mat = np.zeros((startpositions.shape[0], window), dtype="int64")
    for i in range(startpositions.shape[0]):
        startpos = startpositions[i]
        mat[i, :] = data[startpos:startpos + window]
    if startid is not None:
        mat[:, 0] = startid
    return mat


def loaddata_hutter(p="../../datasets/hutter/enwik8.h5", window=200, subsample=1000):
    return loaddata(p=p, window=window, subsample=subsample, utfdic=True)


def loaddata_text8(p="../../datasets/text8/text8.h5", window=200, subsample=1000):
    return loaddata(p=p, window=window, subsample=subsample, utfdic=False)


def loaddata(p="../../datasets/hutter/enwik8.h5", window=200, subsample=1000, utfdic=True):
    tt = q.ticktock("dataloader")
    tt.tick("loading data")
    with h5py.File(p, "r") as f:
        if utfdic:
            charlist = [x.decode("unicode-escape") for x in list(f["dict"][:, 0])]
        else:
            charlist = list(f["dict"][:, 0])
        chardic = dict(zip(range(len(charlist)), charlist))
        train, valid, test = f["train"][:], f["valid"][:], f["test"][:]
    tt.tock("data loaded")
    tt.tick("making mats")

    startid = max(chardic.keys()) + 1
    chardic[startid] = "<START>"

    def pp(charseq):
        if charseq.ndim == 1:
            return "".join([chardic[x] for x in charseq])
        elif charseq.ndim == 2:
            ret = []
            for i in range(len(charseq)):
                ret.append(pp(charseq[i]))
            return ret

    ret = (makemat(train, window, subsample, startid=startid),
           makemat(valid, window, subsample, startid=startid),
           makemat(test, window, subsample, startid=startid),
           chardic, pp)
    tt.tock("made mats")
    return ret
# endregion


def makeiter(dl):
    def inner():
        for i in dl:
            yield i
    dli = inner()
    while True:
        try:
            ret = next(dli)
            yield ret[0]
        except StopIteration as e:
            dli = inner()


def run(lr=0.0003,
        wreg=0.000001,
        embdim=64,
        noisedim=64,
        gendim=256,
        discdim=256,
        batsize=256,
        niter=-1,
        niterD=10,
        niterG=1,
        seqlen=32,
        cuda=False,
        gpu=1,
        mode="acha",       # "normal", "ganesh", "wrongfool"
        wrongfoolmode="dual",
        subsample=1000,
        inspectdata=False,
        pw=10,
        temperature=1.,
        clrate=0,
        logiter=50,
        gsharp=True,
        ):
    if cuda:
        torch.cuda.set_device(gpu)

    if niter == -1:
        # niter = (seqlen * 200 + 500) + 500        # seqlen * amortize_step + amortize_headstart + afterburn
        niter = clrate * (seqlen + 2)
        print("niter: {}".format(niter))
    # get data and dict
    # datagen = get_data_gen(vocsize, batsize, seqlen)()
    trainmat, validmat, testmat, rcd, pp = loaddata_text8(window=seqlen, subsample=subsample)
    cd = {v: k for k, v in rcd.items()}
    traingen = makeiter(q.dataload(trainmat, batch_size=batsize, shuffle=True))
    validgen = makeiter(q.dataload(validmat, batch_size=batsize, shuffle=False))
    testgen = makeiter(q.dataload(testmat, batch_size=batsize, shuffle=False))

    vocsize = np.max(trainmat) + 1

    print("Number of training sequences: {}, vocabulary of {}"
          .format(trainmat.shape[0], vocsize))

    if inspectdata:
        embed()

    # build networks
    if mode == "acha":
        (netD4D, netD4G), (netG4D, netG4G), netR, sampler, amortizers \
            = make_nets_wrongfool_acha(vocsize, embdim, gendim, discdim,
                               cd["<START>"], seqlen, noisedim,
                               temperature=temperature, clrate=clrate,
                               g_sharp=gsharp)
    elif mode == "normal":
        (netD4D, netD4G), (netG4D, netG4G), netR, sampler, amortizers \
            = make_nets_normal(vocsize, embdim, gendim, discdim,
                               cd["<START>"], seqlen, noisedim,
                               temperature=temperature,
                               mode="acha-sync")
    elif mode == "ganesh":
        (netD4D, netD4G), (netG4D, netG4G), netR, sampler, amortizers \
            = make_nets_ganesh(vocsize, embdim, gendim, discdim,
                               cd["<START>"], seqlen, noisedim, temperature=temperature,
                               clrate=clrate)
    elif mode == "wrongfool":
        (netD4D, netD4G), (netG4D, netG4G), netR, sampler, amortizers \
            = make_nets_wrongfool(vocsize, embdim, gendim, discdim,
                               cd["<START>"], seqlen, noisedim, temperature=temperature,
                               clrate=clrate, mode=wrongfoolmode)
    else:
        raise q.SumTingWongException("unknown mode {}".format(mode))

    # train
    optimD = torch.optim.Adam(q.params_of(netD4D), lr=lr, weight_decay=wreg)
    optimG = torch.optim.Adam(q.params_of(netG4G), lr=lr, weight_decay=wreg)
    gantrainer = q.GANTrainer(mode="WGAN-GP", one_sided=True, noise_dim=noisedim,
                              penalty_weight=pw,
                 optimizerD=optimD, optimizerG=optimG, logger=Logger("gan", logiter=logiter))

    for amortizer in amortizers:
        gantrainer.add_dyn_hyperparams(amortizer)

    def pp_scores(x):  # 2D
        if isinstance(x, torch.autograd.Variable):
            x = x.data
        if not isinstance(x, np.ndarray):
            x = x.cpu().numpy()
        for xe in x:
            print(sparkline.sparkify(xe))
        idxseq = np.arange(0, x.shape[1]) % 10
        print("".join([str(a) for a in list(idxseq)]))

    def samplepp(noise=None, cuda=True, gen=testgen, ret_all=False, rawlen=None):
        y, o, score, ograds, gold = sampler(noise=noise, cuda=cuda, gen=gen, rawlen=rawlen)
        y = y.cpu().data.numpy()[:, 1:]
        if ret_all:
            return pp(y), o, score, ograds, gold
        return pp(y)

    print(samplepp(cuda=False, rawlen=40))

    gantrainer.train((netD4D, netD4G), (netG4D, netG4G), niter=niter, niterD=niterD, niterG=niterG,
                     data_gen=traingen, cuda=cuda, netR=netR, sample_real_for_gen=mode in ("normal", "acha"))

    q.embed()


if __name__ == "__main__":
    q.argprun(run)