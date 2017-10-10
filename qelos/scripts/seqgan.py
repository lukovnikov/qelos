import qelos as q
import torch
from torch import nn
import numpy as np
import h5py
from IPython import embed
import sparkline


class DCB(object):
    def __init__(self, vocsize, dim):
        super(DCB, self).__init__()
        self.weight = nn.Parameter(torch.randn(vocsize, dim))
        self.reset_parameters()
        self.embedder = nn.Embedding(vocsize, dim)
        self.embedder.weight = self.weight
        self.linout = nn.Linear(dim, vocsize, bias=False)
        self.linout.weight = self.weight

    def reset_parameters(self):
        nn.init.xavier_normal(self.weight.data)

    def get_embedder(self):
        return self.embedder

    def get_linout(self):
        return self.linout


class Logger(object):
    def __init__(self, name="", ):
        self.tt = q.ticktock(name)

    def log(self, _iter=None, niter=None, errD=None, errG=None, scoreD_real=None, scoreD_fake=None, lip_loss=None, **kw):
        if (_iter+1) % 2 == 0:
            self.tt.live("[{}/{}] Loss_D: {:.4f} Loss_G: {:.4f} Score Real: {:.4f} Score Fake: {:.4f} Loss Lip: {:.4f}"
                         .format(_iter, niter, errD, errG, scoreD_real, scoreD_fake, lip_loss))



def get_data_gen(vocsize, batsize, seqlen):
    def data_gen():
        while True:
            yield np.random.randint(0, vocsize, (batsize, seqlen)).astype("int64")
    return data_gen


def make_nets_mine():
    dcb = DCB(vocsize, embdim)

    g_rnu = q.GRUCell(embdim+noisedim, gendim)

    # generator
    netG4D = q.ContextDecoderCell(
        dcb.get_embedder(),
        g_rnu,
        dcb.get_linout(),
        nn.LogSoftmax(),
        q.Lambda(lambda x: torch.max(x, 1)[1])
    )
    netG4D.teacher_unforce(seqlen, block=lambda x: x[0], startsymbols=1)
    netG4D = netG4D.to_decoder()

    netG4G = q.ContextDecoderCell(
        dcb.get_embedder(),
        g_rnu,
    )
    netG4G.teacher_unforce(seqlen,
                           q.Stack(
                               dcb.get_linout(),
                               nn.LogSoftmax(),
                               q.Lambda(lambda x: torch.max(x, 1)[1])
                           ),
                           startsymbols=1)
    netG4G = netG4G.to_decoder()

    # discriminator
    d_rnn = q.GRULayer(embdim, discdim)
    d_scorer = nn.Linear(discdim, 1)

    netD4D = q.RecurrentStack(
        dcb.get_embedder(),
        d_rnn,
    ).return_final()
    netD4D = q.Stack(netD4D, d_scorer)

    netD4G = q.RecurrentStack(
        d_rnn,
    ).return_final()
    netD4G = q.Stack(netD4G, d_scorer)

    # real processing
    netR = q.RecurrentStack(
        dcb.get_embedder(),
    )


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
            out = torch.cat([override[:, :-amortizer], dec[:, -amortizer:]], 1)
        return out


def make_nets_normal(vocsize, embdim, gendim, discdim, startsym,
                     seqlen, noisedim, soft_next=False, temperature=1., mode="acha-sync"):

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
                             q.Lambda(lambda x: torch.max(x[0], 1)[1]) if not soft_next else q.Lambda(lambda x: x[0])
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
        q.Forward(discdim, discdim, activation="relu"),
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
        if gen is None or rawlen is not None:
            data = q.var(np.ones((1, seqlen), dtype="int64") * startsym).cuda(cuda).v
        else:
            data = next(gen)[0:1]
            gold = data
            data = q.var(data).cuda(cuda).v
        if rawlen is not None:
            netG.decoder.block.teacher_unforce(rawlen, q.Lambda(lambda x: torch.max(x[0], 1)[1]), data.data[0][0])
            o = netG.decoder(noise, maxtime=rawlen)
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


def run(lr=0.00005,
        embdim=128,
        noisedim=128,
        gendim=256,
        discdim=256,
        batsize=256,
        niter=-1,
        niterD=10,
        niterG=1,
        seqlen=32,
        cuda=False,
        gpu=1,
        mode="normal",       # "normal"
        subsample=1000,
        inspectdata=False,
        pw=10,
        temperature=1.,
        ):
    if cuda:
        torch.cuda.set_device(gpu)

    if niter == -1:
        niter = (seqlen * 200 + 500) + 500        # seqlen * amortize_step + amortize_headstart + afterburn
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
    if mode == "normal":
        (netD4D, netD4G), (netG4D, netG4G), netR, sampler, amortizers \
            = make_nets_normal(vocsize, embdim, gendim, discdim,
                               cd["<START>"], seqlen, noisedim, temperature=temperature,
                               mode="clth")
    elif mode == "mine":
        raise NotImplemented()
        (netD4D, netD4G), (netG4D, netG4G), netR = make_nets_mine()
    else:
        raise q.SumTingWongException("unknown mode {}".format(mode))

    # train
    optimD = torch.optim.RMSprop(q.params_of(netD4D), lr=lr)
    optimG = torch.optim.RMSprop(q.params_of(netG4G), lr=lr)
    gantrainer = q.GANTrainer(mode="WGAN-GP", one_sided=True, noise_dim=noisedim,
                              penalty_weight=pw,
                 optimizerD=optimD, optimizerG=optimG, logger=Logger("gan"))

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
        y = y.cpu().data.numpy()
        if ret_all:
            return pp(y), o, score, ograds, gold
        return pp(y)

    # print(samplepp(cuda=False, rawlen=30))

    gantrainer.train((netD4D, netD4G), (netG4D, netG4G), niter=niter, niterD=niterD, niterG=niterG,
                     data_gen=traingen, cuda=cuda, netR=netR,
                     sample_real_for_gen=True)

    q.embed()


if __name__ == "__main__":
    q.argprun(run)