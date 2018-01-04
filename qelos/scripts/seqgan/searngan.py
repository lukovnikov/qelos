import qelos as q
import torch
from torch import nn
import numpy as np
import h5py
from IPython import embed
import sparkline
import copy
import re


class Logger(object):
    def __init__(self, name="", logiter=50):
        self.tt = q.ticktock(name)
        self.logiter = logiter

    def log(self, _iter=None, niter=None, errD=None, errG=None, scoreD_real=None, scoreD_fake=None, lip_loss=None, **kw):
        if (_iter+0) % self.logiter == 0:
            self.tt.msg("[{}/{}] Loss_D: {:.4f} Loss_G: {:.4f} Score Real: {:.4f} Score Fake: {:.4f}"
                         .format(_iter, niter, errD, errG, scoreD_real, scoreD_fake))


def get_data_gen(vocsize, batsize, seqlen):
    def data_gen():
        while True:
            yield np.random.randint(0, vocsize, (batsize, seqlen)).astype("int64")
    return data_gen


# region obama loading
def loadobama(p="../../../datasets/langmod/obama.txt", window=32, subsample=1):
    tt = q.ticktock("obama loader")
    tt.tick("loading")
    stracc = ""
    tokens = []
    with open(p) as f:
        for line in f:
            line = q.tokenize(line.strip())
            if len(line) > 0:
                tokens += line
    stracc = " ".join(tokens)
    stracc = re.sub("[\[\]\(\)%&\+\*\<\>]", "", stracc)
    sm = q.StringMatrix(indicate_start_end=True)
    sm.tokenize = lambda x: [xe for xe in x]
    j = 0
    for i in range(len(stracc)):
        if stracc[i] == " ":    # valid start
            j += 1
            if i + window > len(stracc):
                break
            if j == subsample:
                sm.add(stracc[i+1:i+window])
                j = 0
    sm.finalize()
    D = sm.D
    matrix = sm.matrix[:, :-1]

    revdic = {v: k for k, v in D.items()}

    def pp(charseq):
        if charseq.ndim == 1:
            return "".join([revdic[x] for x in charseq])
        elif charseq.ndim == 2:
            ret = []
            for i in range(len(charseq)):
                ret.append(pp(charseq[i]))
            return ret

    select = np.arange(0, matrix.shape[0])
    np.random.shuffle(select)
    trainids = select[:int(len(select)*0.8)]
    validids = select[int(len(select) * 0.8): int(len(select) * 0.9)]
    testids = select[int(len(select) * 0.9):]
    tt.tock("loaded")
    return matrix[trainids], matrix[validids], matrix[testids], revdic, pp



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


def loaddata_hutter(p="../../../datasets/hutter/enwik8.h5", window=200, subsample=1000):
    return loaddata(p=p, window=window, subsample=subsample, utfdic=True)


def loaddata_text8(p="../../../datasets/text8/text8.h5", window=200, subsample=1000):
    return loaddata(p=p, window=window, subsample=subsample, utfdic=False)


def loaddata(p="../../../datasets/hutter/enwik8.h5", window=200, subsample=1000, utfdic=True):
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


def makenets(vocsize, embdim, gendim, discdim, startsym,
             seqlen, noisedim, temperature=1.,
             clrate=0, mode="acha-sync", debug=False):     # "dual" or "single"

    def get_amortizer_update_rule(offset=0, headstart=clrate, interval=clrate):
        def amortizer_update_rule(current_value=0, iter=0, state=None):
            if debug:
                return 5
            if iter < (offset + headstart):
                return 0
            else:
                ret = 1 + (iter - (offset + headstart)) // max(1, interval)
                if current_value != ret:
                    print("AMORTIZED FROM {} TO {}".format(current_value, ret))
                return ret

        return amortizer_update_rule

    amortizer = q.DynamicHyperparam(update_rule=get_amortizer_update_rule())
    amortizers = [amortizer]

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
        nn.Embedding(vocsize, embdim),
        q.GRUCell(embdim + noisedim, gendim),
        nn.Linear(gendim, vocsize),
        q.Softmax(temperature=temperature)
    )

    idxtoonehot = q.IdxToOnehot(vocsize)

    clcutter = get_cl_cutter(amortizer)

    paramdec = [None]

    class Gen(nn.Module):
        def __init__(self, gmode=False):
            super(Gen, self).__init__()
            self.decodercell = decodercell
            self.freeblock = q.Stack(q.Lambda(lambda x: torch.max(x[0], 1)[1]))
            self.decodercell._y_tm1_to_x_t = self.freeblock
            self.decoder = self.decodercell.to_decoder()
            self.gmode = gmode
            self._saved_decselected = None
            self.samplemode = False

        def forward(self, noise, sequences, maxtime=None):
            # generate teacher forcing mask,
            # generate override mask for literal copy
            fseqlen = sequences.size(1)
            # fseqlen = maxtime if maxtime is not None else fseqlen

            teacherforce_mask_np = np.ones((sequences.size(0), fseqlen - 1)).astype("uint8")
            override_mask_np = np.ones((sequences.size(0), fseqlen - 1)).astype("uint8")
            if self.samplemode:
                unforce_lens = np.ones((sequences.size(0),), dtype="int64") * amortizer.v
            else:
                unforce_lens = np.random.randint(0, amortizer.v+1, (sequences.size(0),))
            unforce_lens = np.minimum(unforce_lens, fseqlen - 1)
            for i in range(sequences.size(0)):
                if unforce_lens[i] > 0:
                    teacherforce_mask_np[i, -unforce_lens[i]:] = 0
                override_mask_np[i, -unforce_lens[i] - 1:] = 0
            teacherforce_mask = q.var(teacherforce_mask_np).cuda(sequences).v
            override_mask = q.var(override_mask_np).cuda(sequences).v

            seqs = sequences

            select_ptrs = None

            if self.gmode:   # training generator
                batsize = sequences.size(0)
                teacherforce_mask = teacherforce_mask.unsqueeze(1).repeat(1, vocsize, 1)
                override_mask_np = override_mask.unsqueeze(1).repeat(1, vocsize, 1)\
                    .cpu().data.numpy()
                seqs = sequences.unsqueeze(1).repeat(1, vocsize, 1)
                seqs_np = seqs.cpu().data.numpy()
                select_ptrs_np = np.zeros((batsize,), dtype="int64")
                for i in range(batsize):
                    seqs_np[i, :, -unforce_lens[i]-1] = np.arange(0, vocsize).astype(seqs_np.dtype)
                    select_ptrs_np[i] = -unforce_lens[i]-1
                    override_mask_np[i, :, -unforce_lens[i]-1] = 1
                seqs = q.var(seqs_np).cuda(sequences).v
                override_mask = q.var(override_mask_np).cuda(sequences).v
                select_ptrs = q.var(select_ptrs_np).cuda(sequences).v

                teacherforce_mask = teacherforce_mask.view(batsize * vocsize, -1)
                override_mask = override_mask.view(batsize * vocsize, -1)
                seqs = seqs.view(batsize * vocsize, -1)
                noise = noise.unsqueeze(1).repeat(1, vocsize, 1).view(batsize * vocsize, -1)

            # decode with teacherforce mask
            inpseqs = seqs[:, :-1].contiguous()
            dec = self.decoder(inpseqs, noise, maxtime=maxtime, teacherforce_mask=teacherforce_mask)
            if debug:
                dec = q.var(torch.randn(dec.size()), requires_grad=True).v
                paramdec[0] = dec
                dec = q.Softmax()(dec)

            # select
            decselected = None
            if select_ptrs is not None:

                decreshaped = dec.view(batsize, vocsize, dec.size(1), -1)
                decreshaped = decreshaped[:, 0].contiguous()
                decreshaped = decreshaped.view(batsize * dec.size(1), -1)
                select_ptrs_add = torch.arange(0, batsize).long()
                select_ptrs_add *= dec.size(1)
                select_ptrs = dec.size(1) + select_ptrs
                select_ptrs += q.var(select_ptrs_add).cuda(sequences).v
                decselected = torch.index_select(decreshaped, 0, select_ptrs)

            self._saved_decselected = decselected

            # sample and override
            _, decmax = torch.max(dec, 2)

            if not self.gmode and not self.samplemode:
                deccshape = dec.size()
                deccreshaped = dec.view(-1, dec.size(-1))
                deccsam = torch.multinomial(deccreshaped, 1).squeeze(1)
                deccsam = deccsam.view(deccshape[:-1])
                decmax = deccsam

            overseqs = seqs[:, 1:].contiguous()
            ret = overseqs * override_mask.long() + (1 + (-1) * override_mask.long()) * decmax

            return ret

    # disccell = q.RecStack(
    #     nn.Embedding(vocsize, embdim),
    #     q.GRUCell(embdim, discdim, use_cudnn_cell=True),
    # )
    # discnet = disccell.to_layer()

    discnet = q.RecurrentStack(
        nn.Embedding(vocsize, embdim),
        q.GRULayer(embdim, discdim),
        # q.GRUCell(embdim, discdim, use_cudnn_cell=True).to_layer(),
    )


    disc_summary = q.Stack(
        nn.Linear(discdim, 1),
        nn.Sigmoid(),
    )

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.recnet = discnet
            self.summ = disc_summary

        def forward(self, x):   # sequence of onehot
            recout = self.recnet(x)
            summ = self.summ(recout[:, -1, :])
            return summ.unsqueeze(1)

    netG4D = Gen(gmode=False)
    netG4G = Gen(gmode=True)
    netD = Discriminator()

    samplenetG = copy.deepcopy(netG4D)
    samplenetG.samplemode = True
    samplenetD = copy.deepcopy(netD)

    def sample(noise=None, cuda=False, gen=None, rawlen=None):
        samplenetG.load_state_dict(netG4D.state_dict())
        if noise is None:
            noise = q.var(torch.randn(1, noisedim), volatile=True).cuda(cuda).v
            noise.data.normal_(0, 1)
        if rawlen is not None and gen is None:  # start form startsyms
            data = q.var(np.ones((1, 1), dtype="int64") * startsym, volatile=True).cuda(cuda).v
        else:  # start from data
            data = next(gen)[0:1]
            data = q.var(data, volatile=True).cuda(cuda).v
        if rawlen is not None:
            raise q.SumTingWongException("rawlen not supported yet")
            if gen is None:  # completely unforce, generate rawlen
                samplenetG.decoder.block.teacher_unforce(rawlen, q.Lambda(lambda x: torch.max(x[0], 1)[1]))
                o = samplenetG.decoder(noise, maxtime=rawlen)
            else:  # generate from data up to rawlen
                o = samplenetG(noise, data, maxtime=rawlen)
        else:
            o = samplenetG(noise, data)

        y = o

        samplenetD.load_state_dict(netD.state_dict())
        score = samplenetD(o)
        return y, o[0], score, None, None

    def customgbackward(scores):
        scores = scores.detach()
        saved_decisions = netG4G._saved_decselected
        scores = scores.view(saved_decisions.size())
        # try rescaling scores
        score_mins, _ = torch.min(scores, 1)
        score_maxs, _ = torch.max(scores, 1)
        # scores = (scores - score_mins.unsqueeze(1)) / (score_maxs - score_mins).unsqueeze(1)
        # scores = (scores - 0.5) * 2
        # logscores = -torch.log(scores)
        logdecisions = -torch.log(saved_decisions)
        losses = scores * logdecisions
        # TODO: correct loss needed
        losses = losses.sum(1)
        loss = losses.mean()
        loss.backward()
        if debug:
            print(paramdec[0].grad.norm())
        return loss

    netR = q.Lambda(lambda x: x[:, 1:])

    return (netD, netD), (netG4D, netG4G), netR, sample, amortizers, customgbackward


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


def run(lrd=0.00005,
        lrg=0.00005,
        wreg=0.00000001,
        embdim=128,
        noisedim=128,
        gendim=512,
        discdim=512,
        batsize=256,
        niter=1,
        niterD=5,
        niterG=1,
        seqlen=32,
        cuda=False,
        gpu=1,
        subsample=5,
        inspectdata=False,
        pw=10,
        clrate=1000,
        logiter=50,
        debug=False
        ):
    if cuda:
        torch.cuda.set_device(gpu)
    if debug:
        pass
        inspectdata = True
    if niter == -1:
        # niter = (seqlen * 200 + 500) + 500        # seqlen * amortize_step + amortize_headstart + afterburn
        niter = clrate * (seqlen + 2)
        print("niter: {}".format(niter))
    # get data and dict
    # datagen = get_data_gen(vocsize, batsize, seqlen)()
    # trainmat, validmat, testmat, rcd, pp = loaddata_text8(window=seqlen, subsample=subsample)
    trainmat, validmat, testmat, rcd, pp = loadobama(window=seqlen, subsample=subsample)
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
    (netD4D, netD4G), (netG4D, netG4G), netR, sampler, amortizers, customgbackward \
        = makenets(vocsize, embdim, gendim, discdim,
                           cd["<START>"], seqlen, noisedim,
                           clrate=clrate, debug=debug)

    # train
    optimD = torch.optim.RMSprop(q.params_of(netD4D), lr=lrd, weight_decay=wreg)
    optimG = torch.optim.RMSprop(q.params_of(netG4G), lr=lrg, weight_decay=wreg)
    gantrainer = q.GANTrainer(mode="GAN", noise_dim=noisedim,
                              penalty_weight=pw,
                              optimizerD=optimD, optimizerG=optimG,
                              logger=Logger("gan", logiter=logiter))

    gantrainer.set_custom_g_backward(customgbackward)

    for amortizer in amortizers:
        gantrainer.add_dyn_hyperparams(amortizer)

    def samplepp(noise=None, cuda=False, gen=testgen, ret_all=False, rawlen=None):
        y, o, score, ograds, gold = sampler(noise=noise, cuda=cuda, gen=gen, rawlen=rawlen)
        y = y.cpu().data.numpy()
        if ret_all:
            return pp(y), o, score, ograds, gold
        return pp(y)

    print(samplepp(cuda=False))

    gantrainer.train((netD4D, netD4G), (netG4D, netG4G), niter=niter, niterD=niterD, niterG=niterG,
                     data_gen=traingen, cuda=cuda, sample_real_for_gen=True, netR=netR)

    q.embed()


if __name__ == "__main__":
    q.argprun(run)