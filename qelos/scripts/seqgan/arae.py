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
        if (_iter+0) % self.logiter == 0:
            self.tt.live("[{}/{}] Loss_D: {:.4f} Loss_G: {:.4f} Score Real: {:.4f} Score Fake: {:.4f} Loss Lip: {:.4f}"
                         .format(_iter, niter, errD, errG, scoreD_real, scoreD_fake, lip_loss))


def get_data_gen(vocsize, batsize, seqlen):
    def data_gen():
        while True:
            yield np.random.randint(0, vocsize, (batsize, seqlen)).astype("int64")
    return data_gen


def makenets(vocsize, embdim, encdim, noisedim, seqlen, startsym):
    emb = nn.Embedding(vocsize, embdim)

    enccell = q.GRUCell(embdim, encdim)
    encoder = q.RecurrentStack(emb, enccell.to_layer()).return_final()

    deccell = q.ContextDecoderCell(nn.Embedding(vocsize, embdim),
                                   q.GRUCell(embdim + encdim, encdim),
                                   nn.Linear(encdim, vocsize, bias=False),
                                   q.LogSoftmax())
    decoder = deccell.to_decoder()

    netG = q.Stack(q.Forward(noisedim, encdim, activation="relu"),
                   q.Forward(encdim, encdim, activation="relu"),
                   q.Forward(encdim, encdim, activation="relu"))

    netD = q.Stack(q.Forward(encdim, encdim, activation="relu"),
                   q.Forward(encdim, encdim, activation="relu"),
                   q.Forward(encdim, 1, activation="relu"),
                   q.Lambda(lambda x: x[:, 0]))

    class AE(nn.Module):
        def __init__(self):
            super(AE, self).__init__()
            self.enc = encoder
            self.dec = decoder

        def forward(self, x, y):
            encoding = self.enc(x)
            decoding = self.dec(y, encoding)
            return decoding

    ae = AE()

    def sample(cuda=False, rawlen=seqlen):
        noise = q.var(torch.randn(1, noisedim)).cuda(cuda).v
        noise.data.normal_(0, 1)

        noise = netG(noise)

        startsymbols = np.zeros((1,), dtype="int64")
        startsymbols[0] = startsym
        startsymbols = q.var(startsymbols).cuda(cuda).v
        deccell.teacher_unforce(startsymbols=startsymbols)

        dec = decoder(noise, maxtime=rawlen)

        deccell.teacher_force()

        _, ret = torch.max(dec, 2)

        return ret


    return (encoder, decoder, ae), (netD, netG, encoder), sample


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


def run(lrgan=0.0003,
        lrae=0.1,
        wreg=0.000001,
        embdim=64,
        noisedim=64,
        encdim=256,
        batsize=256,
        niter=100,
        niterD=5,
        niterG=1,
        seqlen=32,
        cuda=False,
        gpu=1,
        subsample=1000,
        inspectdata=False,
        pw=10,
        temperature=1.,
        logiter=50,
        ):
    if cuda:
        torch.cuda.set_device(gpu)

    print("niter: {}".format(niter))
    # get data and dict
    # datagen = get_data_gen(vocsize, batsize, seqlen)()
    trainmat, validmat, testmat, rcd, pp = loaddata_text8(window=seqlen, subsample=subsample)
    cd = {v: k for k, v in rcd.items()}
    traingen = q.makeiter(q.dataload(trainmat, batch_size=batsize, shuffle=True))
    validgen = q.makeiter(q.dataload(validmat, batch_size=batsize, shuffle=False))
    testgen = q.makeiter(q.dataload(testmat, batch_size=batsize, shuffle=False))

    vocsize = np.max(trainmat) + 1

    print("Number of training sequences: {}, vocabulary of {}"
          .format(trainmat.shape[0], vocsize))

    if inspectdata:
        embed()

    # build networks
    startsym = cd["<START>"]
    (encoder, decoder, ae), (netD, netG, netR), sampler \
        = makenets(vocsize, embdim, encdim, noisedim, seqlen, startsym)

    def samplepp(cuda=True, rawlen=None):
        y = sampler(cuda=cuda, rawlen=rawlen)
        y = y.cpu().data.numpy()[:, 1:]
        return pp(y)

    print(samplepp(cuda=False, rawlen=40))

    # gan training
    optimD = torch.optim.Adam(q.params_of(netD), lr=lrgan, weight_decay=wreg)
    optimG = torch.optim.Adam(set(q.params_of(netG)) | set(q.params_of(netR)), lr=lrgan, weight_decay=wreg)
    gantrainer = q.GANTrainer(mode="WGAN-GP", one_sided=True, noise_dim=noisedim,
                              penalty_weight=pw,
                 optimizerD=optimD, optimizerG=optimG, logger=Logger("gan", logiter=logiter))

    # ae traning
    aedata = q.dataload(trainmat, batch_size=batsize, shuffle=True)
    aelosses = q.lossarray(q.SeqNLLLoss(time_average=False))
    ae_bt = lambda x: (x, x[:, :-1], x[:, 1:])
    aeoptim = torch.optim.Adadelta(q.params_of(ae), lr=lrae, weight_decay=wreg)

    aetrainer = q.aux_train(ae).train_on(aedata, aelosses)\
        .set_batch_transformer(ae_bt).optimizer(aeoptim).cuda(cuda)\
        .initialize()
    aetrainer.logiter = logiter

    # chain
    gantrainer.chain_trainer(aetrainer)

    # def pp_scores(x):  # 2D
    #     if isinstance(x, torch.autograd.Variable):
    #         x = x.data
    #     if not isinstance(x, np.ndarray):
    #         x = x.cpu().numpy()
    #     for xe in x:
    #         print(sparkline.sparkify(xe))
    #     idxseq = np.arange(0, x.shape[1]) % 10
    #     print("".join([str(a) for a in list(idxseq)]))
    #

    gantrainer.train(netD, netG, niter=niter, niterD=niterD, niterG=niterG,
                     data_gen=traingen, cuda=cuda, netR=netR)

    q.embed()


if __name__ == "__main__":
    q.argprun(run)