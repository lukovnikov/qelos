# TODO: CHAR LEVEL RNN LM
import h5py, numpy as np
import qelos as q
import torch
from torch import nn
from IPython import embed


def makemat(data, window, subsample):
    startpositions = np.arange(0, data.shape[0] - window)
    np.random.shuffle(startpositions)
    numex = startpositions.shape[0] // subsample
    startpositions = startpositions[:numex]
    mat = np.zeros((startpositions.shape[0], window), dtype="int64")
    for i in range(startpositions.shape[0]):
        startpos = startpositions[i]
        mat[i, :] = data[startpos:startpos + window]
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

    def pp(charseq):
        if charseq.ndim == 1:
            return "".join([chardic[x] for x in charseq])
        elif charseq.ndim == 2:
            ret = []
            for i in range(len(charseq)):
                ret.append(pp(charseq[i]))
            return ret

    ret = (makemat(train, window, subsample),
           makemat(valid, window, subsample),
           makemat(test, window, subsample),
           chardic, pp)
    tt.tock("made mats")
    return ret


def run(window=100, subsample=10000, inspectdata=False,
        embdim=200,
        encdim=300,
        layers=2,
        rnu="memgru",      # "gru" or "ppgru" or "memgru"
        zoneout=0.2,
        dropout=0.1,
        lr=0.1,
        gradnorm=5.,
        batsize=100,
        epochs=100,
        hardmem=True
        ):
    np.random.seed(1337)
    trainmat, validmat, testmat, rcd, pp = loaddata_text8(window=window, subsample=subsample)
    #for x in pp(trainmat[:10]): print x
    if inspectdata:
        embed()

    # config
    vocsize = max(rcd.keys()) + 1
    if rnu == "gru":
        rnu = lambda bdim, ndim: q.GRUCell(bdim, ndim, dropout_in=dropout).to_layer()
    elif rnu == "memgru":
        rnu = lambda bdim, ndim: q.MemGRUCell(bdim, ndim, memsize=5, dropout_in=dropout).to_layer()

    chardic = {"<MASK>": 0}
    chardic.update({v: k for k, v in rcd.items()})
    # make model
    emb = q.WordEmb(embdim, worddic=chardic)
    recm = q.RecurrentStack(
        q.persist_kwargs(),
        emb,
        q.argmap.spec(0, mask=1),
        rnu(embdim, encdim),
    ).return_final()

    m = q.Stack(q.persist_kwargs(), recm, nn.Linear(encdim, vocsize), nn.LogSoftmax())

    # print(m)

    # train
    trainloader = q.dataload(trainmat[:, :-1], trainmat[:, -1], batch_size=batsize, shuffle=True)
    validloader = q.dataload(validmat[:, :-1], validmat[:, -1], batch_size=batsize)

    trainlosses = q.lossarray(nn.NLLLoss())
    validlosses = q.lossarray(nn.NLLLoss())

    optim = torch.optim.Adadelta(q.params_of(m), lr=lr)

    q.train(m)\
        .train_on(trainloader, trainlosses)\
        .valid_on(validloader, validlosses)\
        .optimizer(optim).clip_grad_norm(gradnorm)\
        .train(epochs)


if __name__ == "__main__":
    q.argprun(run)