import qelos as q
import torch
from torch import nn
import numpy as np
import random


class RankModel(nn.Module):
    def __init__(self, lmodel, rmodel, similarity, **kw):
        super(RankModel, self).__init__(**kw)
        self.lmodel = lmodel
        self.rmodel = rmodel
        self.sim = similarity

    def forward(self, ldata, posrdata, negrdata):
        ldata = ldata if q.issequence(ldata) else (ldata,)
        posrdata = posrdata if q.issequence(posrdata) else (posrdata,)
        negrdata = negrdata if q.issequence(negrdata) else (negrdata,)
        lvecs = self.lmodel(*ldata)      # 2D
        rvecs = self.rmodel(*posrdata)      # 2D
        nrvecs = self.rmodel(*negrdata)
        psim = self.sim(lvecs, rvecs)    # 1D:(batsize,)
        nsim = self.sim(lvecs, nrvecs)
        return psim - nsim


class InpFeeder(object):
    def __init__(self, ldata, rdata, eid2lid, eid2rids, eid2rid_gold=None, scores=None):
        """ if eid2rid_gold is specified, doing "gold" mode
        else doing "score" mode and scores must be specified """
        super(InpFeeder, self).__init__()
        self.ldata = ldata if q.issequence(ldata) else (ldata,)
        self.rdata = rdata if q.issequence(rdata) else (rdata,)
        self.eid2lid = eid2lid
        self.eid2rids = eid2rids
        self.eid2rid_gold = eid2rid_gold
        self.scores = scores

    def __call__(self, eids):
        eids_np = eids.cpu().data.numpy()
        ldata = []
        posdata = []
        negdata = []
        goldseps = []
        for eid in eids_np:
            lid = self.eid2lid[eid]
            rids = self.eid2rids[eid]
            if self.eid2rid_gold is not None:   # gold mode
                pos_rid = self.eid2rid_gold[eid]
                neg_rid = random.sample(rids - {pos_rid,}, 1)
                goldsep = 1
            else:
                pos_rid = random.sample(rids, 1)
                neg_rid = random.sample(rids - {pos_rid,}, 1)
                pos_score = self.scores[pos_rid]
                neg_score = self.scores[neg_rid]
                if neg_score > pos_score:
                    pos_rid, neg_rid = neg_rid, pos_rid
                    pos_score, neg_score = neg_score, pos_score
                goldsep = pos_score - neg_score
            left_data = tuple([ldat[lid] for ldat in self.ldata])
            ldata.append(left_data)
            right_data = tuple([rdat[pos_rid] for rdat in self.rdata])
            posdata.append(right_data)
            right_data = tuple([rdat[neg_rid] for rdat in self.rdata])
            negdata.append(right_data)
            goldseps.append(goldsep)
        ldata = zip(*ldata)
        ldata = [q.var(np.stack(ldata_e)).cuda(eids).v for ldata_e in ldata]
        posdata = zip(*posdata)
        posdata = [q.var(np.stack(posdata_e)).cuda(eids).v for posdata_e in posdata]
        negdata = zip(*negdata)
        negdata = [q.var(np.stack(negdata_e)).cuda(eids).v for negdata_e in negdata]
        goldseps = q.var(np.asarray(goldseps)).cuda(eids).v
        return ldata, posdata, negdata, goldseps


def run(lr=0.1,
        rankmode="gold",       # "gold" or "score"
        margin=1.,
        wreg=0.00001,
        epochs=100,
        batsize=50,
        ):
    left_model = None       # model takes something, produces vector
    right_model = None      # model takes something, produces vector
    similarity = q.DotDistance()    # computes score

    ldata = None            # should be tensor
    rdata = None            # should be tensor
    rscores = None          # should be vector of scores for each rdata 0-axis slice

    eid2lid = {}            # mapping from example ids to ldata ids
    eid2rid_gold = {}       # mapping from example ids to rdata ids for gold
    eid2rids = {}           # maps from example ids to sets of rdata ids

    #################
    rankmodel = RankModel(left_model, right_model, similarity)

    eids = np.arange(0, len(ldata))
    eidsloader = q.dataload(eids, batch_size=batsize, shuffle=True)

    inptransform = InpFeeder(ldata, rdata, eid2lid, eid2rids,
                             eid2rid_gold=eid2rid_gold if rankmode == "gold" else None,
                             scores=rscores if rankmode == "score" else None)

    losses = q.lossarray(q.PairRankingLoss(margin=margin if rankmode == "gold" else None))

    optim = torch.optim.Adadelta(q.params_of(rankmodel), lr=lr, weight_decay=wreg)

    # TODO: add validation and test
    q.train(rankmodel).train_on(eidsloader, losses)\
        .set_batch_transformer(inptransform)\
        .optimizer(optim)\
        .train(epochs)


if __name__ == "__main__":
    q.argprun(run)