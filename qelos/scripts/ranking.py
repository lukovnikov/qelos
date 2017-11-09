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

    def forward(self, ldata, rdata):
        lvecs = self.lmodel(ldata)      # 2D
        rvecs = self.rmodel(rdata)      # 2D
        sim = self.sim(lvecs, rvecs)    # 1D:(batsize,)
        return sim


class InpFeeder(object):
    def __init__(self, ldata, rdata, eid2lid, eid2rids, eid2rid_gold=None):
        super(InpFeeder, self).__init__()
        self.ldata = ldata if q.issequence(ldata) else (ldata,)
        self.rdata = rdata if q.issequence(rdata) else (rdata,)
        self.eid2lid = eid2lid
        self.eid2rids = eid2rids
        self.eid2rid_gold = eid2rid_gold

    def __call__(self, eids):
        eids = eids.cpu().data.numpy()
        ldata = []
        rdata = []
        for eid in eids:
            lid = self.eid2lid[eid]
            rids = self.eid2rids[eid]
            if self.eid2rid_gold is not None:



def run(lr=0.1,
        rankmode="gold",       # "gold" or "score"
        ):
    left_model = None       # model takes something, produces vector
    right_model = None      # model takes something, produces vector

    rankmodel = RankModel(left_model, right_model)

    ldata = None            # should be tensor
    rdata = None            # should be tensor
    rscores = None          # should be vector of scores for each rdata 0-axis slice

    eid2lid = {}            # mapping from example ids to ldata ids
    eid2rid_gold = {}       # mapping from example ids to rdata ids for gold
    eid2rids = {}           # maps from example ids to sets of rdata ids

    eids = np.arange(0, len(ldata))
    eidsloader = q.dataload(eids)




if __name__ == "__main__":
    q.argprun(run)