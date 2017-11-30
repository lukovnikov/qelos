import qelos as q
import torch
from torch import nn
import numpy as np
import random
import re
import ujson


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


def load_data(p="../../../../datasets/lcquad/lcquad.multilin",
              cp="lcquad.multilin.chains"):
    # get gold chains and questions
    questions = {}
    goldchains = {}
    with open(p) as f:
        for line in f:
            m = re.match("(Q\d+):(.+)", line)
            if m:
                qid = m.group(1)
                question = m.group(2)
                questions[qid] = question
            m = re.match("(Q\d+\.P\d+):\s(.+)", line)
            if m:
                qpid = m.group(1)
                qp = m.group(2)
                qps = qp.split()[1:]
                incore = True
                goldchains[qpid] = []
                dontadd = False
                for qpse in qps:
                    if qpse in "<<BRANCH>> <<EQUALS>> <<COUNT>>".split():
                        incore = False
                    elif qpse in "<<RETURN>> <<JOIN>>".split():
                        incore = True
                        dontadd = True
                    if incore and not dontadd:
                        qpse_m = re.match(":(-?)<http://dbpedia\.org/(?:ontology|property)/([^>]+)>", qpse)
                        if qpse_m:
                            qpse_ = "{}{}".format(qpse_m.group(1), qpse_m.group(2))
                        else:
                            qpse_m_type = re.match(":(-?)<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>", qpse)
                            qpse_ = "{}{}".format(qpse_m_type.group(1), "<type>")
                        goldchains[qpid].append(qpse_)
    # load candidate core chains
    oldchains = ujson.load(open(cp))
    newchains = {}
    for qpid, chains in oldchains.items():
        newchains[qpid] = []
        for chain in chains:
            newchain = []
            for chaine in chain:
                qpse_m = re.match(":(-?)<http://dbpedia\.org/(?:ontology|property)/([^>]+)>", chaine)
                if qpse_m:
                    qpse_ = "{}{}".format(qpse_m.group(1), qpse_m.group(2))
                else:
                    qpse_m_type = re.match(":(-?)<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>", chaine)
                    qpse_ = "{}{}".format(qpse_m_type.group(1), "<type>")
                newchain.append(qpse_)
            newchains[qpid].append(newchain)
    chains = newchains
    # check if goldchains are in chains
    # q.embed()
    notinchains = set()
    for qpid, goldchain in goldchains.items():
        inchains = False
        for chain in chains[qpid]:
            if chain == goldchain:
                inchains = True
        if not inchains:
            notinchains.add(qpid)
    if len(notinchains) > 0:
        print("{} questions don't have gold chain in core chains".format(len(notinchains)))
    else:
        print("all questions have corechains")
    return questions, goldchains, chains


def run_preload(d=0):
    questions, goldchains, chains = load_data()
    tosave = {}
    tosave["questions"] = questions
    tosave["goldchains"] = goldchains
    tosave["chains"] = chains
    ujson.dump(tosave, open("lcquad.multilin.chains.preload", "w"))


def load_preloaded(p="lcquad.multilin.chains.preload"):
    # load
    preload = ujson.load(open(p))
    questions = preload["questions"]
    goldchains = preload["goldchains"]
    chains = preload["chains"]

    # process questions
    qsm = q.StringMatrix()
    questions = sorted(questions.items(), key=lambda (x, y): x)
    qids = []
    for qid, question in questions:
        qids.append(qid)
        qsm.add(question)
    qsm.finalize()

    # process chains
    # get all unique relations and chains
    goldchains = sorted(goldchains.items(), key=lambda (x, y): x)
    qpids = []
    eids2goldchains = {}
    set_unique_goldchains = set()
    unique_goldchains = []
    eid = 0
    for qpid, goldchain in goldchains:
        qpids.append(qpid)

        eid += 1


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

    load_preloaded()

    ldata = None            # should be tensor
    rdata = None            # should be tensor
    rscores = None          # should be vector of scores for each rdata 0-axis slice

    eid2lid = {}            # mapping from example ids to ldata ids
    eid2rid_gold = {}       # mapping from example ids to rdata ids for gold
    eid2rids = {}           # maps from example ids to sets of rdata ids

    numex = len(ldata)

    #################
    rankmodel = RankModel(left_model, right_model, similarity)

    eids = np.arange(0, len(numex))
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
    # q.argprun(run)
    q.argprun(run_preload)