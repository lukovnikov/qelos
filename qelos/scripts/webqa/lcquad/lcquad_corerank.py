import qelos as q
import torch
from torch import nn
import numpy as np
import random
import re
import ujson
from collections import OrderedDict
import os
import dill
import pickle

_test = True


class RankModel(nn.Module):
    def __init__(self, lmodel, rmodel, similarity, catscores=False, **kw):
        super(RankModel, self).__init__(**kw)
        self.lmodel = lmodel
        self.rmodel = rmodel
        self.sim = similarity
        self.catscores = catscores

    def forward(self, ldata, posrdata, negrdata):
        ldata = ldata if q.issequence(ldata) else (ldata,)
        posrdata = posrdata if q.issequence(posrdata) else (posrdata,)
        negrdata = negrdata if q.issequence(negrdata) else (negrdata,)
        lvecs = self.lmodel(*ldata)      # 2D
        rvecs = self.rmodel(*posrdata)      # 2D
        nrvecs = self.rmodel(*negrdata)
        psim = self.sim(lvecs, rvecs)    # 1D:(batsize,)
        nsim = self.sim(lvecs, nrvecs)
        if not self.catscores:
            return psim - nsim
        else:
            return torch.stack([psim, nsim], 1)


class StupidRankModel(RankModel):
    EPS = 1e-6
    def __init__(self, lmodel, rmodel, sim, catscores=False, margin=1., **kw):
        super(StupidRankModel, self).__init__(lmodel, rmodel, sim, catscores=catscores, **kw)
        self.margin = margin

    def forward(self, ldata, posrdata, negrdata):
        ldata = ldata if q.issequence(ldata) else (ldata,)
        posrdata = posrdata if q.issequence(posrdata) else (posrdata,)
        negrdata = negrdata if q.issequence(negrdata) else (negrdata,)
        lvecs_frst, lvecs_scnd, _, term = self.lmodel(*ldata)
        rvecs1, rvecs2 = self.rmodel(*posrdata)
        nrvecs1, nrvecs2 = self.rmodel(*negrdata)
        sim1, sim2 = self.sim(lvecs_frst, rvecs1), self.sim(lvecs_scnd, rvecs2)
        nsim1, nsim2 = self.sim(lvecs_frst, nrvecs1), self.sim(lvecs_scnd, nrvecs2)
        goldterm = (rvecs2.abs().sum(1) > 0).float()   # 1 if two-hop
        zeros = q.var(torch.zeros(sim1.size(0))).cuda(sim1).v
        # print(type(zeros), type(sim1), type(sim2), type(nsim1))
        loss1 = torch.max(zeros, self.margin - (sim1 - nsim1))
        loss2 = torch.max(zeros, self.margin - (sim2 - nsim2))
        losst = - (goldterm * torch.log(term+self.EPS) + (1 - goldterm) * torch.log(1 - term + self.EPS))
        loss = loss1 + goldterm * loss2 + losst
        return loss


class ScoreModel(nn.Module):
    def __init__(self, lmodel, rmodel, similarity, **kw):
        super(ScoreModel, self).__init__(**kw)
        self.lmodel = lmodel
        self.rmodel = rmodel
        self.sim = similarity

    def forward(self, ldata, rdata):
        ldata = ldata if q.issequence(ldata) else (ldata,)
        rdata = rdata if q.issequence(rdata) else (rdata,)
        # q.embed()
        lvecs = self.lmodel(*ldata)      # 2D
        rvecs = self.rmodel(*rdata)      # 2D
        psim = self.sim(lvecs, rvecs)    # 1D:(batsize,)
        return psim


class StupidScoreModel(ScoreModel):
    LRG = 1e6
    def forward(self, ldata, rdata):
        ldata = ldata if q.issequence(ldata) else (ldata,)
        rdata = rdata if q.issequence(rdata) else (rdata,)
        lvecs_frst, lvecs_scnd, _, term = self.lmodel(*ldata)
        rvecs1, rvecs2 = self.rmodel(*rdata)
        lterm = (term > 0).float()                      # 1 if two-hop
        rterms = (rvecs2.abs().sum(1) > 0).float()       # 1 if two-hop
        sim1, sim2 = self.sim(lvecs_frst, rvecs1), self.sim(lvecs_scnd, rvecs2)
        score = torch.abs(lterm - rterms) * self.LRG        # disagreement between number of hops
        score += sim1
        score += sim2 * lterm
        return score


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
                pos_rid = random.sample(self.eid2rid_gold[eid], 1)[0]
                neg_rid = random.sample(rids - {pos_rid,}, 1)[0]
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
            line = line.strip()
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


def load_preloaded(p="lcquad.multilin.chains.preload", allparses=True, cachep="lcquad.multilin.chains.cache"):
    if os.path.exists(cachep):
        return dill.load(open(cachep))
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
    qidsD = dict(zip(qids, range(len(qids))))

    # process chains
    # get all unique relations and chains
    relD = {"<MASK>": 0, "<RARE>": 1}
    relD = OrderedDict(relD)
    goldchains = sorted(goldchains.items(), key=lambda (x, y): x)
    qpids = []
    eids2goldchains = {}
    set_unique_goldchains = set()
    unique_goldchains = []
    eid = 0

    MAXCHAINLEN = 2

    # get all relations
    # relD: maps from relation string to relation id
    chainsD = {}        # maps from tuple of relation ids to id in all_chains
    qpid2goldchainid = {}   # maps from qpid string to gold chain id in all_chains
    qpid2negchainids = {}   # maps from qpid string to set of chain ids in all_chains
    all_chains = []

    for qpid, goldchain in goldchains:
        goldchainids = []
        for goldchainrel in goldchain:
            if not goldchainrel in relD:
                relD[goldchainrel] = len(relD)
            goldchainids.append(relD[goldchainrel])
        goldchainids = goldchainids + [0]*(MAXCHAINLEN-len(goldchainids))     # pad with zeros
        _goldchainids = tuple(goldchainids)
        if _goldchainids not in chainsD:
            chainsD[_goldchainids] = len(chainsD)
            all_chains.append(goldchainids)
            assert(len(all_chains) == len(chainsD))
        qpid2goldchainid[qpid] = chainsD[_goldchainids]
    for qpid, chainses in chains.items():
        for chain in chainses:
            chainids = []
            for rel in chain:
                if not rel in relD:
                    relD[rel] = len(relD)
                chainids.append(relD[rel])
            chainids = chainids + [0]*(MAXCHAINLEN-len(chainids))
            _chainids = tuple(chainids)
            if _chainids not in chainsD:
                chainsD[_chainids] = len(chainsD)
                all_chains.append(chainids)
                assert(len(all_chains) == len(chainsD))
            if qpid not in qpid2negchainids:
                qpid2negchainids[qpid] = set()
            qpid2negchainids[qpid].add(chainsD[_chainids])

    rev_relD = {v: k for k, v in relD.items()}
    # TEST
    for qpid, goldchain in goldchains:
        # get gold chain from indexes
        _goldchainids = all_chains[qpid2goldchainid[qpid]]
        # convert to list of strings of rel names
        _goldchain = [rev_relD[e] for e in _goldchainids if e != 0]
        assert(_goldchain == goldchain)
    print("TEST: goldchain reconstruction worked")
    for qpid, chainses in chains.items():
        chainses = set([" ".join(e) for e in chainses])
        _chainses = set()
        for _chainid in qpid2negchainids[qpid]:
            _chainids = all_chains[_chainid]
            _chainses.add(" ".join([rev_relD[e] for e in _chainids if e != 0]))
        assert(chainses & _chainses == chainses)
    print("TEST: negchain reconstruction worked")

    # do example ids mappings
    eids = range(len(qidsD))
    eid2qid = dict(zip(eids, ["Q{}".format(eid+1) for eid in eids]))
    # q.embed()
    assert(set(eid2qid.values()) & set(qidsD.keys()) == set(qidsD.keys()))  # all Qids covered

    eid2lid = dict(zip(eids, [qidsD[eid2qid[e]] for e in eids]))

    qids2qpids = {}
    for qid in qidsD:
        if qid not in qids2qpids:
            qids2qpids[qid] = set()
        for i in range(2):
            qpid = qid + ".P{}".format(i)
            if qpid in preload["goldchains"]:
                qids2qpids[qid].add(qpid)

    eid2rid_gold = {}
    eid2rid_neg = {}

    for eid in eids:
        if eid not in eid2rid_gold:
            eid2rid_gold[eid] = set()
        qid = eid2qid[eid]
        qpids = qids2qpids[qid]
        if allparses:
            for qpid in qpids:      # core chain of every parse is positive
                eid2rid_gold[eid].add(qpid2goldchainid[qpid])
        else:
            qpid = list(qpids)[0]
            eid2rid_gold[eid].add(qpid2goldchainid[qpid])

        if eid not in eid2rid_neg:
            eid2rid_neg[eid] = set()
        for qpid in qpids:
            eid2rid_neg[eid].update(qpid2negchainids[qpid])
        eid2rid_neg[eid] = eid2rid_neg[eid] - eid2rid_gold[eid]

    chainsm = q.StringMatrix()
    chainsm._dictionary = relD
    chainsm._rd = rev_relD
    chainsm._matrix = np.asarray(all_chains)

    if cachep is not None:
        dill.dump((qsm, chainsm, eid2lid, eid2rid_gold, eid2rid_neg), open(cachep, "w"))

    return qsm, chainsm, eid2lid, eid2rid_gold, eid2rid_neg

    q.embed()


class RankingComputer(object):
    """ computes rankings based on ranking model for full validation/test ranking
        provides separate loss objects to put into lossarray
    """

    def __init__(self, scoremodel, ldata, rdata, eid2lid, eid2rid_gold, eid2rid_neg):
        self.scoremodel = scoremodel
        self.ldata = ldata if q.issequence(ldata) else (ldata,)
        self.rdata = rdata if q.issequence(rdata) else (rdata,)
        self.eid2lid = eid2lid
        self.eid2rid_neg = eid2rid_neg
        self.eid2rid_gold = eid2rid_gold
        self.current_batch = None
        self.current_batch_input = None

    def get_rankings(self, eids):
        # q.embed()
        _eids = eids[0].cpu().data.numpy()
        if self.current_batch_input is None or not np.all(_eids == self.current_batch_input):
            self.compute_rankings(_eids, cuda=eids[0])
            self.current_batch_input = _eids
        return self.current_batch

    def compute_rankings(self, eids, cuda=False):
        # get all pairs to score
        self.current_batch = []
        for eid in eids:
            ldata = []
            rdata = []
            lid = self.eid2lid[eid]
            rids = list(self.eid2rid_gold[eid]) + list(self.eid2rid_neg[eid] - self.eid2rid_gold[eid])
            trueornot = [1]*len(self.eid2rid_gold[eid]) + [0]*len(self.eid2rid_neg[eid] - self.eid2rid_gold[eid])
            for rid in rids:
                left_data = tuple([ldat[lid] for ldat in self.ldata])
                ldata.append(left_data)
                right_data = tuple([rdat[rid] for rdat in self.rdata])
                rdata.append(right_data)
            ldata = zip(*ldata)
            ldata = [q.var(np.stack(ldata_e), volatile=True).cuda(cuda).v for ldata_e in ldata]
            rdata = zip(*rdata)
            rdata = [q.var(np.stack(posdata_e), volatile=True).cuda(cuda).v for posdata_e in rdata]
            scores = self.scoremodel(ldata, rdata)
            _scores = list(scores.cpu().data.numpy())
            ranking = sorted(zip(_scores, rids, trueornot), key=lambda x: x[0], reverse=True)
            self.current_batch.append((eid, lid, ranking))


class RecallAt(q.Loss):
    callwithoriginalinputs = True
    def __init__(self, k, rankcomp=None, take=None, **kw):
        super(RecallAt, self).__init__(**kw)
        self.k = k
        self.take = take
        self.bind(rankcomp)

    def bind(self, x):
        self.ranking_computer = x

    def _forward(self, _w, _z, original_inputs=None, **kw):
        rankings = self.ranking_computer.get_rankings(original_inputs)
        # list or (eid, lid, ranking)
        # ranking is a list of triples (_scores, rids, trueornot)
        ys = []
        for _, _, ranking in rankings:
            topktrue = 0.
            totaltrue = 0.
            for i in range(len(ranking)):
                score, rid, trueornot = ranking[i]
                if i <= self.k:
                    topktrue += trueornot
                if trueornot == 1:
                    totaltrue += 1.
            topktrue = topktrue / totaltrue
            ys.append(topktrue)
        ys = np.stack(ys)
        ret = q.var(ys).cuda(original_inputs).v
        return ret, None


class MRR(q.Loss):
    callwithoriginalinputs = True
    def __init__(self, rankcomp=None, **kw):
        super(MRR, self).__init__(**kw)
        self.bind(rankcomp)

    def bind(self, x):
        self.ranking_computer = x

    def _forward(self, _w, _z, original_inputs=None, **kw):
        rankings = self.ranking_computer.get_rankings(original_inputs)
        # list or (eid, lid, ranking)
        # ranking is a list of triples (_scores, rids, trueornot)
        ys = []
        for _, _, ranking in rankings:
            res = 0
            for i in range(len(ranking)):
                if ranking[i][2] == 1:
                    res = 1./(i+1)
                    break
            ys.append(res)
        ys = np.stack(ys)
        ret = q.var(ys).cuda(original_inputs).v
        return ret, None


def run_dummy(lr=.01,
              margin=1.,
              wreg=0.00001,
              epochs=10000):
    ldata = np.arange(0, 500)
    rdata = np.arange(0, 500)

    D = dict(zip(map(str, range(500)), range(500)))

    eid2lid = dict(zip(range(500), range(500)))
    eid2rid_gold = {}
    eid2rids = {}
    for i in range(500):
        eid2rid_gold[i] = {i,}
        eid2rids[i] = set(np.random.randint(0, 500, (50,))) - {i,}

    _left_model = q.WordEmb(50, worddic=D)       # TODO model takes something, produces vector
    q.set_lr(_left_model, 0.1)
    _right_model = q.WordEmb(50, worddic=D)      # TODO model takes something, produces vector
    # _right_model = _left_model
    left_model = q.Lambda(lambda x: _left_model(x)[0], register_modules=[_left_model])
    right_model = q.Lambda(lambda x: _right_model(x)[0], register_modules=[_right_model])
    similarity = q.DotDistance()    # computes score

    rankmodel = RankModel(left_model, right_model, similarity)

    scoremodel = ScoreModel(left_model, right_model, similarity)
    rankcomp = RankingComputer(scoremodel, ldata, rdata, eid2lid, eid2rid_gold, eid2rids)
    recallat1, recallat5 = RecallAt(1, rankcomp=rankcomp), RecallAt(5, rankcomp=rankcomp)
    mrr = MRR(rankcomp=rankcomp)


    eids = np.arange(0, 500)
    eidstrain, eidsvalid = q.split([eids])

    alleidsloader = q.dataload(eids, batch_size=100, shuffle=True)
    eidsloader = q.dataload(*eidstrain, batch_size=100, shuffle=True)
    eidsvalidloader = q.dataload(*eidsvalid, batch_size=100, shuffle=True)

    inptransform = InpFeeder(ldata, rdata, eid2lid, eid2rids,
                             eid2rid_gold=eid2rid_gold)

    losses = q.lossarray(q.PairRankingLoss(margin=margin))
    validlosses = q.lossarray(q.PairRankingLoss(margin=margin), recallat1, recallat5, mrr)

    optim = torch.optim.Adam(q.params_of(rankmodel), lr=lr, weight_decay=wreg)

    # TODO: add validation and test
    q.train(rankmodel).train_on(eidsloader, losses) \
        .set_batch_transformer(inptransform) \
        .optimizer(torch.optim.Adam, lr=lr, weight_decay=wreg) \
        .valid_on(eidsloader, validlosses) \
        .train(epochs)


def make_relemb(relD, dim=50, dropout=0):
    rev_relD = {v: k for k, v in relD.items()}
    max_id = max(relD.values()) + 1
    direction = [0 if (rev_relD[i][0] == "-" if i in rev_relD else False) else 1
                 for i in range(max_id)]
    direction = np.asarray(direction)
    diremb = q.WordEmb(dim, worddic={"-": 0, "+": 1})
    wordsm = q.StringMatrix()   # matrix of words that constitute label for each rel
    wordsm.tokenize = lambda x: x.split()
    relovers = []
    for i in range(max_id):
        if i not in rev_relD or re.match(r'-?<[^>]+>', rev_relD[i]):
            wordsm.add("<MASK>")
            relovers.append(rev_relD[i])
        else:
            relname = rev_relD[i] if i in rev_relD else None
            relname = relname[1:] if relname[0] == "-" else relname
            relname = re.sub(r"([A-Z])", r" \1", relname).lower()
            wordsm.add(relname)
    wordsm.finalize()
    wordemb = q.WordEmb(dim=dim, worddic=wordsm.D)
    gloveemb = q.PretrainedWordEmb(dim=dim, worddic=wordsm.D, fixed=True)
    wordemb = wordemb.override(gloveemb)

    class RelEmbComp(nn.Module):
        def __init__(self, wordemb, diremb, **kw):
            super(RelEmbComp, self).__init__(**kw)
            self.wordemb = wordemb
            self.diremb = diremb
            self.dense = nn.Linear(wordsm.matrix.shape[1]*dim, dim)
            self.dropout = nn.Dropout(p=dropout)

        def forward(self, inpdata):
            dirs = inpdata[:, 0]
            wordids = inpdata[:, 1:]

            rel_wordembs, rel_wordembs_mask = self.wordemb(wordids)
            rel_wordembs = rel_wordembs.view(rel_wordembs.size(0), -1)
            rel_wordembs = self.dropout(rel_wordembs)
            rel_wordembs = self.dense(rel_wordembs)
            rel_wordembs = nn.Tanh()(rel_wordembs)
            rel_dirembs, rel_dirembs_mask = self.diremb(dirs)
            rel_embs = rel_wordembs + rel_dirembs
            return rel_embs

    relembcomp = RelEmbComp(wordemb, diremb)

    # test computer
    if _test:
        test_words = np.random.randint(2, 20, (10, 4))
        test_words[:, 2:] = 0
        test_words = q.var(test_words).v
        test_dirs = q.var(np.random.randint(0, 2, (10,))).v
        test_inpdata = torch.cat([test_dirs.unsqueeze(1), test_words], 1)
        test_relembs = relembcomp(test_inpdata)
        loss = test_relembs.sum()
        loss.backward()
        print("gradnorm wordemb base: {}".format(wordemb.base.embedding.weight.grad.norm()))
        print("gradnorm wordemb base for mask: {}".format(wordemb.base.embedding.weight.grad[0, :].norm()))
        # print("gradnorm wordemb over: {}".format(wordemb.over.inner.embedding.weight.grad.norm()))
        # print("gradnorm wordemb base for mask: {}".format(wordemb.over.inner.embedding.weight.grad[0, :].norm()))
        print("test backwarded")

    relembcompdata = np.concatenate([direction[:, np.newaxis], wordsm.matrix], axis=1)
    relemb = q.ComputedWordEmb(relembcompdata, relembcomp, worddic=relD)
    reloverdic = dict(zip(relovers, range(len(relovers))))
    relembover = q.WordEmb(dim, worddic=reloverdic)
    relemb = relemb.override(relembover)

    # test relemb
    if _test:
        test_relids = q.var(np.asarray([0, 1, 2, 9, 821, 834])).v
        test_relembs, test_relembs_mask = relemb(test_relids)
        loss = test_relembs.sum()
        loss.backward()
        print("gradnorm for over: {}".format(relembover.embedding.weight.grad.norm()))
        print("relemb test backwarded")

    return relemb


def run(lr=0.001,
        rankmode="gold",       # "gold" or "score"
        margin=1.,
        wreg=0.00001,
        dropout=0.3,
        epochs=10000,
        batsize=50,
        dim=50,
        encdim=100,
        validinter=10,
        cuda=False,
        gpu=0,
        ):
    _test = False
    if cuda:
        torch.cuda.set_device(gpu)
    tt = q.ticktock("script")
    tt.tick("loading data")
    qsm, chainsm, eid2lid, eid2rid_gold, eid2rid_neg = load_preloaded()
    tt.tock("data loaded")
    # q.embed()

    ldata = qsm.matrix            # should be tensor
    rdata = chainsm.matrix            # should be tensor
    rscores = None          # should be vector of scores for each rdata 0-axis slice

    eid2lid = eid2lid            # mapping from example ids to ldata ids
    eid2rid_gold = eid2rid_gold       # mapping from example ids to rdata ids for gold
    eid2rids = eid2rid_neg           # maps from example ids to sets of rdata ids

    max_rids_gold = 0.
    max_rids_neg = 0.
    avg_rids_neg = 0.
    for eid, rid_gold in eid2rid_gold.items():
        max_rids_gold = max(max_rids_gold, len(rid_gold))
        max_rids_neg = max(max_rids_neg, len(eid2rids[eid]))
        avg_rids_neg += len(eid2rids[eid])
    avg_rids_neg /= len(eid2rids)

    print("max nr. of gold rids per eid: {}".format(max_rids_gold))
    print("nr. of neg rids per eid: max: {}, avg: {}".format(max_rids_neg, avg_rids_neg))

    numex = len(ldata)

    #################

    relemb = make_relemb(chainsm.D, dim=dim, dropout=dropout)

    # make left encoder
    leftemb = q.WordEmb(dim, worddic=qsm.D)
    leftemb_pretrained = q.PretrainedWordEmb(dim, worddic=qsm.D, fixed=True)
    leftemb = leftemb.override(leftemb_pretrained)
    # left model
    left_model = q.RecurrentStack(
        leftemb,
        q.wire((-1, 0)),
        q.TimesharedDropout(p=dropout),
        q.wire((-1, 0), mask=(1, 1)),
        q.BidirLSTMLayer(dim, encdim//2, mode="intercat").return_final("only"),
    )

    if _test:
        test_l_encs = left_model(q.var(ldata[:5, :]).v)

    class RightModel(nn.Module):
        def __init__(self, **kw):
            super(RightModel, self).__init__(**kw)
            self.dense = nn.Linear(chainsm.matrix.shape[1]*dim, encdim)
            self.relemb = relemb
            self.dropout = nn.Dropout(p=dropout)

        def forward(self, x):
            rel_embs, rel_embs_mask = self.relemb(x)
            rel_embs = rel_embs.view(rel_embs.size(0), -1)
            rel_embs = self.dropout(rel_embs)
            rel_encs = self.dense(rel_embs)
            rel_encs = nn.Tanh()(rel_encs)
            return rel_encs

    right_model = RightModel()

    if _test:
        test_r_encs = right_model(q.var(rdata[:5, :]).v)

    similarity = q.DotDistance()    # computes score

    rankmodel = RankModel(left_model, right_model, similarity)
    scoremodel = ScoreModel(left_model, right_model, similarity)
    rankcomp = RankingComputer(scoremodel, ldata, rdata, eid2lid, eid2rid_gold, eid2rids)
    recallat1, recallat5 = RecallAt(1, rankcomp=rankcomp), RecallAt(5, rankcomp=rankcomp)
    mrr = MRR(rankcomp=rankcomp)

    #################

    eids = np.arange(0, numex)
    traineids, valideids = q.split([eids], splits=(7, 3), random=False)
    valideids, testeids = q.split(valideids, splits=(1, 2), random=False)

    trainloader = q.dataload(*traineids, batch_size=batsize, shuffle=True)
    validloader = q.dataload(*valideids, batch_size=batsize, shuffle=False)
    testloader = q.dataload(*testeids, batch_size=batsize, shuffle=False)

    inptransform = InpFeeder(ldata, rdata, eid2lid, eid2rids,
                             eid2rid_gold=eid2rid_gold if rankmode == "gold" else None,
                             scores=rscores if rankmode == "score" else None)

    losses = q.lossarray(q.PairRankingLoss(margin=margin if rankmode == "gold" else None))
    validlosses = q.lossarray(q.PairRankingLoss(margin=margin), recallat1, recallat5, mrr)

    optim = torch.optim.Adam(q.paramgroups_of(rankmodel), lr=lr, weight_decay=wreg)

    # TODO: add validation and test
    q.train(rankmodel).train_on(trainloader, losses)\
        .set_batch_transformer(inptransform)\
        .optimizer(optim)\
        .valid_on(validloader, validlosses).valid_inter(validinter) \
        .cuda(cuda).train(epochs)


def make_relemb_stupid(relD, dim=50, dropout=0):
    rev_relD = {v: k for k, v in relD.items()}
    max_id = max(relD.values()) + 1
    direction = [-1. if (rev_relD[i][0] == "-" if i in rev_relD else False) else 1.
                 for i in range(max_id)]
    direction = np.asarray(direction, dtype="float32")
    direction[relD["<MASK>"]] = 0
    diremb = q.WordEmb(dim, worddic={"-": 0, "+": 1})
    wordsm = q.StringMatrix()   # matrix of words that constitute label for each rel
    wordsm.tokenize = lambda x: x.split()
    relovers = []
    for i in range(max_id):
        if i not in rev_relD or rev_relD[i] in "<MASK> <RARE>".split():
            wordsm.add("<MASK>")
            relovers.append(rev_relD[i])
        else:
            relname = rev_relD[i] if i in rev_relD else None
            relname = relname[1:] if relname[0] == "-" else relname
            relname = re.sub(r"([A-Z])", r" \1", relname).lower()
            wordsm.add(relname)
    wordsm.finalize()
    wordemb = q.WordEmb(dim=dim, worddic=wordsm.D)
    gloveemb = q.PretrainedWordEmb(dim=dim, worddic=wordsm.D, fixed=True)
    wordemb = wordemb.override(gloveemb)

    class RelEmbComp(nn.Module):
        def __init__(self, wordemb, diremb, **kw):
            super(RelEmbComp, self).__init__(**kw)
            self.wordemb = wordemb
            self.diremb = diremb
            self.dense = nn.Linear(wordsm.matrix.shape[1]*dim, dim)
            self.dropout = nn.Dropout(p=dropout)

        def forward(self, inpdata):
            wordids = inpdata
            rel_wordembs, rel_wordembs_mask = self.wordemb(wordids)
            rel_wordemb = torch.sum(rel_wordembs, 1) \
                          / (torch.sum(rel_wordembs_mask, 1).float().unsqueeze(1) + 1e-16)
            return rel_wordemb

    relembcomp = RelEmbComp(wordemb, diremb)

    # test computer
    if _test:
        test_words = np.random.randint(2, 20, (10, 4))
        test_words[:, 2:] = 0
        test_words = q.var(test_words).v
        test_dirs = q.var(np.random.randint(0, 2, (10,))).v
        test_inpdata = test_words
        test_relembs = relembcomp(test_inpdata)
        loss = test_relembs.sum()
        loss.backward()
        print("gradnorm wordemb base: {}".format(wordemb.base.embedding.weight.grad.norm()))
        print("gradnorm wordemb base for mask: {}".format(wordemb.base.embedding.weight.grad[0, :].norm()))
        # print("gradnorm wordemb over: {}".format(wordemb.over.inner.embedding.weight.grad.norm()))
        # print("gradnorm wordemb base for mask: {}".format(wordemb.over.inner.embedding.weight.grad[0, :].norm()))
        print("test backwarded")

    relembcompdata = wordsm.matrix
    _relemb = q.ComputedWordEmb(relembcompdata, relembcomp, worddic=relD)
    reloverdic = dict(zip(relovers, range(len(relovers))))
    relembover = q.WordEmb(dim, worddic=reloverdic)
    _relemb = _relemb.override(relembover)

    class RelEmb(nn.Module):
        def __init__(self):
            super(RelEmb, self).__init__()
            self.relemb = _relemb    # only the word part
            self.reldirs = q.val(direction).v

        def forward(self, x):
            rel_embs, rel_embs_mask = self.relemb(x)
            rel_dirs = self.reldirs[x]
            ret = torch.cat([rel_dirs.unsqueeze(1), rel_embs], 1)
            return ret, rel_embs_mask

    relemb = RelEmb()

    # test relemb
    if _test:
        test_relids = q.var(np.asarray([0, 1, 2, 9, 821, 834])).v
        test_relembs, test_relembs_mask = relemb(test_relids)
        loss = test_relembs.sum()
        loss.backward()
        print("gradnorm for over: {}".format(relembover.embedding.weight.grad.norm()))
        print("relemb test backwarded")

    return relemb


def make_relemb_skip(relD, dim=50, dropout=0.):
    rev_relD = {v: k for k, v in relD.items()}
    max_id = max(relD.values()) + 1
    direction = [0 if (rev_relD[i][0] == "-" if i in rev_relD else False) else 1
                 for i in range(max_id)]
    direction = np.asarray(direction)
    diremb = q.WordEmb(dim, worddic={"-": 0, "+": 1})
    wordsm = q.StringMatrix()   # matrix of words that constitute label for each rel
    wordsm.tokenize = lambda x: x.split()
    relovers = []
    for i in range(max_id):
        if i not in rev_relD or re.match(r'-?<[^>]+>', rev_relD[i]):
            wordsm.add("<MASK>")
            relovers.append(rev_relD[i])
        else:
            relname = rev_relD[i] if i in rev_relD else None
            relname = relname[1:] if relname[0] == "-" else relname
            relname = re.sub(r"([A-Z])", r" \1", relname).lower()
            wordsm.add(relname)
    wordsm.finalize()
    wordemb = q.WordEmb(dim=dim, worddic=wordsm.D)
    gloveemb = q.PretrainedWordEmb(dim=dim, worddic=wordsm.D, fixed=True)
    wordemb = wordemb.override(gloveemb)

    class RelEmbComp(nn.Module):
        def __init__(self, wordemb, diremb, **kw):
            super(RelEmbComp, self).__init__(**kw)
            self.wordemb = wordemb
            self.diremb = diremb
            self.dense = nn.Linear(wordsm.matrix.shape[1]*dim, dim)
            self.dropout = nn.Dropout(p=dropout)

        def forward(self, inpdata):
            dirs = inpdata[:, 0]
            wordids = inpdata[:, 1:]

            rel_wordembs, rel_wordembs_mask = self.wordemb(wordids)

            rel_wordemb_basic = torch.sum(rel_wordembs, 1) \
                          / (torch.sum(rel_wordembs_mask, 1).float().unsqueeze(1) + 1e-16)

            rel_wordembs = rel_wordembs.view(rel_wordembs.size(0), -1)
            rel_wordembs = self.dropout(rel_wordembs)
            rel_wordembs = self.dense(rel_wordembs)
            rel_wordembs += rel_wordemb_basic
            rel_wordembs = nn.Tanh()(rel_wordembs)
            rel_dirembs, rel_dirembs_mask = self.diremb(dirs)
            rel_embs = rel_wordembs + rel_dirembs
            return rel_embs

    relembcomp = RelEmbComp(wordemb, diremb)

    # test computer
    if _test:
        test_words = np.random.randint(2, 20, (10, 4))
        test_words[:, 2:] = 0
        test_words = q.var(test_words).v
        test_dirs = q.var(np.random.randint(0, 2, (10,))).v
        test_inpdata = torch.cat([test_dirs.unsqueeze(1), test_words], 1)
        test_relembs = relembcomp(test_inpdata)
        loss = test_relembs.sum()
        loss.backward()
        print("gradnorm wordemb base: {}".format(wordemb.base.embedding.weight.grad.norm()))
        print("gradnorm wordemb base for mask: {}".format(wordemb.base.embedding.weight.grad[0, :].norm()))
        # print("gradnorm wordemb over: {}".format(wordemb.over.inner.embedding.weight.grad.norm()))
        # print("gradnorm wordemb base for mask: {}".format(wordemb.over.inner.embedding.weight.grad[0, :].norm()))
        print("test backwarded")

    relembcompdata = np.concatenate([direction[:, np.newaxis], wordsm.matrix], axis=1)
    relemb = q.ComputedWordEmb(relembcompdata, relembcomp, worddic=relD)
    reloverdic = dict(zip(relovers, range(len(relovers))))
    relembover = q.WordEmb(dim, worddic=reloverdic)
    relemb = relemb.override(relembover)

    # test relemb
    if _test:
        test_relids = q.var(np.asarray([0, 1, 2, 9, 821, 834])).v
        test_relembs, test_relembs_mask = relemb(test_relids)
        loss = test_relembs.sum()
        loss.backward()
        print("gradnorm for over: {}".format(relembover.embedding.weight.grad.norm()))
        print("relemb test backwarded")

    return relemb


def make_stupid_encoder(dim, encdim, qsm, dropout):
    leftemb = q.WordEmb(dim, worddic=qsm.D)
    leftemb_pretrained = q.PretrainedWordEmb(dim, worddic=qsm.D, fixed=True)
    leftemb = leftemb.override(leftemb_pretrained)

    class StupidLeftModel(nn.Module):
        def __init__(self):
            super(StupidLeftModel, self).__init__()
            self.emb = leftemb
            self.dropout = q.TimesharedDropout(p=dropout)
            self.layer = q.BidirLSTMLayer(dim, encdim // 2, mode="intercat").return_final(True)
            self.addr_dense = nn.Linear(encdim, 2)
            self.dirs_dense = nn.Linear(encdim, 1)
            self.term_dense = nn.Linear(encdim, 1)
            self.sm = q.Softmax()

        def forward(self, x):
            embs, mask = self.emb(x)
            embs = self.dropout(embs)
            finalencs, allencs = self.layer(embs, mask=mask)
            scores = self.addr_dense(allencs)
            alphas, _ = self.sm(scores[:, :, 0], mask=mask)
            betas, _ = self.sm(scores[:, :, 1], mask=mask)
            alphasumm = torch.sum(embs * alphas.unsqueeze(2), 1)
            betasumm = torch.sum(embs * betas.unsqueeze(2), 1)
            alphasumm_enc = torch.sum(allencs * alphas.unsqueeze(2), 1)
            betasumm_enc = torch.sum(allencs * betas.unsqueeze(2), 1)
            summ_enc = torch.stack([alphasumm_enc, betasumm_enc], 1)
            dirs = self.dirs_dense(summ_enc)
            dirs = dirs.squeeze(-1)
            dirs = nn.Tanh()(dirs)
            terms = self.term_dense(finalencs)
            terms = terms.squeeze(-1)
            terms = nn.Tanh()(terms)
            return alphasumm, betasumm, dirs, terms

    return StupidLeftModel()


def make_stupid_encoder_skip(dim, encdim, qsm, dropout):
    leftemb = q.WordEmb(dim, worddic=qsm.D)
    leftemb_pretrained = q.PretrainedWordEmb(dim, worddic=qsm.D, fixed=True)
    leftemb = leftemb.override(leftemb_pretrained)

    class StupidLeftModel(nn.Module):
        def __init__(self):
            super(StupidLeftModel, self).__init__()
            self.emb = leftemb
            self.dropout = q.TimesharedDropout(p=dropout)
            self.layer = q.BidirLSTMLayer(dim, encdim // 2, mode="intercat").return_final(True)
            self.addr_dense = nn.Linear(encdim, 2)
            self.dirs_dense = nn.Linear(encdim, 1)
            self.term_dense = nn.Linear(encdim, 1)
            self.reduce_state = nn.Linear(encdim, dim)
            self.sm = q.Softmax()

        def forward(self, x):
            embs, mask = self.emb(x)
            embs = self.dropout(embs)
            finalencs, allencs = self.layer(embs, mask=mask)
            scores = self.addr_dense(allencs)
            alphas, _ = self.sm(scores[:, :, 0], mask=mask)
            betas, _ = self.sm(scores[:, :, 1], mask=mask)
            # weighted embedding sums
            alphasumm = torch.sum(embs * alphas.unsqueeze(2), 1)
            betasumm = torch.sum(embs * betas.unsqueeze(2), 1)
            # weighted encoded sums
            alphasumm_enc = torch.sum(allencs * alphas.unsqueeze(2), 1)
            betasumm_enc = torch.sum(allencs * betas.unsqueeze(2), 1)
            summ_enc = torch.stack([alphasumm_enc, betasumm_enc], 1)
            dirs = self.dirs_dense(summ_enc)
            dirs = dirs.squeeze(-1)
            dirs = nn.Tanh()(dirs)
            terms = self.term_dense(finalencs)
            terms = terms.squeeze(-1)
            terms = nn.Sigmoid()(terms)     # for BCE
            alpharet = alphasumm + self.reduce_state(alphasumm_enc)
            betaret = betasumm + self.reduce_state(betasumm_enc)
            return alpharet, betaret, dirs, terms

    return StupidLeftModel()


class StupidScorer(torch.nn.Module):
    def __init__(self, sim):
        super(StupidScorer, self).__init__()
        self.sim = sim

    def forward(self, lvecs, rvecs):
        # lvecs: tuple of (1st_enc, 2nd_enc, dirs, terms)
        # rvecs: tuple of (1st_enc, 2nd_enc, dirs)
        firstsim = self.sim(lvecs[0], rvecs[0])
        secondsim = self.sim(lvecs[1], rvecs[1])
        sim = firstsim + secondsim * lvecs[-1]
        return firstsim, secondsim, lvecs[-1]


def run_stupid(lr=0.001,
        rankmode="gold",       # "gold" or "score"
        margin=1.,
        wreg=0.00001,
        dropout=0.3,
        epochs=10000,
        batsize=50,
        dim=50,
        encdim=100,
        validinter=10,
        cuda=False,
        gpu=0,
        ):
    _test = False
    if cuda:
        torch.cuda.set_device(gpu)
    tt = q.ticktock("script")
    tt.tick("loading data")
    qsm, chainsm, eid2lid, eid2rid_gold, eid2rid_neg = load_preloaded()
    tt.tock("data loaded")
    # q.embed()

    ldata = qsm.matrix            # should be tensor
    rdata = chainsm.matrix            # should be tensor
    rscores = None          # should be vector of scores for each rdata 0-axis slice

    eid2lid = eid2lid            # mapping from example ids to ldata ids
    eid2rid_gold = eid2rid_gold       # mapping from example ids to rdata ids for gold
    eid2rids = eid2rid_neg           # maps from example ids to sets of rdata ids

    max_rids_gold = 0.
    max_rids_neg = 0.
    avg_rids_neg = 0.
    for eid, rid_gold in eid2rid_gold.items():
        max_rids_gold = max(max_rids_gold, len(rid_gold))
        max_rids_neg = max(max_rids_neg, len(eid2rids[eid]))
        avg_rids_neg += len(eid2rids[eid])
    avg_rids_neg /= len(eid2rids)

    print("max nr. of gold rids per eid: {}".format(max_rids_gold))
    print("nr. of neg rids per eid: max: {}, avg: {}".format(max_rids_neg, avg_rids_neg))

    numex = len(ldata)

    #################

    relemb = make_relemb_skip(chainsm.D, dim=dim, dropout=dropout)

    # make left encoder
    left_model = make_stupid_encoder_skip(dim, encdim, qsm, dropout)

    if _test:
        test_l_encs = left_model(q.var(ldata[:5, :]).v)

    class RightModel(nn.Module):
        def __init__(self, **kw):
            super(RightModel, self).__init__(**kw)
            self.dense = nn.Linear(chainsm.matrix.shape[1]*dim, encdim)
            self.relemb = relemb
            self.dropout = nn.Dropout(p=dropout)

        def forward(self, x):
            rel_embs, rel_embs_mask = self.relemb(x[:, 0])
            rel_embs2, rel_embs_mask2 = self.relemb(x[:, 1])
            #: is rel_embs2 an untrainable all-zeroes if it's a one-hop rel? yes.
            rel_encs = torch.stack([rel_embs, rel_embs2], 2)
            return rel_embs, rel_embs2
            # return rel_embs[:, 1:], rel_embs2[:, 1:], rel_encs[:, 0, :]

    # class RightModel(nn.Module):
    #     def __init__(self, **kw):
    #         super(RightModel, self).__init__(**kw)
    #         self.dense = nn.Linear(chainsm.matrix.shape[1] * dim, encdim)
    #         self.relemb = relemb
    #         self.dropout = nn.Dropout(p=dropout)
    #
    #     def forward(self, x):
    #         rel_embs, rel_embs_mask = self.relemb(x)
    #         rel_embs = rel_embs.view(rel_embs.size(0), -1)
    #         rel_embs = self.dropout(rel_embs)
    #         rel_encs = self.dense(rel_embs)
    #         rel_encs = nn.Tanh()(rel_encs)
    #         return rel_encs
    #
    right_model = RightModel()

    if _test:
        test_r_encs = right_model(q.var(rdata[:5, :]).v)

    similarity = q.DotDistance()  #q.DotDistance()    # computes score
    rankmodel = StupidRankModel(left_model, right_model, similarity, margin=1.)
    scoremodel = StupidScoreModel(left_model, right_model, similarity)
    rankcomp = RankingComputer(scoremodel, ldata, rdata, eid2lid, eid2rid_gold, eid2rids)
    recallat1, recallat5 = RecallAt(1, rankcomp=rankcomp), RecallAt(5, rankcomp=rankcomp)
    mrr = MRR(rankcomp=rankcomp)

    #################

    eids = np.arange(0, numex)
    traineids, valideids = q.split([eids], splits=(7, 3), random=False)
    valideids, testeids = q.split(valideids, splits=(1, 2), random=False)

    trainloader = q.dataload(*traineids, batch_size=batsize, shuffle=True)
    validloader = q.dataload(*valideids, batch_size=batsize, shuffle=False)
    testloader = q.dataload(*testeids, batch_size=batsize, shuffle=False)

    inptransform = InpFeeder(ldata, rdata, eid2lid, eid2rids,
                             eid2rid_gold=eid2rid_gold if rankmode == "gold" else None,
                             scores=rscores if rankmode == "score" else None)

    losses = q.lossarray(q.LinearLoss())
    validlosses = q.lossarray(q.LinearLoss(), recallat1, recallat5, mrr)

    optim = torch.optim.Adam(q.paramgroups_of(rankmodel), lr=lr, weight_decay=wreg)

    # TODO: add validation and test
    q.train(rankmodel).train_on(trainloader, losses)\
        .set_batch_transformer(inptransform)\
        .optimizer(optim)\
        .valid_on(validloader, validlosses).valid_inter(validinter) \
        .cuda(cuda).train(epochs)





if __name__ == "__main__":
    # q.argprun(run_preload)
    # q.argprun(run)
    # q.argprun(run_dummy)
    q.argprun(run_stupid)