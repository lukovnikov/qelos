import numpy as np
import sparkline
import qelos as q
import torch
from torch import nn


def test_model(encoder, decoder, m, questions, queries, vnt):
    batsize = 10
    questions = q.var(questions[:batsize]).v
    queries = q.var(queries[:batsize]).v
    vnt = q.var(vnt[:batsize]).v

    # region try encoder
    ctx, ctxmask, finalctx = encoder(questions)
    # print(ctx.size())
    assert(ctx.size(0) == finalctx.size(0))
    assert(ctx.size(1) == ctxmask.float().size(1))
    assert(ctx.size(2) == finalctx.size(1))
    maskedctx = ctx * ctxmask.unsqueeze(2).float()
    assert((ctx.norm(2) == maskedctx.norm(2)).data.numpy()[0])
    # print(ctx.norm(2) - maskedctx.norm(2))
    loss = finalctx.sum()
    loss.backward()
    encoder.zero_grad()
    ctx, ctxmask, finalctx = encoder(questions)
    loss = finalctx.sum()
    loss.backward()
    encparams = q.params_of(encoder)

    print("dry run of encoder didn't throw errors")
    # endregion
    # region try decoder
    # decoder.block.embedder = nn.Embedding(100000, 200, padding_idx=0)
    # decoder.block.smo = q.Stack(
    #                                      q.persist_kwargs(),
    #                                      q.argsave.spec(mask={"mask"}),
    #                                      q.argmap.spec(0),
    #                                      nn.Linear(200, 11075),
    #                                      q.argmap.spec(0, mask=["mask"]),
    #                                      q.LogSoftmax(),
    #                                      q.argmap.spec(0),
    #                     )
    # decoder.block.smo = None
    # try decoder cell
    for t in range(3):
        # ctx, ctxmask, finalctx = encoder(questions)
        decoder.block.core.reset_state()        # ESSENTIAL !!! otherwise double .backward() error
        decoder.set_init_states(finalctx.detach())
        decoder.block.zero_grad()
        outmaskt=vnt[:, t]
        # outmaskt=q.var(np.ones_like(vnt[:, t].data.numpy()).astype("int64")).v
        y_t, att_t = decoder.block(queries[:, t], ctx.detach(), ctxmask=ctxmask.detach(), t=t, outmask_t=outmaskt)
        loss = torch.max(y_t)
        print(loss)
        loss.backward()
        decparams = q.params_of(decoder)
        print("backward done")
    print("dry run of decoder cell didn't throw errors")
    # endregion

    # region whole model
    m.zero_grad()
    dec, atts, outmask = m(questions, queries[:, :-1], vnt[:, 1:])
    assert(dec.size(0) == questions.size(0))
    assert(dec.size(1) == queries.size(1) - 1)
    # assert(dec.size(2) == 11075)
    loss = dec.sum()
    loss.backward()
    allparams = q.params_of(m)
    assert(len(set(allparams) - set(decparams) - set(encparams)) == 0)
    print("dry run of whole model didn't throw errors")
    # endregion
    # q.embed()


def make_encoder(src_emb, embdim=100, dim=100, dropout=0.0, **kw):
    """ make encoder
    # concatenating bypass encoder:
    #       embedding  --> top GRU
    #                  --> 1st BiGRU
    #       1st BiGRU  --> top GRU
    #                  --> 2nd BiGRU
    #       2nd BiGRU  --> top GRU
    """
    statemergefwd = q.Forward(dim, dim, use_bias=False)
    encoder = q.RecurrentStack(
        q.persist_kwargs(),
        src_emb,        # embs, masks
        q.argsave.spec(emb=0, mask=1),
        q.argmap.spec(0, mask=["mask"]),
        q.BidirGRULayer(embdim, dim/2),
        q.TimesharedDropout(dropout),
        q.argsave.spec(bypass=0),
        q.argmap.spec(0, mask=["mask"]),
        q.BidirGRULayer(dim, dim/2).return_final(True),
        # q.TimesharedDropout(dropout),
        q.argsave.spec(final=0),
        q.argmap.spec(1, ["bypass"]),
        q.Lambda(lambda x, y: x + y),
        statemergefwd,        # to mix up fwd and rev states
        # q.argmap.spec(0, mask=["mask"]),
        # q.GRULayer(dim * 2, dim).return_final(True),
        q.argmap.spec(0, ["mask"], ["final"]),
    )   # returns (all_states, mask, final_state)
    encoder = q.Stack(
        q.persist_kwargs(),
        encoder,
        q.argsave.spec(all=0, mask=1, final=2),
        q.argmap.spec(2),
        statemergefwd,
        q.argmap.spec(["all"], ["mask"], 0),
    )
    return encoder


class make_decoder(object):
    def __init__(self, mode="webqsp"):
        if mode == "webqsp":
            self.branch_token, self.join_token = "<BRANCH>", "<JOIN>"
        elif mode == "lcquad":
            self.branch_token, self.join_token = "<<BRANCH>>", "<<JOIN>>"

    def __call__(self, emb, lin, ctxdim=100, embdim=100, dim=100,
                     attmode="bilin", decsplit=False, celltype="normal", **kw):
        """ makes decoder
        # attention cell decoder that accepts VNT !!!
        """
        ctxdim = ctxdim if not decsplit else ctxdim // 2
        coreindim = embdim + ctxdim     # if ctx_to_decinp is True else embdim

        coretocritdim = dim if not decsplit else dim // 2
        critdim = coretocritdim + embdim          # if decinp_to_att is True else dim

        if attmode == "bilin":
            attention = q.Attention().bilinear_gen(ctxdim, critdim)
        elif attmode == "fwd":
            attention = q.Attention().forward_gen(ctxdim, critdim, dim)
        else:
            raise q.SumTingWongException()

        attcellkw = q.kw2dict(
                             attention=attention,
                             embedder=emb,
                             core=q.RecStack(
                                 q.persist_kwargs(),
                                 q.GRUCell(coreindim, dim),
                                 # q.GRUCell(dim, dim),
                             ),
                             smo=q.Stack(
                                 q.persist_kwargs(),
                                 # q.argsave.spec(mask={"mask"}),
                                 lin,
                                 # q.argmap.spec(0, mask=["mask"]),
                                 # q.LogSoftmax(),
                                 # q.argmap.spec(0),
                             ),
                             att_after_update=True,
                             ctx_to_decinp=True,
                             ctx_to_smo=True,
                             state_to_smo=True,
                             decinp_to_att=True,
                             state_split=decsplit,
                             return_att=True,
                             return_out=True,)
        if celltype == "tree":
            attcellkw(structure_tokens=(emb.D[self.branch_token], emb.D[self.join_token]))
            attcell = q.HierarchicalAttentionDecoderCell(**attcellkw.v)
        elif celltype == "normal":
            attcell = q.AttentionDecoderCell(**attcellkw.v)
        else:
            raise q.SumTingWongException("unknown cell type: {}".format(celltype))
        return attcell.to_decoder()


class Model(nn.Module):
    def __init__(self, encoder, decoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, srcseq, tgtseq, outmask=None):
        ctx, ctxmask, finalstate = self.encoder(srcseq)
        self.decoder.set_init_states(finalstate)
        dec, att = self.decoder(tgtseq, ctx, ctxmask=ctxmask, outmask=outmask)
        return dec, att, outmask


class ErrorAnalyzer(q.LossWithAgg):
    callwithinputs = True

    def __init__(self, questionD, queryD, trainquerymat, mode="webqsp"):
        super(ErrorAnalyzer, self).__init__()
        self.questionD, self.queryD = {v: k for k, v in questionD.items()}, \
                                      {v: k for k, v in queryD.items()}
        self.global_entmistakes = 0
        self.global_entmistakes_notintrain = 0
        self.global_relmistakes = 0
        self.global_relmistakes_notintrain = 0
        self.global_othermistakes = 0
        self.global_count = 0
        self.global_entities = 0
        self.global_relations = 0
        self.global_others = 0

        self.traintokens = {self.queryD[x] for x in set(np.unique(trainquerymat))}

        self.acc = []

        self.mode = mode
        if mode == "webqsp":
            from qelos.scripts.webqa.preprocessing.buildvnt import category, ENT, REL
            self.categorizer, self.ENT, self.REL = category, ENT, REL
        elif mode == "lcquad":
            from qelos.scripts.webqa.lcquad.buildvnt import category, ENT, REL
            self.categorizer, self.ENT, self.REL = category, ENT, REL

    def __call__(self, pred, gold, inputs=None):    # (batsize, seqlen, outvocsize)
        pred, att, mask = pred
        for i in range(len(pred)):
            exampleresult = self.processexample(pred[i], att[i], gold[i], mask[i],
                    inputs=([inputs_e[i] for inputs_e in inputs] if inputs is not None else None))
            self.acc.append(exampleresult)
        return 0

    def processexample(self, pred, att, gold, outmask, inputs=None):
        """ compute how many entity mistakes, relation mistakes;
            store predictions, gold, and inputs in string form
            compute probability of correct sequence
        """
        res = {}

        pred = pred.cpu().data.numpy()      # (seqlen, probs)
        gold = gold.cpu().data.numpy()
        att = att.cpu().data.numpy()
        outmask = outmask.cpu().data.numpy()
        mask = (gold != 0)
        inp = inputs[0].cpu().data.numpy() if inputs is not None else None

        pred[outmask == 0] = -np.infty
        toppred = np.argmax(pred, axis=1)

        # transform to strings
        def pp(seq, dic):
            seq = [dic[x] for x in seq]
            seq = filter(lambda x: x != "<MASK>", seq)
            seq_str = " ".join(seq)
            return seq_str, seq

        question_str, question_seq = pp(inp, self.questionD) if inp is not None else None
        toppred_str, toppred_seq = pp(toppred, self.queryD)
        gold_str, gold_seq = pp(gold, self.queryD)

        # save strings
        res["question_str"] = question_str
        res["toppred_str"] = toppred_str
        res["gold_str"] = gold_str
        res["gold_seq"] = gold_seq
        res["mask"] = mask

        # compute entity/relations/other mistakes
        relmistakes = []
        relmistakes_notintrain = []
        entmistakes = []
        entmistakes_notintrain = []
        othermistakes = []
        totalentities = 0
        totalrelations = 0
        totalothers = 0
        for i in range(len(gold_seq)):
            gold_seq_elem = gold_seq[i]
            pred_seq_elem = toppred_seq[i]
            if self.categorizer(gold_seq_elem) == self.REL:  # relation
                if pred_seq_elem != gold_seq_elem:
                    relmistakes.append((gold_seq_elem, pred_seq_elem))
                    if gold_seq_elem not in self.traintokens:
                        relmistakes_notintrain.append(gold_seq_elem)
                totalrelations += 1
            elif self.categorizer(gold_seq_elem) == self.ENT:    # entity
                if pred_seq_elem != gold_seq_elem:
                    entmistakes.append((gold_seq_elem, pred_seq_elem))
                    if gold_seq_elem not in self.traintokens:
                        entmistakes_notintrain.append(gold_seq_elem)
                totalentities += 1
            else:
                if pred_seq_elem != gold_seq_elem:
                    othermistakes.append((gold_seq_elem, pred_seq_elem))
                totalothers += 1

        self.global_entmistakes += len(entmistakes)
        self.global_entmistakes_notintrain += len(entmistakes_notintrain)
        self.global_relmistakes += len(relmistakes)
        self.global_relmistakes_notintrain += len(relmistakes_notintrain)
        self.global_othermistakes += len(othermistakes)
        self.global_entities += totalentities
        self.global_relations += totalrelations
        self.global_others += totalothers

        res["entmistakes"] = entmistakes
        res["relmistakes"] = relmistakes
        res["othermistakes"] = othermistakes

        # probability of gold sequence, probability of top pred
        goldprobs = pred[np.arange(0, len(pred)), gold] * mask
        toppredprobs = pred[np.arange(0, len(pred)), toppred] * mask
        goldprob = np.sum(goldprobs)
        toppredprob = np.sum(toppredprobs)
        res["goldprobs"] = goldprobs        # (seqlen,)
        res["goldprob"] = goldprob
        res["toppredprobs"] = toppredprobs  # (seqlen,)
        res["toppredprob"] = toppredprob

        res["attention_scores"] = att

        self.global_count += 1

        return res

    def summary(self):
        ret = ""
        ret += "Total examples: {}\n".format(self.global_count)
        ret += "Total entity mistakes: {}({})/{}\n".format(self.global_entmistakes, self.global_entmistakes_notintrain, self.global_entities)
        ret += "Total relation mistakes: {}({})/{}\n".format(self.global_relmistakes, self.global_relmistakes_notintrain, self.global_relations)
        ret += "Total other mistakes: {}/{}\n".format(self.global_othermistakes, self.global_others)
        # print(ret)
        return ret

    def inspect(self):
        i = 0
        while i < len(self.acc):
            res = self.acc[i]
            msg = "Question:\t{}\nPrediction:\t{:.4f} - {}\nGold:   \t{:.4f} - {}\n"\
                .format(res["question_str"],
                        res["toppredprob"], res["toppred_str"],
                        res["goldprob"], res["gold_str"])
            msg += "Seen in train:  {}\n".format(str([1 if x in self.traintokens else 0 for x in res["gold_seq"] if x != 0]))
            msg += "Top pred probs: {}\n".format(sparkline.sparkify(-res["toppredprobs"]).encode("utf-8"))
            msg += "Gold probs:     {}\n".format(sparkline.sparkify(-res["goldprobs"]).encode("utf-8"))
            # attention scores
            goldwords = ["<E0>"] + res["gold_str"].split()
            decwords = res["toppred_str"].split()
            maxlen = 30
            msg += "\t{} - {}\n".format(" "*maxlen, res["question_str"])
            for j, (goldword, decword) in enumerate(zip(goldwords[:-1], decwords)):
                if len(decword) > maxlen:
                    decword = decword[-maxlen:]
                if len(goldword) > maxlen:
                    goldword = goldword[-maxlen:]
                msg += "\t{:^{maxlenn}.{maxlenn}s}".format(goldword, maxlenn=maxlen) \
                       + " -> {} -> ".format(sparkline.sparkify(res["attention_scores"][j]).encode("utf-8")) \
                       + "\t{:^{maxlenn}.{maxlenn}s}".format(decword, maxlenn=maxlen)\
                       + "\n"
            msg += "\t{:^{maxlenn}.{maxlenn}s}\n".format(goldwords[-1], maxlenn=maxlen)
            print(msg)
            rawinp = raw_input("ENTER to continue, 'q'+ENTER to exit:> ")
            if rawinp == "q":
                break
            i += 1

    # region LossWithAgg interface
    def get_agg_error(self):
        return 0

    def reset_agg(self):
        self.acc = []
        self.global_entmistakes = 0
        self.global_relmistakes = 0
        self.global_othermistakes = 0
        self.global_count = 0
        self.global_entities = 0
        self.global_relations = 0
        self.global_others = 0

    def cuda(self, *a, **kw):
        pass
    # endregion


class allgiven_adjust_vnt(object):
    def __init__(self, mode="webqsp"):
        if mode == "webqsp":
            self.special_tokens = "<BRANCH> <ARGMAX> <ARGMIN> <RETURN> <JOIN> <BRANCH> <TIME-NOW>".split()
        elif mode == "lcquad":
            self.special_tokens = "<<BRANCH>> <<JOIN>> <<COUNT>> <<EQUALS>> <<RETURN>>".split()

    def __call__(self, queries, vnt, queryD, force_special_enable=False):
        vnt_filter = np.zeros((vnt.shape[0], vnt.shape[2]), dtype=vnt.dtype)
        for i in range(queries.shape[0]):
            for j in range(queries.shape[1]):
                vnt_filter[i, queries[i, j]] = 1
        # force enable
        if force_special_enable:
            for token in self.special_tokens:
                vnt_filter[:, queryD[token]] = 1
        newvnt = vnt * vnt_filter[:, np.newaxis, :]
        # q.embed()
        return newvnt


class get_corechain(object):
    def __init__(self, mode="webqsp"):
        if mode == "webqsp":
            self.jointoken = "<JOIN>"
            self.branchtoken = "<BRANCH>"
        elif mode == "lcquad":
            self.jointoken = "<<JOIN>>"
            self.branchtoken = "<<BRANCH>>"
        self.masktoken = "<MASK>"

    def __call__(self, querymat, vntmat, queryd):
        print("getting only corechains")
        branchid = queryd[self.branchtoken]
        joinid = queryd[self.jointoken]
        maskid = queryd[self.masktoken]
        maxlen = 0
        for i in range(querymat.shape[0]):
            k = 0
            inbranch = False
            for j in range(querymat.shape[1]):
                sym = querymat[i, j]
                if sym == branchid:
                    inbranch = True
                elif sym == joinid:
                    inbranch = False
                else:
                    if not inbranch:
                        querymat[i, k] = querymat[i, j]
                        vntmat[i, k, :] = vntmat[i, j, :]
                        assert(vntmat[i, k, querymat[i, k]] == 1)
                        k += 1
                if sym != maskid:
                    maxlen = max(maxlen, k + 1)
        querymat = querymat[:, :maxlen]
        vntmat = vntmat[:, :maxlen, :]
        return querymat, vntmat

