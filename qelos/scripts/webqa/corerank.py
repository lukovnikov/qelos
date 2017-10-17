import qelos as q
import torch
import numpy as np
from collections import OrderedDict
import dill as pickle
import re

# region building representations
def zerobasespecialglovecloneoverride(dim, dic, gloveemb):
    baseemb = q.ZeroWordEmb(dim=dim, worddic=dic)
    oogtokens = set(dic.keys()) - set(gloveemb.D.keys())
    oogtokens = ["<MASK>", "<RARE>"] + list(oogtokens)
    oogtokensdic = dict(zip(oogtokens, range(len(oogtokens))))
    oogtokenemb = q.WordEmb(dim=dim, worddic=oogtokensdic)
    emb = baseemb.override(oogtokenemb)
    gloveclonedic = set(dic.keys()) & set(gloveemb.D.keys()) - {"<MASK>", "<RARE>"}
    gloveclonedic = dict(zip(gloveclonedic, range(len(gloveclonedic))))
    gloveclone = gloveemb.subclone(gloveclonedic)
    emb = emb.override(gloveclone)
    return emb


def get_reps(dim=50, glovedim=50, shared_computers=False,
             mergemode="sum", relwhich="urlwords",
             loaded_questions=None,
             lexmatp="../../../../datasets/webqa/webqa.corechains.lexmats"
        ):
    tt = q.ticktock("loader")
    qpids, questionsm, querysm = loaded_questions
    print(len(qpids), questionsm.matrix.shape, querysm.matrix.shape)

    tt.tick("loading lexinfo mats")
    lexinfo = pickle.load(open(lexmatp))
    tt.tock("loaded")

    gloveemb = q.PretrainedWordEmb(glovedim, incl_maskid=False, incl_rareid=False)

    tt.tick("building reps")
    # region get NL reps     # TODO: <RARE> should be trainable, <MASK> can also be from specials
    nl_emb = zerobasespecialglovecloneoverride(glovedim, questionsm.D, gloveemb)
    # endregion

    rel_emb = get_rel_emb(dim=dim, gloveemb=gloveemb, mode=mergemode, which=relwhich)


def load_relation_info_mats(p=None):
    pass
    # TODO !!!


def get_rel_emb(dim, gloveemb, mode="cat", which=("urlwords",)):
    """ Makes composite computed embeddings for relations """
    glovedim = gloveemb.vecdim
    embdim = glovedim
    if mode == "cat":
        dim = dim // len(which)
    reldic, relinfo = load_relation_info_mats()

    tomerge = []

    if "name" in which:
        # 1. encode name
        rel_name_base_inner_emb = q.WordEmb(dim=glovedim, worddic=relinfo.names.D)
        rel_name_inner_emb = rel_name_base_inner_emb.override(gloveemb)
        rel_name_inner_enc = q.RecurrentStack(
            q.persist_kwargs(),
            rel_name_inner_emb,
            q.argmap.spec(0, mask=1),
            q.GRULayer(glovedim, dim).return_final("only"),
            #q.GRUCell(dim, dim).to_layer(),
        )
        rel_name_emb = q.ComputedWordEmb(data=relinfo.names.matrix, computer=rel_name_inner_enc, worddic=reldic)
        tomerge.append(rel_name_emb)

    if "urlwords" in which:
        # 2. encode url with words
        rel_urlwords_base_inner_emb = q.WordEmb(dim=glovedim, worddic=relinfo.urlwords.D)
        rel_urlwords_inner_emb = rel_urlwords_base_inner_emb.override(gloveemb)
        rel_urlwords_inner_enc = q.RecurrentStack(
            q.persist_kwargs(),
            rel_urlwords_inner_emb,
            q.argmap.spec(0, mask=1),
            q.GRULayer(glovedim, dim).return_final("only"),
            # q.GRUCell(dim, dim).to_layer(),
        )
        rel_urlwords_emb = q.ComputedWordEmb(data=relinfo.urlwords.matrix, computer=rel_urlwords_inner_enc, worddic=reldic)
        tomerge.append(rel_urlwords_emb)

    if "urltokens" in which:
        # 3. encode url with tokens
        rel_urltokens_inner_emb = q.WordEmb(dim=embdim, worddic=relinfo.urltokens.D)
        rel_urltokens_comp = q.RecurrentStack(
            q.persist_kwargs(),
            rel_urltokens_inner_emb,
            q.argmap.spec(0, mask=1),
            q.GRULayer(embdim, dim).return_final("only"),
            # q.GRUCell(dim, dim).to_layer(),
        )
        rel_urltokens_emb = q.ComputedWordEmb(data=relinfo.urltokens.matrix, computer=rel_urltokens_comp, worddic=reldic)
        tomerge.append(rel_urltokens_emb)

    if "domainids" in which:
        # 4. embed domain ids and range ids
        rel_domainids_inner_emb = q.WordEmb(dim=dim, worddic=relinfo.domainids.D)
        rel_domainids_emb = q.ComputedWordEmb(data=relinfo.domainids.matrix[:, 0],
                                              computer=rel_domainids_inner_emb,
                                              worddic=reldic)
        tomerge.append(rel_domainids_emb)

    if "rangeids" in which:
        rel_rangeids_inner_emb = q.WordEmb(dim=dim, worddic=relinfo.rangeids.D)
        rel_rangeids_emb = q.ComputedWordEmb(data=relinfo.rangeids.matrix[:, 0],
                                             computer=rel_rangeids_inner_emb,
                                             worddic=reldic)
        tomerge.append(rel_rangeids_emb)

    if "domainwords" in which:
        # 5. encode domain words
        rel_domainwords_base_inner_emb = q.WordEmb(dim=glovedim, worddic=relinfo.domainwords.D)
        rel_domainwords_inner_emb = rel_domainwords_base_inner_emb.override(gloveemb)
        rel_domainwords_inner_enc = q.RecurrentStack(
            q.persist_kwargs(),
            rel_domainwords_inner_emb,
            q.argmap.spec(0, mask=1),
            q.GRULayer(glovedim, dim).return_final("only"),
            # q.GRUCell(dim, dim).to_layer(),
        )
        rel_domainwords_emb = q.ComputedWordEmb(data=relinfo.domainwords.matrix, computer=rel_domainwords_inner_enc, worddic=reldic)

        tomerge.append(rel_domainwords_emb)

    if "rangewords" in which:
        # 6. encode range words
        rel_rangewords_base_inner_emb = q.WordEmb(dim=glovedim, worddic=relinfo.rangewords.D)
        rel_rangewords_inner_emb = rel_rangewords_base_inner_emb.override(gloveemb)
        rel_rangewords_inner_enc = q.RecurrentStack(
            q.persist_kwargs(),
            rel_rangewords_inner_emb,
            q.argmap.spec(0, mask=1),
            q.GRULayer(glovedim, dim).return_final("only"),
            # q.GRUCell(dim, dim).to_layer(),
        )
        rel_rangewords_emb = q.ComputedWordEmb(data=relinfo.rangewords.matrix, computer=rel_rangewords_inner_enc, worddic=reldic)

        tomerge.append(rel_rangewords_emb)

    assert(len(tomerge) > 0)
    rel_emb = tomerge[0]
    for tomerge_e in tomerge[1:]:
        rel_emb = rel_emb.merge(tomerge_e, mode=mode)

    return rel_emb, reldic


# endregion


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


def run(lr=0.1,
        cuda=False,
        gpu=1):
    localvars = locals()
    savesettings = "celltype allgiven entityfilter glovedim encdim decdim attmode gradnorm dropout merge_mode batsize epochs rel_which decsplit".split()
    savesettings = OrderedDict({k: localvars[k] for k in savesettings})
    if cuda:
        torch.cuda.set_device(gpu)

    tt = q.ticktock("script")


if __name__ == "__main__":
    q.argprun(run)