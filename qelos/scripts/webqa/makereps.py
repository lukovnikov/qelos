import qelos as q
from qelos.scripts.webqa.preprocessing.loader \
    import load_questions_inone, load_entity_info_mats, load_relation_info_mats

import numpy as np


defaultp = "../../../datasets/webqsp/flmats/"
defaultqp = "../../../datasets/webqsp/webqsp"


def run(x=0):
    question_sm, query_sm, qids, tx_sep = load_questions_inone(p=defaultqp)
    # test
    src_emb, tgt_emb, tgt_lin = get_all_reps(question_sm=question_sm, query_sm=query_sm)

    x = q.var(np.asarray(
        [tgt_emb.D[":film.actor.film"], tgt_emb.D[":film.performance.character"]])).v

    test_tgt_emb_whole, _ = tgt_emb(x)

    relurlwordencoder = tgt_emb.over

    test_tgt_emb_relurlwords, _ = relurlwordencoder(x)

    assert(np.allclose(test_tgt_emb_whole.data.numpy(), test_tgt_emb_relurlwords.data.numpy()))

    print("same for rel")

    x = q.var(np.asarray(
        [tgt_emb.D["m.017gm7"], tgt_emb.D["m.017gm7"]])).v
    test_tgt_emb_whole, _ = tgt_emb(x)
    entemb = tgt_emb.base.over
    test_tgt_emb_entemb, _ = entemb(x)
    assert(np.allclose(test_tgt_emb_whole.data.numpy(), test_tgt_emb_entemb.data.numpy()))

    print("same for ent")


def get_all_reps(dim=50, glovedim=50, merge_mode="cat", rel_which=("urlwords",),
                 question_sm=None, query_sm=None):
    gloveemb = q.PretrainedWordEmb(glovedim, incl_maskid=False, incl_rareid=False)

    src_emb = get_nl_emb(question_sm, gloveemb)
    tgt_emb = get_fl_emb(query_sm, dim, gloveemb,
                         computedwhat=q.ComputedWordEmb,
                         ent_merge_mode=merge_mode,
                         rel_merge_mode=merge_mode,
                         rel_which=rel_which)
    tgt_lin = get_fl_emb(query_sm, dim, gloveemb,
                         computedwhat=q.ComputedWordLinout,
                         ent_merge_mode=merge_mode,
                         rel_merge_mode=merge_mode,
                         rel_which=rel_which)

    assert(tgt_emb.D == tgt_lin.D)
    print("tgt emb dic and tgt lin dic consistent")

    return src_emb, tgt_emb, tgt_lin


def get_nl_emb(nl_sm, gloveemb):
    """ For questions.
        Takes stringmatrix, builds base emb of dim dim based on its dict
        and overrides by provided glove emb """
    glovedim = gloveemb.vecdim
    baseemb = q.WordEmb(dim=glovedim, worddic=nl_sm.D)
    emb = baseemb.override(gloveemb)
    return emb


def get_fl_emb(fl_sm, dim, gloveemb, computedwhat=q.ComputedWordEmb,
               ent_merge_mode="cat",
               rel_merge_mode="cat", rel_which=("urlwords",)):

    ent_emb, entdic = get_ent_emb(dim, gloveemb, mode=ent_merge_mode, computedwhat=computedwhat)
    rel_emb, reldic = get_rel_emb(dim, gloveemb, mode=rel_merge_mode, computedwhat=computedwhat,
                          which=rel_which)

    basedict = {}
    basedict.update(fl_sm.D)
    nextvalididx = max(basedict.values()) + 1
    for k, v in reldic.items():
        if not k in basedict:
            basedict[k] = nextvalididx
            nextvalididx += 1
    for k, v in entdic.items():
        if not k in basedict:
            basedict[k] = nextvalididx
            nextvalididx += 1

    for k, v in fl_sm.D.items():
        assert(basedict[k] == v)
    print("fl_sm and basedict consistent")

    if computedwhat == q.ComputedWordEmb:
        baseemb = q.WordEmb(dim=dim, worddic=basedict)       # base embedding (to be overridden by computed embeddings for entities and relations)
    elif computedwhat == q.ComputedWordLinout:
        baseemb = q.WordLinout(indim=dim, worddic=basedict)
    else:
        raise q.SumTingWongException()

    emb = baseemb.override(ent_emb).override(rel_emb)
    for k, v in fl_sm.D.items():
        assert(emb.D[k] == v)
    print("fl_sm and tgt's emb.D consistent")
    return emb


def get_ent_emb(dim, gloveemb, mode="cat", computedwhat=q.ComputedWordEmb):
    """ Makes composite computed embeddings for entities """
    glovedim = gloveemb.vecdim
    if mode == "cat":
        dim = dim // 2
    entdic, entinfo = load_entity_info_mats(p=defaultp)
    # entdic maps ent ids to idx, entinfo contains info about ent ids indexed by entdic's idx
    # 1. encode entity name
    ent_name_base_inner_emb = q.WordEmb(dim=glovedim, worddic=entinfo.names_word.D)
    ent_name_inner_emb = ent_name_base_inner_emb.override(gloveemb)
    ent_name_inner_enc = q.RecurrentStack(
        q.persist_kwargs(),
        ent_name_inner_emb,
        q.argmap.spec(0, mask=1),
        q.GRULayer(glovedim, dim).return_final("only"),
        #q.GRUCell(dim, dim).to_layer(),
    )
    ent_name_emb_computer = ent_name_inner_enc
    ent_name_emb = computedwhat(data=entinfo.names_word.matrix, computer=ent_name_emb_computer, worddic=entdic)
    # 2. encode entity notable type
    ent_notabletype_base_inner_emb = q.WordEmb(dim=glovedim, worddic=entinfo.notabletypes_word.D)
    ent_notabletype_inner_emb = ent_notabletype_base_inner_emb.override(gloveemb)
    ent_notabletype_inner_enc = q.RecurrentStack(
        q.persist_kwargs(),
        ent_notabletype_inner_emb,
        q.argmap.spec(0, mask=1),
        q.GRULayer(glovedim, dim).return_final("only"),
        #q.GRUCell(dim, dim).to_layer(),
    )
    ent_notabletype_emb_computer = ent_notabletype_inner_enc
    ent_notabletype_emb = computedwhat(data=entinfo.notabletypes_word.matrix, computer=ent_notabletype_emb_computer, worddic=entdic)
    # 3. merge
    ent_emb = ent_name_emb.merge(ent_notabletype_emb, mode=mode)
    return ent_emb, entdic


def get_rel_emb(dim, gloveemb, mode="cat", which=("urlwords",), computedwhat=q.ComputedWordEmb):
    """ Makes composite computed embeddings for relations """
    glovedim = gloveemb.vecdim
    embdim = glovedim
    if mode == "cat":
        dim = dim // len(which)
    reldic, relinfo = load_relation_info_mats(p=defaultp)

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
        rel_name_emb = computedwhat(data=relinfo.names.matrix, computer=rel_name_inner_enc, worddic=reldic)
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
        rel_urlwords_emb = computedwhat(data=relinfo.urlwords.matrix, computer=rel_urlwords_inner_enc, worddic=reldic)
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
        rel_urltokens_emb = computedwhat(data=relinfo.urltokens.matrix, computer=rel_urltokens_comp, worddic=reldic)
        tomerge.append(rel_urltokens_emb)

    if "domainids" in which:
        # 4. embed domain ids and range ids
        rel_domainids_inner_emb = q.WordEmb(dim=dim, worddic=relinfo.domainids.D)
        rel_domainids_emb = computedwhat(data=relinfo.domainids.matrix[:, 0],
                                              computer=rel_domainids_inner_emb,
                                              worddic=reldic)
        tomerge.append(rel_domainids_emb)

    if "rangeids" in which:
        rel_rangeids_inner_emb = q.WordEmb(dim=dim, worddic=relinfo.rangeids.D)
        rel_rangeids_emb = computedwhat(data=relinfo.rangeids.matrix[:, 0],
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
        rel_domainwords_emb = computedwhat(data=relinfo.domainwords.matrix, computer=rel_domainwords_inner_enc, worddic=reldic)

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
        rel_rangewords_emb = computedwhat(data=relinfo.rangewords.matrix, computer=rel_rangewords_inner_enc, worddic=reldic)

        tomerge.append(rel_rangewords_emb)

    assert(len(tomerge) > 0)
    rel_emb = tomerge[0]
    for tomerge_e in tomerge[1:]:
        rel_emb = rel_emb.merge(tomerge_e, mode=mode)

    return rel_emb, reldic


if __name__ == "__main__":
    q.PretrainedWordEmb.defaultpath = "../data/glove/miniglove.%dd"
    q.argprun(run)