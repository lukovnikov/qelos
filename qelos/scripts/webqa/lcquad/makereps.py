import qelos as q
import re, dill as pickle, codecs, editdistance, numpy as np, os
from scipy.sparse import dok_matrix


def load_all(dim=50, glovedim=50, shared_computers=False, mergemode="sum",
             dirp="../../../../datasets/lcquad/",
             qfile="lcquad.multilin",
             lexfile="lcquad.multilin.lex",
             lexmatfile="lcquad.multilin.lexmats",
             vntfile="lcquad.multilin.vnt",
             replace_topic=True, replace_rdftype=True, replace_dbp=True):
    tt = q.ticktock("alloader")
    tt.tick("loading questions and reps")
    loadedq = load_questions(dirp+qfile, dirp+lexfile, replacetopic=replace_topic, replace_rdftype=replace_rdftype, replace_dbp=replace_dbp)
    nl_emb, fl_emb, fl_linout = get_reps(dim=dim, glovedim=glovedim, shared_computers=shared_computers, mergemode=mergemode,
                                         loaded_questions=loadedq, lexmatp=dirp+lexmatfile, replace_dbp=replace_dbp)

    tt.tock("loaded questions and reps").tick("loading vnts")
    vntmatcachep = "lcquad.multilin.vntmat.cache"
    if os.path.isfile(vntmatcachep):
        vntmat = q.load_sparse_tensor(open(vntmatcachep))
    else:
        vntmat = get_vnts(loadedq, fl_emb_d=fl_emb.D, replace_rdftype=replace_rdftype, replace_dbp=replace_dbp, vntp=dirp+vntfile)
        q.save_sparse_tensor(vntmat, open(vntmatcachep, "w"))
    tt.tock("loaded vnts")
    qpids, questionsm, querysm = loadedq

    q.embed()

    return (qpids, questionsm, querysm, vntmat), (nl_emb, fl_emb, fl_linout)


def get_vnts(loaded_questions=None,
        fl_emb_d=None,
        replace_rdftype=True, replace_dbp=True,
        vntp="../../../../datasets/lcquad/lcquad.multilin.vnt"):
    tt = q.ticktock("vnt loader")
    tt.tick("loading vnt file")
    vnt = pickle.load(open(vntp))
    tt.tock("loaded")

    qpids, questionsm, querysm = loaded_questions

    numex = querysm.matrix.shape[0]
    seqlen = querysm.matrix.shape[1]
    vocsize = max(fl_emb_d.values()) + 1
    print(numex, seqlen, vocsize)

    tt.tick("making vnts mat")
    # vntmat = [[dok_matrix((vocsize, 1), dtype="uint8") for i in range(seqlen)] for j in range(numex)]
    maxlen = 0
    for qpid in qpids:
        for timestep_vnt in vnt[qpid]:
            maxlen = max(maxlen, len(timestep_vnt))
    vntmat = np.zeros((numex, seqlen+1, maxlen+2), dtype="int64")
    vntmat[:, :, 2] = 1
    vntmat[:, :, 0] = 1
    vntmat[:, :, 1] = vocsize
    for i, qpid in enumerate(qpids):
        for j, timestep_vnt in enumerate(vnt[qpid]):
            l = 1
            if len(timestep_vnt) > 0:
                vntmat[i, j, 2] = 0
                l = 0
            for timestep_vnt_element in timestep_vnt:
                if replace_dbp:
                    timestep_vnt_element = re.sub("(:-?<http://dbpedia\.org/)property/([^>]+>)",
                                                  "\g<1>ontology/\g<2>", timestep_vnt_element)
                if replace_rdftype:
                    timestep_vnt_element = re.sub(":<http://www\.w3\.org/1999/02/22-rdf-syntax-ns#type>",
                                                  "<<TYPE>>", timestep_vnt_element)
                k = fl_emb_d[timestep_vnt_element]
                vntmat[i, j, l+2] = k+1
                l += 1
            vntmat[i, j, 0] = l     # number of elements
    tt.tock("made")
    # q.embed()
    return vntmat


def get_reps(dim=50, glovedim=50, shared_computers=False, mergemode="sum",
        loaded_questions=None, replace_dbp=True,
        lexmatp="../../../../datasets/lcquad/lcquad.multilin.lexmats"
        ):
    tt = q.ticktock("loader")
    qpids, questionsm, querysm = loaded_questions
    print(len(qpids), questionsm.matrix.shape, querysm.matrix.shape)

    tt.tick("loading lexinfo mats")
    lexinfo = pickle.load(open(lexmatp))
    tt.tock("loaded")

    gloveemb = q.PretrainedWordEmb(glovedim, incl_maskid=False, incl_rareid=False)

    tt.tick("building reps")
    # get NL reps
    baseemb_question = q.ZeroWordEmb(dim=glovedim, worddic=questionsm.D)

    nl_emb = baseemb_question.override(gloveemb)

    # region typ reps
    typsm = lexinfo["typsm"]
    typdic = lexinfo["typdic"]
    baseemb_typmat = q.WordEmb(dim=glovedim, worddic=typsm.D)
    emb_typmat = baseemb_typmat.override(gloveemb)
    typ_rep_inner = q.RecurrentStack(
        emb_typmat,
        q.argmap.spec(0, mask=1),
        q.GRULayer(glovedim, dim).return_final("only")
    )
    typ_emb = q.ComputedWordEmb(data=typsm.matrix, computer=typ_rep_inner, worddic=typdic)
    if not shared_computers:
        typ_rep_inner = q.RecurrentStack(
            emb_typmat,
            q.argmap.spec(0, mask=1),
            q.GRULayer(glovedim, dim).return_final("only")
        )
    typ_linout = q.ComputedWordLinout(data=typsm.matrix, computer=typ_rep_inner, worddic=typdic)
    # endregion

    # region rel reps
    relsm = lexinfo["relsm"]
    reldatamat = relsm.matrix
    reldatadic = relsm.D
    reldic = lexinfo["reldic"]

    if replace_dbp:
        newreldatamat = np.zeros_like(reldatamat)
        newreldic = {}
        i = 0
        for k, v in reldic.items():
            relname = k
            reldata = reldatamat[v]
            m = re.match("(:<http://dbpedia.org/)property/([^>]+>)", k)
            if m:
                relname = m.group(1) + "ontology/" + m.group(2)
            if relname not in newreldic:
                newreldic[relname] = i
                newreldatamat[i, :] = reldata
                i += 1
            else:
                # print(relname, k)
                pass
        reldatamat = newreldatamat[:i]
        reldic = newreldic

    # fwdorrev = np.ones((reldatamat.shape[0]), dtype="int32")
    reldic_rev = {k[0:1]+"-"+k[1:]: v + max(reldic.values()) + 1 for k, v in reldic.items()}
    reldic.update(reldic_rev)
    # fwdorrev = np.concatenate([fwdorrev * 0, fwdorrev * 1], axis=0)
    baseemb_relmat = q.WordEmb(dim=glovedim, worddic=reldatadic)
    emb_relmat = baseemb_relmat.override(gloveemb)

    rel_rep_inner_recu = q.RecurrentStack(
        emb_relmat,
        q.argmap.spec(0, mask=1),
        q.GRULayer(glovedim, dim).return_final("only")
    )
    rel_direction_emb_emb = q.WordEmb(dim=dim, worddic={"FWD": 0, "REV": 1})
    rel_rep_inner = q.Stack(
        q.Lambda(lambda x: (x[:, 0], x[:, 1:])),
        q.argsave.spec(direction=0, content=1),
        q.argmap.spec(["direction"]),
        rel_direction_emb_emb,
        q.argsave.spec(direction_emb=0),
        q.argmap.spec(["content"]),
        rel_rep_inner_recu,
        q.argmap.spec(0, ["direction_emb"]),
        q.Lambda(lambda x, y: x + y),
    )

    reldatamat = np.concatenate([np.zeros((reldatamat.shape[0], 1), dtype="int64"),
                                 reldatamat], axis=1)
    reldatamat_rev = reldatamat + 0
    reldatamat_rev[:, 0] = 1
    reldatamat = np.concatenate([reldatamat, reldatamat_rev], axis=0)

    rel_emb = q.ComputedWordEmb(data=reldatamat, computer=rel_rep_inner, worddic=reldic)

    # rel_emb(q.var(np.asarray([0, 1, 2, 3, 4, 616], dtype="int64")).v)

    if not shared_computers:
        rel_rep_inner_recu = q.RecurrentStack(
            emb_relmat,
            q.argmap.spec(0, mask=1),
            q.GRULayer(glovedim, dim).return_final("only")
        )
        rel_direction_emb_emb = q.WordEmb(dim=dim, worddic={"FWD": 0, "REV": 1})
        rel_rep_inner = q.Stack(
            q.Lambda(lambda x: (x[:, 0], x[:, 1:])),
            q.argsave.spec(direction=0, content=1),
            q.argmap.spec(["direction"]),
            rel_direction_emb_emb,
            q.argsave.spec(direction_emb=0),
            q.argmap.spec(["content"]),
            rel_rep_inner_recu,
            q.argmap.spec(0, ["direction_emb"]),
            q.Lambda(lambda x, y: x + y),
        )
    rel_linout = q.ComputedWordLinout(data=reldatamat, computer=rel_rep_inner, worddic=reldic)
    # endregion

    # region ent reps
    entsm = lexinfo["entsm"]
    entdic = lexinfo["entdic"]
    baseemb_entmat = q.WordEmb(dim=glovedim, worddic=entsm.D)
    emb_entmat = baseemb_entmat.override(gloveemb)
    ent_rep_inner = q.RecurrentStack(
        emb_entmat,
        q.argmap.spec(0, mask=1),
        q.GRULayer(glovedim, dim).return_final("only")
    )
    ent_emb = q.ComputedWordEmb(data=entsm.matrix, computer=ent_rep_inner, worddic=entdic)
    if not shared_computers:
        ent_rep_inner = q.RecurrentStack(
            emb_entmat,
            q.argmap.spec(0, mask=1),
            q.GRULayer(glovedim, dim).return_final("only")
        )
    ent_linout = q.ComputedWordLinout(data=entsm.matrix, computer=ent_rep_inner, worddic=entdic)
    # endregion

    # region ent typ reps
    typtrans = lexinfo["verybesttypes"]
    if not shared_computers:
        typ_rep_inner_for_ent = q.RecurrentStack(
            emb_typmat,
            q.argmap.spec(0, mask=1),
            q.GRULayer(glovedim, dim).return_final("only")
        )
        typ_emb_for_ent = q.ComputedWordEmb(data=typsm.matrix,
                computer=typ_rep_inner_for_ent, worddic=typdic)
    else:
        typ_emb_for_ent = typ_emb
    ent_typ_emb = q.ComputedWordEmb(data=typtrans,
                computer=typ_emb_for_ent, worddic=entdic)

    if not shared_computers:
        typ_rep_inner_for_ent = q.RecurrentStack(
            emb_typmat,
            q.argmap.spec(0, mask=1),
            q.GRULayer(glovedim, dim).return_final("only")
        )
        typ_emb_for_ent = q.ComputedWordEmb(data=typsm.matrix,
                computer=typ_rep_inner_for_ent, worddic=typdic)
    ent_typ_linout = q.ComputedWordLinout(data=typtrans,
        computer=typ_emb_for_ent, worddic=entdic)

    ent_emb_final = ent_emb.merge(ent_typ_emb, mode=mergemode)
    ent_linout_final = ent_linout.merge(ent_typ_linout, mode=mergemode)
    # endregion

    # region merge reps
    basedict = {}
    basedict.update(querysm.D)
    nextvalididx = max(basedict.values()) + 1
    for k, v in reldic.items():
        if not k in basedict:
            basedict[k] = nextvalididx
            nextvalididx += 1
    for k, v in entdic.items():
        if not k in basedict:
            basedict[k] = nextvalididx
            nextvalididx += 1
    for k, v in typdic.items():
        if not k in basedict:
            basedict[k] = nextvalididx
            nextvalididx += 1

    for k, v in querysm.D.items():
        assert (basedict[k] == v)
    print("querysm.D and basedict consistent")

    specialdict = set(basedict.keys()) - set(reldic.keys()) - set(entdic.keys()) - set(typdic.keys())
    specialdict = dict(zip(list(specialdict), range(len(specialdict))))
    special_emb = q.WordEmb(dim=dim, worddic=specialdict)
    special_linout = q.WordLinout(indim=dim, worddic=specialdict)

    fl_emb_base = q.ZeroWordEmb(dim=dim, worddic=basedict)
    fl_emb = fl_emb_base.override(special_emb).override(ent_emb_final)\
                        .override(rel_emb).override(typ_emb)

    fl_linout_base = q.ZeroWordLinout(indim=dim, worddic=basedict)
    fl_linout = fl_linout_base.override(special_linout).override(ent_linout_final)\
                              .override(rel_linout).override(typ_linout)

    for k, v in querysm.D.items():
        assert(fl_emb.D[k] == v)
        assert(fl_linout.D[k] == v)
    print("querysm and tgt's emb.D consistent")
    # endregion
    tt.tock("reps built")

    return nl_emb, fl_emb, fl_linout


def load_questions(p="../../../../datasets/lcquad/lcquad.multilin",
                   lexp="../../../../datasets/lcquad/lcquad.multilin.lex",
                   replacetopic=True, replace_rdftype=True, replace_dbp=True):
    lexinfo = pickle.load(open(lexp))
    labels = lexinfo["labels"]
    # replaces property predicates with ontology !
    xsm = q.StringMatrix()
    ysm = q.StringMatrix()

    xsm.tokenize = lambda x: x.split()
    ysm.tokenize = lambda x: x.split()
    qpids = []

    qid = None
    qpid = None
    question = None
    parse = None
    with codecs.open(p, encoding="utf-8-sig") as f:
        for line in f:
            if len(line) > 0:
                qm = re.match("(?:[^Q]+)?(Q\d+):\s(.+)\n", line)
                if qm:
                    qid = qm.group(1)
                    question = qm.group(2)
                    continue
                pm = re.match("(Q\d+\.P\d+):\s(.+)\n", line)
                if pm:
                    qpid = pm.group(1)
                    parse = pm.group(2)
                    if qpid == "Q27.P1":
                        pass
                    if replace_dbp:
                        # replace property predicates by ontology
                        parse = re.sub("(:-?<http://dbpedia\.org/)property/([^>]+>)", "\g<1>ontology/\g<2>", parse)
                    if replace_rdftype:
                        parse = re.sub(":<http://www\.w3\.org/1999/02/22-rdf-syntax-ns#type>", "<<TYPE>>", parse)
                    topicent = parse.split()[0]
                    topiclabel = labels[topicent][0]
                    # find topic entity in question and replace
                    if replacetopic:
                        try:
                            newquestion, replaced = replace_longest_common_substring_span(topiclabel, "<E0>", question)
                            if len(topiclabel) * 1. / len(replaced) < 2:
                                # print(len(replaced) - len(topiclabel), qpid, newquestion, topiclabel, replaced)
                                # print("\t", parse)
                                xsm.add(newquestion)
                                qpids.append(qpid)
                                ysm.add(parse)
                        except Exception as e:
                            pass
                            # print("NO MATCH", question, qpid, topiclabel)
                            # raise e
                    else:
                        newquestion = question
                        xsm.add(newquestion)
                    continue
    xsm.finalize()
    ysm.finalize()
    return qpids, xsm, ysm


def replace_longest_common_substring_span_re(needle, repl, x):
    xtokens = q.tokenize(x)
    searchtokens = q.tokenize(needle)
    searchtokens = [re.escape(a) for a in searchtokens]
    searchretokens = ["(?:{})?".format(a) for a in searchtokens]
    searchre = "\s?".join(searchretokens)
    finditerator = re.finditer(searchre, " ".join(xtokens))

    longest_start, longest_end = None, None

    for found in finditerator:
        if found.start() < found.end() - 1:
            if longest_end is None or \
                found.end() - found.start() > longest_end - longest_start:
                longest_start, longest_end = found.start(), found.end()
            # print(found, found.start(), found.end())

    if longest_start is None:
        raise q.SumTingWongException("re couldn't find needle")
    joined = " ".join(xtokens)
    replaced = joined[longest_start:longest_end]
    out = joined[:longest_start] + " " + repl + " " + joined[longest_end:]
    out = re.sub("\s+", " ", out)
    return out, replaced


def replace_longest_common_substring_span(needle, repl, x):
    searchtokens = q.tokenize(needle)
    xtokens = q.tokenize(x)
    longest_start = None
    longest_end = None
    i = 0
    while i < len(xtokens):
        j = 1
        k = 0
        while j <= len(searchtokens):
            xtokenses = xtokens[i:i+j]
            searchtokenses = searchtokens[k:k+j]
            ed = edit_distance(" ".join(xtokenses), " ".join(searchtokenses))
            closeenough = ed < 2 or ed < (max(len(" ".join(xtokenses)), len(" ".join(searchtokenses))) / 5.)
            # closeenough = True
            # for l in range(j):
            #     if edit_distance(xtokenses[l], searchtokenses[l]) > 1:
            #         closeenough = False
            if closeenough:
                if longest_end is None or (longest_end - longest_start) < j:
                    longest_start, longest_end = i, i+j
            else:
                 break
            j += 1
        i += 1
    if longest_start is None:
        return replace_longest_common_substring_span_re(needle, repl, x)
    replaced = " ".join(xtokens[longest_start:longest_end])
    del xtokens[longest_start+1:longest_end]
    xtokens[longest_start] = repl
    out = " ".join(xtokens)
    return out, replaced


def edit_distance(a, b):
    if a == b:
        return 0
    else:
        return editdistance.eval(a, b)


if __name__ == "__main__":
    # print(replace_longest_common_substring_span_re("microsoft visual studio", "<E0>", "Name the company founded in US and created Visual Studio"))
    # get_vnts()
    q.argprun(load_all)