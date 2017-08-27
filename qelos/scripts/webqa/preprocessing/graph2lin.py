from __future__ import print_function
import qelos as q
import re


def run(prefix="../../../../datasets/webqsp/webqsp.", files=("train", "test")):
    for file in files:
        p = prefix+file+".graph"
        relins = load_graph_dataset(p)
        with open(prefix+file+".butd", "w") as f:
            for qid, question, answer, dics, info in relins:
                f.write("{}\t{}\t{}\t({}|{}|{})\n".format(qid, question, answer, "<E0>", dics[0]["<E0>"], dics[1]["<E0>"]))


def load_graph_dataset(inp="../../../../datasets/webqsp/webqsp.train.graph"):
    """
    Loads graph-based dataset, relinearizes to bottom-up core, top-down constraints tree
    """
    ret = []
    c = 0
    not_loaded = []
    with open(inp) as f:
        maxlen = 0
        for line in f:
            loaded = load_graph_question(line)
            if loaded is not None:
                qid, question, answer, (ent_nl_dic, ent_fl_dic), info = loaded
                answer = relinearize(answer)

                print(qid, question)
                print(qid, answer)

                maxlen = max(maxlen, len(answer.split()))
                ret.append((qid, question, answer, (ent_nl_dic, ent_fl_dic), info))
            else:
                # print(c, qid, "not loaded")
                not_loaded.append(qid)
            c += 1
        print("{} max len".format(maxlen))
        print("{} loaded, {} not loaded out of {}".format(len(ret), len(not_loaded), c))
    print("done")
    return ret


def relinearize(graph):
    triples = [tuple(x.strip().split()) for x in graph.strip().split(";") if len(x) > 0]
    corechain, othertriples = _get_core_chain(triples, "OUT")
    constraintpoints = filter(lambda x: re.match("var\d", x) or x == "OUT", corechain)
    constraints = {}
    ret = " ".join(corechain)
    for constraintpoint in constraintpoints:
        constraint, othertriples = _get_constraint(othertriples, constraintpoint)
        constraints[constraintpoint] = " ".join(["<BRANCH> {} <JOIN>".format(x) for x in constraint])
    for constraintpoint in constraintpoints:
        ret = ret.replace(constraintpoint, constraints[constraintpoint])
    ret += " <RETURN>"
    ret = re.sub(r'\s+', ' ', ret)
    ret = ret.replace("ARGMAX", "<ARGMAX>")
    ret = ret.replace("ARGMIN", "<ARGMIN>")
    ret = ret.replace("COUNT", "<COUNT>")
    return ret


def _get_constraint(triples, root):
    if not (re.match("var\d", root) or root == "OUT"):
        return [root], triples

    roottriples = []
    othertriples = []
    for s, p, o in triples:
        if s == root:
            roottriples.append((s, p, o))
        else:
            othertriples.append((s, p, o))
    branches = []
    for s, p, o in roottriples:
        branch, othertriples = _get_constraint(othertriples, o)
        if len(branch) == 1:
            app = "{} {}".format(p, branch[0])
            branches.append(app)
        else:   # join returned branches
            branchelems = []
            for branchelem in branch[:-1]:
                branchelem = "<BRANCH> {} <JOIN>".format(branchelem)
                branchelems.append(branchelem)
            branchelems.append(branch[-1])
            app = "{} {}".format(p, " ".join(branchelems))
            branches.append(app)
    return branches, othertriples


def _get_core_chain(triples, root):
    if not (re.match("var\d", root) or root == "OUT"):
        return [root], triples

    roottriples = []
    othertriples = []
    for s, p, o in triples:
        if s == root:
            othertriples.append((s, p, o))
        elif o == root:
            roottriples.append((s, p, o))
        else:
            othertriples.append((s, p, o))
    assert(len(roottriples) == 1)
    prechain, othertriples = _get_core_chain(othertriples, roottriples[0][0])
    return prechain + [roottriples[0][1]] + [root], othertriples


def _relinearize_rec(triples, root):
    if not (re.match("var\d", root) or root == "OUT"):      # root is not a variable (intermediate or output)
        return [root]       # because it's a leaf

    # separate triples in triples resulting in root and other triples
    roottriples = []
    othertriples = []
    constrainttriples = []
    for s, p, o in triples:
        if s == root:
            # constraint on root
            constrainttriples.append((s, ":"+p, o))
        elif o == root:
            roottriples.append((s, ":"+p, o))
        else:
            othertriples.append((s, p, o))

    # get sublinearizations
    sublinearisations = []
    constraints = []
    for s, p, o in constrainttriples:
        constraint_sublinearisation = _relinearize_rec(othertriples, o)
        for csl in constraint_sublinearisation:
            constraints.append("{} {}".format(p, csl))
    constraints = " ".join(["<BRANCH> {}".format(x) for x in constraints])
    for s, p, o in roottriples:
        core_sublinearisation = _relinearize_rec(othertriples, s)        # can only be one
        core_sublinearisation = core_sublinearisation[0]
        core_sublinearisation = "{} {}".format(core_sublinearisation, p)
    return


def load_graph_question(line):
    """ Takes line for .lin file (triple-based query graph with extra info)
        and relinearizes to bottom-up core, top-down constraints tree.
        Replaces topic entity with token."""
    splits = line.split("\t")
    if len(splits) > 3:
        qid, question, unlinkedents, numrels, numvars, valconstraints, query = splits
        unlinkedents, numrels, numvars, valconstraints = map(int, (unlinkedents, numrels, numvars, valconstraints))

        # replace topic entity by placeholder both in question and in query
        # get topic entity id and mention from query
        topicentity = re.findall(r"([a-z]+\.[^\s\[]+)\[([^\]]+)\]\*", query)[0]
        topic_freebase_id = topicentity[0]
        topic_surface_form = topicentity[1]

        # replace mention in question and query by token
        question = question.replace(topic_surface_form, "<E0>")
        query = query.replace(topic_freebase_id, "<E0>")
        query = re.sub(r'\[[^\]]+\]\*?', "", query)     # clean the surface form info from query

        # return info
        entity_token_to_nl_dic = {"<E0>": topic_surface_form}
        entity_token_to_fl_dic = {"<E0>": topic_freebase_id}

        return qid, question, query, (entity_token_to_nl_dic, entity_token_to_fl_dic), \
               {"qid": qid, "unlinkedents": unlinkedents,
                "numrels": numrels, "numvars": numvars,
                "valconstraints": valconstraints}
    else:
        return None


if __name__ == "__main__":
    _, _, x, _, _ = load_graph_question(
        """WebQTest-1779	x	0	3	1	0	var1 r.r.core2 OUT ;
        var1 r.r.consA1 var2 ;
        var2 r.r.consA2 m.060d2[president] ;
        var2 r.r.consA3 m.0123[bla] ;
        OUT r.r.consB1 var3 ;
        var3 r.r.consB2 m.azer[ty] ;
        m.07hyk[president theodore roosevelt]* r.r.core1 var1 ;
        """)
    #relinearize(x)
    q.argprun(run)