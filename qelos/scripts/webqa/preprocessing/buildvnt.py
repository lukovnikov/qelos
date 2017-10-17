from __future__ import print_function
import qelos as q
import re, time, pickle
from SPARQLWrapper import SPARQLWrapper, JSON
from SPARQLWrapper.SPARQLExceptions import EndPointNotFound, EndPointInternalError


#########################################################
# Get valid next tokens for a .lin query based on graph #
#########################################################



# rules:
# ARGMAX/ARGMIN only after relation, only in constraints
# JOIN only if there was BRANCH before and only after an entity or ARGMAX/ARGMIN
# RETURN only if no branch open and at least one relation
# in bottom-up, relations can only follow other relations or entities
# in top-down, entities can only follow relations

# token categories
MASK = -1
NONE = 0
ENT = 1
REL = 2
BRANCH = 3
JOIN = 4
ARGOPT = 5
RETURN = 6
VAR = 7
PLACEHOLDER = 8
TIMETOKEN = 9

# mode
BOTTOMUP = 1
TOPDOWN = 2


def category(token):
    if token[0] == ":":
        return REL
    elif token == "<ARGMAX>" or token == "<ARGMIN>":
        return ARGOPT
    elif token == "<BRANCH>":
        return BRANCH
    elif token == "<JOIN>":
        return JOIN
    elif token == "<RETURN>":
        return RETURN
    elif re.match("var\d+", token):
        return VAR
    elif re.match("<E\d+>", token):
        return PLACEHOLDER
    elif re.match("<TIME-.+>", token):
        return TIMETOKEN
    else:
        return ENT


def fbfy(x):
    if category(x) == REL:
        return "<http://rdf.freebase.com/ns/{}>".format(x[1:])
    elif category(x) == ENT:
        return "<http://rdf.freebase.com/ns/{}>".format(x)
    elif category(x) == VAR:
        return "?{}".format(x)
    else:
        raise Exception("can't fbfy this")


def unfbfy(x):
    retem = re.match("http://rdf\.freebase\.com/ns/(.+)", x)
    if retem:
        rete = retem.group(1)
        return rete
    else:
        return None


class Querier(object):
    limit = 50
    rel_blacklist = ["http://rdf\.freebase\.com/key/.+",
                     "http://www\.w3\.org/.+",
                     "http://rdf\.freebase\.com/ns/common\.topic\..+",
                     "http://rdf\.freebase\.com/ns/type\.object\..+", ]

    rel_unblacklist = ["http://rdf\.freebase\.com/ns/common\.topic\.notable_types",
                       "http://rdf\.freebase\.com/ns/common\.topic\.notable_for",
                       "http://rdf\.freebase\.com/ns/type\.object\.name"]

    def __init__(self, address="http://drogon:9890/sparql", **kw):
        #address = "http://localhost:9890/sparql"        # TODO remote testing
        super(Querier, self).__init__(**kw)
        self.sparql = SPARQLWrapper(address)
        self.sparql.setReturnFormat(JSON)

    def _exec_query(self, query):
        self.sparql.setQuery(query)
        retries = 10
        while True:
            try:
                res = self.sparql.query().convert()
                return res
            except (EndPointInternalError, EndPointNotFound) as e:
                print("retrying {}".format(retries))
                retries -= 1
                if retries < 0:
                    raise e
                time.sleep(1)

    def get_relations_of_mid(self, mid, only_reverse=False, incl_reverse=False):
        revrels = set()
        if not q.iscollection(mid):
            mid = (mid,)
        for mide in mid:  # check if every mid in input is valid
            if not re.match("[a-z]{1,2}\..+", mide):
                return set()
        if incl_reverse:
            revrels = self.get_relations_of_mid(mid, only_reverse=True, incl_reverse=False)
            revrels = {"reverse:{}".format(revrel) for revrel in revrels}
        if only_reverse:
            query = "SELECT DISTINCT(?p) WHERE {{ ?o ?p ?s VALUES ?s {{ {} }} }}" \
                .format(" ".join([fbfy(srce) for srce in mid]))
        else:
            query = "SELECT DISTINCT(?p) WHERE {{ ?s ?p ?o VALUES ?s {{ {} }} }}" \
                .format(" ".join([fbfy(srce) for srce in mid]))
        # print q
        ret = set()
        res = self._exec_query(query)
        results = res["results"]["bindings"]
        for result in results:
            rete = result["p"]["value"]
            toadd = True
            for rel_blacklister in self.rel_blacklist:
                if re.match(rel_blacklister, rete):
                    toadd = False
                    break
            for rel_unblacklister in self.rel_unblacklist:
                if re.match(rel_unblacklister, rete):
                    toadd = True
                    break
            if toadd:
                rete = ":"+unfbfy(rete)
                ret.add(rete)
        ret.update(revrels)
        return ret

    def _arg_opt_at_triples(self, expr_so_far, mode, basevar, critvar, crittype):
        mode = "DESC" if mode == "<ARGMAX>" else "ASC"
        subquery = """{{
                    SELECT {} WHERE
                    {{ {} }}
                    ORDER BY {}({}({})) LIMIT 1
                   }}\n""".format(fbfy(basevar), expr_so_far, mode, crittype, fbfy(critvar))
        return subquery

    def triples2expr(self, triples):
        """ freebaseify everything and join """
        expr = ""
        for triple in triples:
            if category(triple[0]) == ARGOPT:
                expr = self._arg_opt_at_triples(expr, *triple)
            else:
                s, p, o = triple
                expr += "{} {} {}.\n".format(fbfy(s), fbfy(p), fbfy(o))
        return expr

    def values2sparql(self, value_dict):
        if len(value_dict) == 0:        # no values
            return ""
        else:
            ret = ""
            for var, val in value_dict.items():
                ret += "VALUES {} {{ {} }}\n".format(fbfy(var), fbfy(val))
            return ret

    def const2orderby(self, const):
        if len(const) > 1:
            raise NotImplemented("no support for more than one argmax/argmin")
        if len(const) == 0:
            return ""
        else:
            const = const[0]
        ### TODO WARNING !!! - doesn't do argmax on intermediate, instead order final result by intermediate
        return "ORDER BY {}({}({})) LIMIT 1".format(
            const["optmode"], const["littype"], fbfy(const["criterium"]))

    def get_range_of_relation(self, rel):
        query = "SELECT DISTINCT ?o WHERE {{ {} <http://www.w3.org/2000/01/rdf-schema#range> ?o}}".format(fbfy(rel))
        ret = set()
        res = self._exec_query(query)
        results = res["results"]["bindings"]
        for result in results:
            rete = result["o"]["value"]
            toadd = True
            if toadd:
                rete = unfbfy(rete)
                if rete is not None:
                    ret.add(rete)
        if len(ret) == 0:
            return None
        else:
            return list(ret)[0]

    def rel_is_datetime(self, rel):
        rng = self.get_range_of_relation(rel)
        if rng == "type.datetime":
            return True
        return False

    def get_relations_of_var(self, spec, var, only_reverse=False, incl_reverse=False):
        entityquery = self.get_entity_query(spec, var, noargopt=True)
        # triples, values, const = spec
        # sparqlvalues = self.values2sparql(values)
        # expr = self.triples2expr(triples)
        # orderby = self.const2orderby(const)
        revrels = set()
        if incl_reverse:
            revrels = self.get_relations_of_var(spec, var, only_reverse=True, incl_reverse=False)
            revrels = {"reverse:{}".format(revrel) for revrel in revrels}
        if only_reverse:
            query = "SELECT DISTINCT(?p) WHERE {{ {{ {} }} ?o ?p {}. }}" \
                .format(entityquery, fbfy(var))
        else:
            query = "SELECT DISTINCT(?p) WHERE {{ {{ {} }} {} ?p ?o. }}" \
                .format(entityquery, fbfy(var))
        # print q
        ret = set()
        res = self._exec_query(query)
        results = res["results"]["bindings"]
        for result in results:
            rete = result["p"]["value"]
            toadd = True
            for rel_blacklister in self.rel_blacklist:
                if re.match(rel_blacklister, rete):
                    toadd = False
                    break
            for rel_unblacklister in self.rel_unblacklist:
                if re.match(rel_unblacklister, rete):
                    toadd = True
                    break
            if toadd:
                rete = ":"+unfbfy(rete)
                if rete is not None:
                    ret.add(rete)
        ret.update(revrels)
        return ret

    def get_entities_of_var(self, spec, var):
        query = self.get_entity_query(spec, var, noargopt=True)
        # print q
        ret = set()
        res = self._exec_query(query)
        results = res["results"]["bindings"]
        for result in results:
            rete = result[var]["value"]
            toadd = True
            if toadd:
                rete = unfbfy(rete)
                if rete is not None:
                    ret.add(rete)
        return ret

    def get_entity_query(self, spec, var, noargopt=False):
        triples, values, const = spec
        expr = self.triples2expr(triples)
        sparqlvalues = self.values2sparql(values)
        orderby = self.const2orderby(const)
        orderby = "" if noargopt else orderby
        query = "SELECT DISTINCT({}) WHERE {{ {} {}}} {}" \
            .format(fbfy(var), expr,
                    sparqlvalues,
                    orderby)
        return query

    def get_entity_query_fn(self, spec, var):
        query = self.get_entity_query(spec, var)
        def inner():
            # print q
            ret = set()
            res = self._exec_query(query)
            results = res["results"]["bindings"]
            for result in results:
                rete = result[var]["value"]
                toadd = True
                if toadd:
                    rete = unfbfy(rete)
                    if rete is not None:
                        ret.add(rete)
            return ret
        return inner, query


class Traverser(object):
    """
    Keeps state of traversal (query), consumes one token at a time, outputs next valid tokens.
    Incrementally builds up query.
    Final query is answer query.
    """

    def __init__(self, address="http://drogon:9890/sparql", repl_dic=None, **kw):
        address = "http://localhost:9890/sparql"        # TODO remote testing
        super(Traverser, self).__init__(**kw)
        self.que = Querier(address=address)

        self.repl_dic = repl_dic

        # main incremental state
        self.topic_mid = None
        self.current_var = None     # where relations are added
        self.last_new_var = None
        self.var_count = 0
        self.current_butd = []
        self.current_value_query = ([], {}, [])

        # some state flags
        self.current_type = NONE
        self.prev_type = NONE   # or ENT REL ARGOPT BRANCH JOIN RETURN
        self.mode = BOTTOMUP        # or BOTTOMUP TOPDOWN
        self.branchdepth = 0

        # cache
        self.prevrelstack = [None]          # stacked in branching levels
        self.varstack = [None]

        # final result
        self.query = None
        self.query_fn = None

    def newvar(self):
        ret = "var{}".format(self.var_count)
        self.var_count += 1
        self.last_new_var = ret
        return ret

    def next_token(self, token):
        """ Consume token, update current query state, return valid next tokens"""
        if re.match(r'<E\d>', token):
            token = self.repl_dic[token]
        self.current_butd.append(token)

        vnt = set()

        if self.topic_mid is None:
            # must be starting entity of bottomup
            assert(category(token) == ENT)
            relations = self.que.get_relations_of_mid(token)
            self.topic_mid = token
            self.mode = BOTTOMUP
            self.current_type = ENT
            self.varstack[-1] = token
            self.prevrelstack[-1] = set(relations)
            vnt |= set(relations)
        else:
            self.current_type = category(token)
            if category(token) == ENT:
                if self.mode == BOTTOMUP:
                    raise Exception("impossible to get entity in BOTTOMUP")
                elif self.mode == TOPDOWN:  # set top var on varstack to value
                    self.current_value_query[1][self.varstack[-1]] = token
            elif category(token) == REL:
                # if a relation, attach to current var, make new var
                self.current_value_query[0].append((self.varstack[-1], token, self.newvar()))
                self.varstack[-1] = self.last_new_var
                relations = self.que.get_relations_of_var(self.current_value_query, self.varstack[-1])
                self.prevrelstack[-1] = set(relations)
                vnt |= set(relations)
                if self.mode == TOPDOWN:
                    if self.prev_type == REL:
                        raise q.SumTingWongException("can't have more than one rel in branch currently")
                    if self.que.rel_is_datetime(token):
                        for k, v in self.repl_dic.items():      # add time tokens, if any
                            if re.match("<TIME-.+>", k):
                                vnt |= {k}
                        vnt |= {"<TIME-NOW>"}
                    else:
                        entities = self.que.get_entities_of_var(self.current_value_query, self.varstack[-1])
                        vnt |= set(entities)
                        vnt |= {"<E0>"}
            elif category(token) == TIMETOKEN:
                # remove last added branch from current value query
                # TODO: BEWARE WHEN MULTIPLE HOPS INSIDE BRANCH WON'T WORK
                del self.current_value_query[0][-1]
            elif category(token) == BRANCH:
                self.branchdepth += 1
                self.mode = TOPDOWN
                self.prevrelstack.append(self.prevrelstack[-1])
                self.varstack.append(self.varstack[-1])
                vnt |= self.prevrelstack[-1]
            elif category(token) == JOIN:       # must requery !
                self.branchdepth -= 1
                if self.branchdepth == 0:
                    self.mode = BOTTOMUP
                del self.prevrelstack[-1]
                del self.varstack[-1]
                if self.prev_type == ARGOPT:
                    self.current_value_query[2][-1]["select"] = self.varstack[-1]       # complete argopt
                    # self.current_value_query[0][-1][1] = self.varstack[-1]
                relations = self.que.get_relations_of_var(self.current_value_query, self.varstack[-1])
                self.prevrelstack[-1] = set(relations)
                vnt |= set(relations)
            elif category(token) == ARGOPT:
                rng = self.que.get_range_of_relation(self.current_butd[-2])     # TODO hacky
                if rng == "type.datetime":
                    literaltype = "xsd:datetime"
                else:
                    literaltype = "xsd:float"
                self.current_value_query[2].append({"optmode": "DESC" if token == "<ARGMAX>" else "ASC",
                                                    "select": None,     # must be completed by JOIN
                                                    "littype": literaltype,
                                                    "criterium": self.varstack[-1]})        # TODO: incorrect sorting if type wrong --> must specify type
                # self.current_value_query[0].append([token, None, self.varstack[-1], literaltype])
            elif category(token) == RETURN:
                self.query_fn, self.query = self.que.get_entity_query_fn(self.current_value_query, self.varstack[-1])
        if self.current_type == REL:
            vnt |= {"<BRANCH>"}
            if self.mode == TOPDOWN:
                vnt |= {"<ARGMAX>", "<ARGMIN>"}
            else:
                vnt |= {"<RETURN>"}
        elif self.current_type == ENT:
            if self.mode == BOTTOMUP:
                pass
            elif self.mode == TOPDOWN:      # finalize constraint!
                vnt |= {"<JOIN>"}
        elif self.current_type == TIMETOKEN:
            if self.mode == TOPDOWN:
                vnt |= {"<JOIN>"}
            else:
                pass
        elif self.current_type == JOIN:
            if self.branchdepth == 0:
                vnt |= {"<RETURN>"}
            vnt |= {"<BRANCH>"}
        elif self.current_type == ARGOPT:
            vnt |= {"<JOIN>"}
        self.prev_type = self.current_type
        self.current_type = NONE
        return vnt


def get_vnt_for_butd(butd="WebQTest-523	what is the largest nation in <E0>	<E0> :base.locations.continents.countries_within <BRANCH> :topic_server.population_number <ARGMAX> <JOIN> <RETURN>	(<E0>|europe|m.02j9z)"):
    qid, question, query, replacements = butd.split("\t")
    replacements = replacements.split(",")
    nl_repl_dic = {}
    fl_repl_dic = {}
    for replacement in replacements:
        replacement = re.match("\(([^\]]+)\)", replacement.strip()).group(1)
        for replel in replacement.split(";"):
            tkn, nlrep, flrep = replel.split("|")
            nl_repl_dic[tkn] = nlrep
            fl_repl_dic[tkn] = flrep
    # print(query, fl_repl_dic)
    traverser = Traverser(repl_dic=fl_repl_dic)
    vnts = [{"<E0>"}]
    for token in query.split():
        if not token in vnts[-1]:
            # EXCEPTIONS DUE TO MISMATCH DATA/GOLD
            solved = False
            if qid in "WebQTest-238 WebQTest-863 WebQTest-974 WebQTest-305".split():
                vnts[-1].add(token)
                solved = True
            if re.match(r'WebQTrn-\d+', qid):
                vnts[-1].add(token)
                solved = True
            if re.match(r'WebQTest-\d+', qid):
                vnts[-1].add(token)
                solved = True
            if solved:
                print("{}: added token {} to vnt manually".format(qid, token))
            else:
                raise Exception("{}: token {} not in vnt".format(qid, token))
        vnt = traverser.next_token(token)
        vnts.append(vnt)
    for vnt, token in zip(vnts, query.split()):
        # print(token, len(vnt), vnt)
        pass


    res = traverser.query_fn()
    # print(traverser.query)
    # print(res)
    return qid, vnts, traverser.query_fn


class RelDicAcc(object):
    def __init__(self):
        self.reldic = {"<NONE>": 0}
        self.relcounts = {"<NONE>": 0}

    def get_id_for_rel(self, rel):
        if rel not in self.reldic:
            newid = len(self.reldic)
            self.reldic[rel] = newid
            self.relcounts[newid] = 0
        id = self.reldic[rel]
        self.relcounts[id] += 1
        return id


def get_propchains_for_ent(ent, upto=1, reldic=None):
    que = Querier()
    chains = set()
    prevchains = set()
    newprevchains = set()
    i = 1
    singlerelchains = que.get_relations_of_mid(ent, incl_reverse=False)
    for singlerelchain in singlerelchains:
        if reldic is not None:
            tochains = reldic.get_id_for_rel(singlerelchain)
        else:
            tochains = singlerelchain
        chains.add((tochains,))
        prevchains.add((singlerelchain,))
    i += 1
    if upto >= 2:
        for prevchain in prevchains:
            m = re.match(":(-?)(.+)", prevchain[0])
            if m.group(1) == "-":
                triples = [("var0", m.group(2), ent)]
            else:
                triples = [(ent, m.group(2), "var0")]
            newrels = que.get_relations_of_var((triples, {}, {}), "var0", incl_reverse=False)
            for newrel in newrels:
                if reldic is not None:
                    toadd = (reldic.get_id_for_rel(prevchain[0]), reldic.get_id_for_rel(newrel))
                else:
                    toadd = (prevchain[0], newrel)
                chains.add(toadd)
        # tworelchains = que.get_twohop_chains_from_entity(ent, incl_reverse=True)
        # for tworelchain in tworelchains:
        #     chains.add(tworelchain)
    if upto > 2:
        raise q.SumTingWongException("chains of more than 2 hops not supported")
    return chains


def get_prop_chains_for_dataset(p="../../../../datasets/webqsp/webqsp.",
                         files=("test.json", "train.json"),
                         outfile="webqsp.all.chains",
                         upto=2):
    tt = q.ticktock("propchains builder")
    reldic = RelDicAcc()
    import json
    parsesdata = {}
    for file in files:
        tt.tick("doing {}".format(file))
        filep = p + file
        chains = {}
        with open(filep) as f:
            qf = json.load(f)
            i = 0
            for question in qf["Questions"]:
                qid = question["QuestionId"]
                questiontext = question["ProcessedQuestion"]
                for parse in question["Parses"]:
                    qpid = parse["ParseId"]
                    topicent = parse["TopicEntityMid"]
                    topicentname = parse["TopicEntityName"]
                    topicentmention = parse["PotentialTopicEntityMention"]
                    corechain = parse["InferentialChain"]
                    if topicent is None or corechain is None:
                        print("lacking something: {}".format(qpid))
                        continue
                    qpidchains = get_propchains_for_ent(topicent, upto=upto, reldic=reldic)
                    corechain = tuple([reldic.get_id_for_rel(":"+corerel) for corerel in corechain])
                    if corechain in qpidchains:
                        negchains = qpidchains - {corechain}
                    else:
                        print("{} core chain not in qpidchains".format(qpid))
                        negchains = qpidchains
                    tx = qpid[:8] == u"WebQTest"
                    parsesdata[qpid] = {"qid": qid,
                                        "question": questiontext,
                                        "topicmid": topicent,
                                        "topicname": topicentname,
                                        "topicmention": topicentmention,
                                        "corechain": corechain,
                                        "negchains": negchains,
                                        "tx": tx}
                    i += 1
                    if i % 1 == 0:
                        tt.msg("{} {}, len: {}".format(i, qpid, str(len(qpidchains))))
                # break

    outp = p + outfile
    outdata = {"parses": parsesdata, "reldic": reldic}
    pickle.dump(outdata, open(outp, "w"))
    tt.tock("done {}".format(outfile))
    reloaded = pickle.load(open(outp))
    q.embed()


def get_vnt_for_datasets(p="../../../../datasets/webqsp/webqsp.",
                         files=("test.time", "train.time")):
    tt = q.ticktock("vnt builder")
    for file in files:
        tt.tick("doing {}".format(file))
        filep = p+file+".butd"
        vnts = {}
        with open(filep) as f:
            i = 0
            for line in f:
                qid, vnt, _ = get_vnt_for_butd(line)
                vnts[qid] = vnt
                i += 1
                if i % 10 == 0:
                    tt.msg("{}".format(i))
        pickle.dump(vnts, open(p+file+".butd.vnt", "w"))
        tt.tock("done {}".format(file))


if __name__ == "__main__":
    q.argprun(get_prop_chains_for_dataset)
    # _, _, query = get_vnt_for_butd("WebQTest-305	where does <E0> live	<E0> :people.person.places_lived <BRANCH> :people.place_lived.start_date <TIME-NOW> <JOIN> <BRANCH> :people.place_lived.end_date <TIME-NOW> <JOIN> :people.place_lived.location <RETURN>	(<E0>|bradley walsh|m.05h48b)")
    # print(query())
    # q.argprun(get_vnt_for_datasets)