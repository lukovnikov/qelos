# -*- coding: utf8 -*-

from __future__ import print_function
import qelos as q
import re, time, pickle, sys, codecs
from SPARQLWrapper import SPARQLWrapper, JSON
from SPARQLWrapper.SPARQLExceptions import EndPointNotFound, EndPointInternalError




#########################################################
# Get valid next tokens for a .lin query based on graph #
#########################################################
ENT = 0
REL = 1
TYP = 2
BRANCH = 3
JOIN = 4
RETURN = 5
COUNT = 6
EQUALS = 7
OFTYPE = 8

BOTTOMUP = 0
TOPDOWN = 1


def category(token):
    if token[0] == ":":
        return REL
    elif re.match("<http://dbpedia\.org/resource/.+>", token):
        return ENT
    elif re.match("<http://dbpedia\.org/ontology/.+>", token):
        return TYP
    elif token == "<<BRANCH>>":
        return BRANCH
    elif token == "<<JOIN>>":
        return JOIN
    elif token == "<<COUNT>>":
        return COUNT
    elif token == "<<EQUALS>>":
        return EQUALS
    elif token == "<<RETURN>>":
        return RETURN
    elif token == "<<TYPE>>":
        return OFTYPE
    else:
        raise q.SumTingWongException("unknown category for token: {}".format(token))


# rules:
# ARGMAX/ARGMIN only after relation, only in constraints
# JOIN only if there was BRANCH before and only after an entity or ARGMAX/ARGMIN
# RETURN only if no branch open and at least one relation
# in bottom-up, relations can only follow other relations or entities
# in top-down, entities can only follow relations


def dbfy(x):
    return u"<{}>".format(x)


def undbfy(x):
    if x[:2] == ":-":
        x = x[2:]
    elif x[:1] == ":":
        x = x[1:]
    return x


class Querier(object):
    limit = 100

    rel_blacklist = [
    ]

    rel_unblacklist = []

    rel_filter = [
        "http://dbpedia\.org/.+",
        "http://www\.w3\.org/1999/02/22-rdf-syntax-ns#type",
    ]

    ent_filter = [
        "http://dbpedia\.org/resource/.+",
        "http://dbpedia\.org/ontology/.+",
    ]

    ent_blacklist = [
        "http://dbpedia\.org/class/yago/.+",
        "http://dbpedia\.org/ontology/Wikidata.+",
    ]

    def __init__(self, endpoint="http://localhost:7890/sparql",
                 relwhitelistp=None, **kw):
        super(Querier, self).__init__(**kw)
        self.endpoint = endpoint
        self.sparql = SPARQLWrapper(endpoint)
        self.sparql.setReturnFormat(JSON)

        self.relwhitelist = None
        if relwhitelistp is not None:
            self.relwhitelist = set()
            with open(relwhitelistp) as f:
                for line in f:
                    self.relwhitelist.add(line.split(",")[0])

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

    def get_types_of_id(self, id):
        if not q.issequence(id):
            id = [id]
        query = u"SELECT DISTINCT ?s ?t WHERE {{ ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?t VALUES ?s {{ {} }} }}"\
            .format(" ".join(id))
        ret = {}
        res = self._exec_query(query)
        results = res["results"]["bindings"]
        for result in results:
            s = result["s"]["value"]
            t = result["t"]["value"]
            toadd = True
            for ent_filterer in self.ent_filter:
                toadd = False
                if re.match(ent_filterer, t):
                    toadd = True
                    break
            for ent_blacklister in self.ent_blacklist:
                if re.match(ent_blacklister, t):
                    toadd = False
                    break
            if toadd:
                s = u"<{}>".format(s)
                t = u"<{}>".format(t)
                if s not in ret:
                    ret[s] = set()
                ret[s].add(t)
        return ret

    def get_entity_property(self, entities, property, language=None):
        if not q.issequence(entities):
            entities = [entities]
        # entities = [fbfy(entity) for entity in entities]
        # propertychain = [fbfy(p) for p in property.strip().split()]
        propchain = ""
        prevvar = "?s"
        varcount = 0
        for prop in propertychain:
            newvar = "?var{}".format(varcount)
            varcount += 1
            propchain += "{} {} {} .\n".format(prevvar, prop, newvar)
            prevvar = newvar
        propchain = propchain.replace(prevvar, "?o")

        query = """SELECT ?s ?o WHERE {{
                        {}
                        VALUES ?s {{ {} }}
                        {}
                    }}""".format(
            propchain,
            " ".join(entities),
            "FILTER (lang(?o) = '{}')".format(language) if language is not None else "")
        ret = {}
        res = self._exec_query(query)
        results = res["results"]["bindings"]
        for result in results:
            s = unfbfy(result["s"]["value"])
            if s not in ret:
                ret[s] = set()
            val = result["o"]["value"]
            if language is None:
                val = unfbfy(val)
            ret[s].add(val)
        return ret

    def get_relations_of_id(self, id, only_reverse=False, incl_reverse=True):
        revrels = set()
        if not q.iscollection(id):
            id = (id,)
        if incl_reverse:
            revrels = self.get_relations_of_id(id, only_reverse=True, incl_reverse=False)
            revrels = {u":-{}".format(revrel[1:]) for revrel in revrels}
        if only_reverse:
            query = u"SELECT DISTINCT(?p) WHERE {{ ?o ?p ?s VALUES ?s {{ {} }} }}" \
                .format(u" ".join(id))
        else:
            query = u"SELECT DISTINCT(?p) WHERE {{ ?s ?p ?o VALUES ?s {{ {} }} }}" \
                .format(u" ".join(id))
        # print q
        ret = set()
        res = self._exec_query(query)
        results = res["results"]["bindings"]
        for result in results:
            rete = result["p"]["value"]
            toadd = True
            if self.relwhitelist is not None:
                toadd = rete in self.relwhitelist
            else:
                for rel_filterer in self.rel_filter:
                    toadd = False
                    if re.match(rel_filterer, rete):
                        toadd = True
                        break
                for rel_blacklister in self.rel_blacklist:
                    if re.match(rel_blacklister, rete):
                        toadd = False
                        break
                for rel_unblacklister in self.rel_unblacklist:
                    if re.match(rel_unblacklister, rete):
                        toadd = True
                        break
            if toadd:
                rete = u":{}".format(dbfy(rete))
                ret.add(rete)
        ret.update(revrels)
        return ret

    def get_entity_query(self, triples, replacements, outvar, limit=None):
        expr = self.triples2expr(triples)
        sparqlvalues = self.values2sparql(replacements)
        if limit is not None:
            limitopt = u"LIMIT {}".format(limit)
        else:
            limitopt = u""
        query = u"SELECT DISTINCT({}) WHERE {{\n {} \n {} \n}} {}" \
            .format(outvar, expr, sparqlvalues, limitopt)
        return query

    def get_entity_query_fn(self, triples, replacements, outvar):
        query = self.get_entity_query(triples, replacements, outvar)
        def inner():
            # print q
            ret = set()
            res = self._exec_query(query)
            results = res["results"]["bindings"]
            for result in results:
                rete = result[var]["value"]
                toadd = True
                if toadd:
                    if rete is not None:
                        ret.add(rete)
            return ret
        return inner, query

    def triples2expr(self, triples):
        expr = u""
        for triple in triples:
            s, p, o = triple
            if p[:2] == ":-":
                o, p, s = s, u":"+p[2:], o
            expr += u"{} {} {} .\n".format(s, undbfy(p), o)
        return expr

    def values2sparql(self, value_dict):
        if len(value_dict) == 0:        # no values
            return u""
        else:
            ret = u""
            for var, val in value_dict.items():
                ret += u"VALUES {} {{ {} }}\n".format(var, val)
            return ret

    def get_twohop_chains_from_entity(self, entity, incl_reverse=True):
        queries = []
        queries.append((u"SELECT DISTINCT ?p1 ?p2 WHERE {{\n {} ?p1 ?var1 .\n ?var1 ?p2 ?var2 }} LIMIT 100000".format(entity), True, True))
        if incl_reverse:
            queries.append((u"SELECT DISTINCT ?p1 ?p2 WHERE {{\n ?var1 ?p1 {} .\n ?var1 ?p2 ?var2 }} LIMIT 100000".format(entity), False, True))
            queries.append((u"SELECT DISTINCT ?p1 ?p2 WHERE {{\n ?var1 ?p1 {} .\n ?var2 ?p2 ?var1 }} LIMIT 100000".format(entity), False, False))
            queries.append((u"SELECT DISTINCT ?p1 ?p2 WHERE {{\n {} ?p1 ?var1 .\n ?var2 ?p2 ?var1 }} LIMIT 100000".format(entity), True, False))
        ret = set()
        for query, p1rev, p2rev in queries:
            res = self._exec_query(query)
            results = res["results"]["bindings"]
            for result in results:
                rete_p1 = result["p1"]["value"]
                rete_p2 = result["p2"]["value"]
                toadd = True
                if self.relwhitelist is not None:
                    toadd = rete_p1 in self.relwhitelist
                    toadd = toadd and rete_p2 in self.relwhitelist
                else:
                    for rel_filterer in self.rel_filter:
                        toadd = False
                        if re.match(rel_filterer, rete_p1) and re.match(rel_filterer, rete_p2):
                            toadd = True
                            break
                    for rel_blacklister in self.rel_blacklist:
                        if re.match(rel_blacklister, rete_p1) or re.match(rel_blacklister, rete_p2):
                            toadd = False
                            break
                    for rel_unblacklister in self.rel_unblacklist:
                        if re.match(rel_unblacklister, rete_p1) or re.match(rel_unblacklister, rete_p2):
                            # TODO THIS IS WRONG !!!
                            toadd = True
                            break
                if toadd:
                    rete_p1 = u":{}{}".format(u"-" if p1rev == False else u"",
                                           dbfy(rete_p1))
                    rete_p2 = u":{}{}".format(u"-" if p2rev == False else u"",
                                           dbfy(rete_p2))
                    ret.add((rete_p1, rete_p2))
        return ret


    def get_relations_of_value(self, triples, replacements, outvar, only_reverse=False, incl_reverse=True, innerlimit=None):
        entityquery = self.get_entity_query(triples, replacements, outvar, limit=innerlimit)

        revrels = set()
        if incl_reverse:
            revrels = self.get_relations_of_value(triples, replacements, outvar, only_reverse=True, incl_reverse=False)
            revrels = {u":-{}".format(revrel[1:]) for revrel in revrels}
        if only_reverse:
            query = u"SELECT DISTINCT(?p) WHERE {{\n {{ {} }} \n ?o ?p {}. \n}}" \
                .format(entityquery, outvar)
        else:
            query = u"SELECT DISTINCT(?p) WHERE {{\n {{ {} }} \n {} ?p ?o. \n}}" \
                .format(entityquery, outvar)
        # print q
        ret = set()
        res = self._exec_query(query)
        results = res["results"]["bindings"]
        for result in results:
            rete = result["p"]["value"]
            toadd = True
            if self.relwhitelist is not None:
                toadd = rete in self.relwhitelist
            else:
                for rel_filterer in self.rel_filter:
                    toadd = False
                    if re.match(rel_filterer, rete):
                        toadd = True
                        break
                for rel_blacklister in self.rel_blacklist:
                    if re.match(rel_blacklister, rete):
                        toadd = False
                        break
                for rel_unblacklister in self.rel_unblacklist:
                    if re.match(rel_unblacklister, rete):
                        toadd = True
                        break
            if toadd:
                rete = u":{}".format(dbfy(rete))
                if rete is not None:
                    ret.add(rete)
        ret.update(revrels)
        return ret

    def get_entities_of_value(self, triples, replacements, outvar, limit=10):
        query = self.get_entity_query(triples, replacements, outvar, limit=limit)
        # print q
        ret = set()
        res = self._exec_query(query)
        results = res["results"]["bindings"]
        for result in results:
            rete = result[outvar[1:]]["value"]
            toadd = True
            for ent_filterer in self.ent_filter:
                toadd = False
                if re.match(ent_filterer, rete):
                    toadd = True
                    break
            for ent_blacklister in self.ent_blacklist:
                if re.match(ent_blacklister, rete):
                    toadd = False
                    break
            if toadd:
                rete = u"<{}>".format(rete)
                if rete is not None:
                    ret.add(rete)
        return ret


class Traversal(object):
    def __init__(self, endpoint="http://localhost:7890/sparql",
                 entities_linked=None,
                 relwhitelistp="../../../../datasets/lcquad/relations.txt"):

        super(Traversal, self).__init__()
        self.que = Querier(endpoint=endpoint, relwhitelistp=relwhitelistp)

        self.current_butd = []
        self.current_triples = []
        self.current_triples_varvals = {}

        self.branchdepth = 0
        self.var_count = 0
        self.in_ask_condition = False
        self.entities_so_far = set()

        self.relsforbranchlevel = [None]
        self.lastvarforbranchlevel = [None]

        self.entities_linked = entities_linked

        self.query_fn = None
        self.query = None

        self.final_vnts = []

    @property
    def mode(self):
        if self.branchdepth == 0:
            return BOTTOMUP
        else:
            return TOPDOWN

    def newvar(self):
        ret = u"?var{}".format(self.var_count)
        self.var_count += 1
        self.last_new_var = ret
        return ret

    def next_token(self, token):
        vnt = set()
        intopic = False

        if len(self.current_butd) == 0:      # must be starting entity
            assert(category(token) == ENT)
            current_type = ENT
            self.final_vnts.append({token})
            intopic = True
            self.entities_so_far.add(token)
            relations = self.que.get_relations_of_id(token)
            self.relsforbranchlevel[-1] = set(relations)
            self.lastvarforbranchlevel[-1] = token
            vnt |= set(relations)
        else:
            current_type = category(token)
            if category(token) == ENT or category(token) == TYP:
                self.entities_so_far.add(token)
                if self.mode == BOTTOMUP:
                    assert(self.in_ask_condition)
                    if self.entities_linked is not None:
                        self.final_vnts[-1] |= set(self.entities_linked)
                    else:
                        self.final_vnts[-1] |= self.entities_so_far      # assumes is properly linked
                    # raise q.SumTingWongException("can't get entity in BOTTOMUP")
                elif self.mode == TOPDOWN:
                    self.final_vnts[-1] |= {token}
                    self.current_triples_varvals[self.lastvarforbranchlevel[-1]] = token
            elif category(token) == REL:
                self.current_triples.append((self.lastvarforbranchlevel[-1], token, self.newvar()))
                self.lastvarforbranchlevel[-1] = self.last_new_var
                relations = self.get_relations_of_value()
                self.relsforbranchlevel[-1] = relations
                vnt |= relations
                if self.mode == TOPDOWN:
                    entities = self.get_entities_of_value()
                    vnt |= entities
            elif category(token) == BRANCH:
                self.branchdepth += 1
                self.relsforbranchlevel.append(self.relsforbranchlevel[-1])
                self.lastvarforbranchlevel.append(self.lastvarforbranchlevel[-1])
                vnt |= self.relsforbranchlevel[-1]
            elif category(token) == JOIN:
                self.branchdepth -= 1
                del self.relsforbranchlevel[-1]
                del self.lastvarforbranchlevel[-1]
                relations = self.get_relations_of_value()
                self.relsforbranchlevel[-1] = relations
                vnt |= relations
            elif category(token) == RETURN:
                self.query_fn, self.query = self.get_entity_query_fn()
            elif category(token) == COUNT:
                pass
        if current_type == REL:
            vnt |= {"<<BRANCH>>"}
            if self.mode == BOTTOMUP:
                vnt |= {"<<RETURN>>", "<<COUNT>>", "<<EQUALS>>"}
        elif current_type == ENT or current_type == TYP:
            if self.mode == TOPDOWN:
                vnt |= {"<<JOIN>>"}
            elif self.mode == BOTTOMUP and not intopic:
                assert(self.in_ask_condition)
                vnt |= {"<<RETURN>>"}
                self.in_ask_condition = False
        elif current_type == JOIN:
            if self.mode == BOTTOMUP:
                vnt |= {"<<RETURN>>", "<<COUNT>>", "<<EQUALS>>"}
            vnt |= {"<<BRANCH>>"}
        elif current_type == BRANCH:
            pass
        elif current_type == EQUALS:
            self.in_ask_condition = True
        elif current_type == COUNT:
            vnt |= {"<<RETURN>>"}
        self.prev_type = current_type
        self.current_butd.append(token)
        self.final_vnts.append(vnt)
        return vnt

    def get_relations_of_value(self):
        relations = self.que.get_relations_of_value(self.current_triples, self.current_triples_varvals, self.lastvarforbranchlevel[-1])
        return set(relations)

    def get_entities_of_value(self):
        entities = self.que.get_entities_of_value(self.current_triples, self.current_triples_varvals, self.lastvarforbranchlevel[-1])
        return set(entities)

    def get_entity_query_fn(self):
        return self.que.get_entity_query_fn(self.current_triples, self.current_triples_varvals, self.lastvarforbranchlevel[-1])


def get_vnt_for_butd(x):
    try:
        x = re.sub("\s+", u" ", x)
    except Exception:
        q.embed()
    tokens = x.split()
    t = Traversal()

    for token in tokens:
        token = re.sub(":(-?)<https", u":\g<1><http", token)
        vnt = t.next_token(token)

    # check vnts
    vnts = t.final_vnts

    invnt = [token in vnt for token, vnt in zip(tokens, vnts[:-1])]

    complete = True
    if not all(invnt):
        print(tokens)
        print(invnt)
        complete = False
        # sys.exit()

    return vnts, complete


def get_vnt_for_dataset(p="../../../../datasets/lcquad/",
                        files=("lcquad.multilin",),
                        outp="lcquad.multilin.vnt.smaller"):
    tt = q.ticktock("vnt builder")
    for file in files:
        tt.tick("doing {}".format(file))
        filep = p + file
        vnts = {}
        incompletevnts = []
        with codecs.open(filep, encoding="utf-8-sig") as f:
            i = 0
            for line in f:
                m = re.match("(Q\d+\.P\d+):\s(.+)", line)
                if m:
                    qpid, parse = m.group(1), m.group(2)
                    vnt, complete = get_vnt_for_butd(parse)
                    vntlen = [len(vnte) for vnte in vnt]
                    maxvntlen = max(vntlen)
                    if not complete:
                        incompletevnts.append(qpid)
                    vnts[qpid] = vnt
                    i += 1
                    if i % 1 == 0:
                        tt.msg("{} {}, max: {}, {}".format(i, qpid, maxvntlen, str(vntlen)))
                    # break
        if outp is None:
            outp = p + file + ".vnt"
        pickle.dump(vnts, open(outp, "w"))
        print("{} incomplete: {}".format(len(incompletevnts), " ".join(incompletevnts)))
        tt.tock("done {}".format(file))
        # reloaded = pickle.load(open(p+file+".vnt"))
        # q.embed()


def get_propchains_for_ent(ent, upto=1):
    que = Querier(relwhitelistp="../../../../datasets/lcquad/relations.txt")
    chains = set()
    prevchains = set()
    newprevchains = set()
    i = 1
    singlerelchains = que.get_relations_of_id(ent, incl_reverse=True)
    for singlerelchain in singlerelchains:
        chains.add((singlerelchain,))
        prevchains.add((singlerelchain,))
    i += 1
    if upto >= 2:
        for prevchain in prevchains:
            m = re.match(":(-?)(.+)", prevchain[0])
            if m.group(1) == "-":
                triples = [("?var0", m.group(2), ent)]
            else:
                triples = [(ent, m.group(2), "?var0")]
            newrels = que.get_relations_of_value(triples, {}, "?var0")
            for newrel in newrels:
                chains.add((prevchain[0], newrel))
        # tworelchains = que.get_twohop_chains_from_entity(ent, incl_reverse=True)
        # for tworelchain in tworelchains:
        #     chains.add(tworelchain)
    if upto > 2:
        raise q.SumTingWongException("chains of more than 2 hops not supported")
    return chains


def chains_replace_property(chains):
    newchains = set()
    for chain in chains:
        newchain = []
        for chainrel in chain:
            m = re.match("(:-?<http://dbpedia\.org/)property/([^>]+>)", chainrel)
            if m:
                newrel = m.group(1) + u"ontology/" + m.group(2)
            else:
                newrel = chainrel
            newchain.append(newrel)
        newchain = tuple(newchain)
        newchains.add(newchain)
    return newchains


def get_prop_chains_for_dataset(p="../../../../datasets/lcquad/",
                        files=("lcquad.multilin",),
                        outp="lcquad.multilin.chains",
                        upto=2, replace_property=True):
    tt = q.ticktock("propchains builder")
    import os
    for file in files:
        if outp is None:
            outp = p + file + ".vnt"
        if os.path.isfile(outp):
            with open(outp) as outf:
                chains = pickle.load(outf)
        else:
            chains = {}
        tt.tick("doing {}".format(file))
        filep = p + file

        checkpointfreq = 10
        with codecs.open(filep, encoding="utf-8-sig") as f:
            i = 0
            for line in f:
                m = re.match("(Q\d+\.P\d+):\s(.+)", line)
                if m:
                    qpid, parse = m.group(1), m.group(2)
                    if qpid in chains:
                        continue
                    startent = parse.split()[0]
                    qpidchains = get_propchains_for_ent(startent, upto=upto)
                    if replace_property:
                        qpidchains = chains_replace_property(qpidchains)
                    chains[qpid] = qpidchains
                    i += 1
                    if i % 1 == 0:
                        tt.msg("{} {}, len: {}".format(i, qpid, str(len(qpidchains))))
                    if (i+1) % checkpointfreq == 0:
                        with open(outp, "w") as outf:
                            pickle.dump(chains, outf)
                        tt.msg("checkpoint!")
                    # break

        with open(outp, "w") as outf:
            pickle.dump(chains, outf)
        tt.tock("done {}".format(file))
        # reloaded = pickle.load(open(p+file+".vnt"))
        # q.embed()


def run(query="<http://dbpedia.org/resource/Buckhurst_Hill_County_High_School> :<http://dbpedia.org/ontology/localAuthority> <<BRANCH>> :-<http://dbpedia.org/property/placeOfBurial> <http://dbpedia.org/resource/Elizabeth_of_Rhuddlan> <<JOIN>> <<RETURN>>"):
    q = Querier()
    print(q.get_types_of_id("<http://dbpedia.org/resource/Lenka>"))
    # get_vnt_for_butd(u"<http://dbpedia.org/resource/Pavel_Moroz> :<http://dbpedia.org/property/hometown> <<BRANCH>> :-<http://dbpedia.org/ontology/deathPlace> <http://dbpedia.org/resource/Yakov_Estrin> <<JOIN>> <<RETURN>>")
    get_vnt_for_butd(u"<http://dbpedia.org/resource/Selwyn_Lloyd> :<http://dbpedia.org/ontology/primeMinister> <<EQUALS>> <http://dbpedia.org/resource/Winston_Churchill> <<RETURN>>")
    get_vnt_for_dataset()


if __name__ == "__main__":
    q.argprun(get_prop_chains_for_dataset)