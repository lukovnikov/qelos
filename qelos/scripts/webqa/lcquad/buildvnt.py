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
        query = u"SELECT DISTINCT (?t) WHERE {{ {} <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?t }}"\
            .format(id)
        ret = set()
        res = self._exec_query(query)
        results = res["results"]["bindings"]
        for result in results:
            rete = result["t"]["value"]
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

    def get_entities_of_value(self, triples, replacements, outvar, limit=100):
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


def get_vnt_for_dataset(p="../../../../datasets/lcquad/", files=("lcquad.multilin",)):
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
        pickle.dump(vnts, open(p+file+".vnt", "w"))
        print("{} incomplete: {}".format(len(incompletevnts), " ".join(incompletevnts)))
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
    q.argprun(run)