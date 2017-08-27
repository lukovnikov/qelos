import rdflib, re, time
from SPARQLWrapper import SPARQLWrapper, JSON
from SPARQLWrapper.SPARQLExceptions import EndPointInternalError, EndPointNotFound
from teafacto.util import iscollection
from IPython import embed


class Traverser(object):

    rel_blacklist = ["http://rdf\.freebase\.com/key/.+",
                     "http://www\.w3\.org/.+",
                     "http://rdf\.freebase\.com/ns/common\.topic\..+",
                     "http://rdf\.freebase\.com/ns/type\.object\..+",]
    rel_unblacklist = ["http://rdf\.freebase\.com/ns/common\.topic\.notable_types",
                       "http://rdf\.freebase\.com/ns/common\.topic\.notable_for",
                        "http://rdf\.freebase\.com/ns/type\.object\.name"]

    limit = 50

    def __init__(self, address="http://drogon:9890/sparql", **kw):
        #address = "http://localhost:9890/sparql"        # TODO remote testing
        super(Traverser, self).__init__(**kw)
        self.sparql = SPARQLWrapper(address)
        self.sparql.setReturnFormat(JSON)

    def name(self, mid):
        q = "SELECT ?o WHERE {{ <{}> <http://rdf.freebase.com/ns/type.object.name> ?o  FILTER (lang(?o) = 'en')}}"\
            .format("http://rdf.freebase.com/ns/{}".format(mid))
        #print q
        ret = set()
        res = self._exec_query(q)
        results = res["results"]["bindings"]
        for result in results:
            rete = result["o"]["value"]
            ret.add(rete)
        return list(ret)[0] if len(ret) > 0 else None

    def get_sparql_from_tree(self, tree):
        tokens = tree.split()
        triples = []
        varstack = []
        argmaxer = None
        varname = 0
        orderby = None
        i = 0
        def fbfy(x):
            return "<http://rdf.freebase.com/ns/{}>".format(x)
        while i < len(tokens):
            token = tokens[i]
            if token[0] == ":":     # relation ==> hop
                if argmaxer is None:
                    newvar = "?var{}".format(varname)
                    varname += 1
                    if token[:8] == ":reverse":
                        triple = [newvar, fbfy(token[9:]), varstack[-1]]
                    else:
                        triple = [varstack[-1], fbfy(token[1:]), newvar]
                    triples.append(triple)
                    varstack[-1] = newvar
                else:
                    pass
            elif re.match("[a-z]+\..+", token):     # entity ==> make ptr
                varstack.append(fbfy(token))
            elif token == "ARGMAX":     # do argmax
                assert(argmaxer is None)
                argmaxer = (tokens[i+1][1:], "max")
            elif token == "ARGMIN":
                assert(argmaxer is None)
                argmaxer = (tokens[i+1][1:], "min")
            elif token == "<JOIN>":     # join, execute argmaxers
                topvar = varstack[-1]
                if argmaxer is not None:    # ignore auxptr, do argmax
                    if argmaxer[0] != "reverse:None":
                        argvar = "?var{}".format(varname)
                        varname += 1
                        triples.append([varstack[-1], fbfy(argmaxer[0][8:]), argvar])
                    else:
                        argvar = varstack[-1]
                    orderby = [argvar, "DESC" if argmaxer[1] == "max" else "ASC"]
                    argmaxer = None
                else:       # intersection of top-2 sets in stack
                    # replace top var in stack with previous var in stack
                    prevvar = varstack[-2]
                    for triple in triples:
                        if triple[0] == topvar:
                            triple[0] = prevvar
                        if triple[2] == topvar:
                            triple[2] = prevvar
                    varstack[-1] = prevvar
                    del varstack[-2]        # delete aux ptr (previous pointer)
            elif token == "<RETURN>":
                pass
            else:
                raise Exception("unsupported token: {}".format(token))
            i += 1
        q = None
        #print triples
        #print orderby
        if len(triples) == 0:
            q = "SELECT DISTINCT ?x WHERE {{ VALUES ?x {{ {} }}}}".format(varstack[-1])
        else:
            for triple in triples:
                if triple[0] == varstack[-1]:
                    triple[0] = "?x"
                if triple[2] == varstack[-1]:
                    triple[2] = "?x"
                if orderby and orderby[0] == varstack[-1]:
                    orderby[0] = "?x"
            q_bgp = " .\n".join([" ".join([x for x in triple]) for triple in triples])
            q = "SELECT ?x WHERE {{\n{}\n}}".format(q_bgp)
            if orderby is not None:
                q = q + "\nORDER BY {}({}) LIMIT 1".format(orderby[1], orderby[0])
        return q

    def exec_tree_and_get_relations(self, subtree, only_reverse=False, incl_reverse=False, noresult=False):
        coresparql = self.get_sparql_from_tree(subtree)
        toexec = coresparql
        if "LIMIT" not in toexec:       # HACK
            toexec += "\n LIMIT 500"
        coresparql = toexec     # limits
        #print coresparql
        if noresult:
            res = []
        else:
            res = self._exec_query(toexec)
            res = [rese["x"]["value"] for rese in res["results"]["bindings"]]
        if incl_reverse:
            relgetter = "{{ ?x ?r1 ?o }} UNION {{ ?o ?r2 ?x }} ."
        elif only_reverse:
            relgetter = "?o ?r2 ?x ."
        else:
            relgetter = "?x ?r1 ?o ."
        relsparql = """SELECT DISTINCT {} WHERE\n
            {{ {}\n
            {{ {} }}\n
            }}""".format("?r1 ?r2" if incl_reverse else "?r2" if only_reverse else "?r1",
                         relgetter, coresparql)
        #print relsparql
        relres = self._exec_query(relsparql)
        fwd = set()
        rew = set()
        if incl_reverse:
            fwd = set([rese["r1"]["value"] for rese in relres["results"]["bindings"] if "r1" in rese])
            rew = set([rese["r2"]["value"] for rese in relres["results"]["bindings"] if "r2" in rese])
        elif only_reverse:
            rew = set([rese["r2"]["value"] for rese in relres["results"]["bindings"] if "r2" in rese])
        else:
            fwd = set([rese["r1"]["value"] for rese in relres["results"]["bindings"] if "r1" in rese])
        # map
        fwd = [re.match("http://rdf\.freebase\.com/ns/(.+)", e).group(1) for e in fwd if re.match("http://rdf\.freebase\.com/ns/(.+)", e)]
        rew = [re.match("http://rdf\.freebase\.com/ns/(.+)", e).group(1) for e in rew if re.match("http://rdf\.freebase\.com/ns/(.+)", e)]
        #print fwd, rew
        rels = set([fwde.decode("utf8") for fwde in fwd] + ["reverse:"+rewe for rewe in rew])
        return res, rels

    def get_relations_of(self, mid, only_reverse=False, incl_reverse=False):
        revrels = set()
        if not iscollection(mid):
            mid = (mid,)
        for mide in mid:        # check if every mid in input is valid
            if not re.match("[a-z]{1,2}\..+", mide):
                return set()
        if incl_reverse:
            revrels = self.get_relations_of(mid, only_reverse=True, incl_reverse=False)
            revrels = {"reverse:{}".format(revrel) for revrel in revrels}
        if only_reverse:
            q = "SELECT DISTINCT(?p) WHERE {{ ?o ?p ?s VALUES ?s {{ {} }} }}"\
                .format(" ".join(["<http://rdf.freebase.com/ns/{}>".format(srce) for srce in mid]))
        else:
            q = "SELECT DISTINCT(?p) WHERE {{ ?s ?p ?o VALUES ?s {{ {} }} }}"\
                .format(" ".join(["<http://rdf.freebase.com/ns/{}>".format(srce) for srce in mid]))
        #print q
        ret = set()
        res = self._exec_query(q)
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
                rete = re.match("http://rdf\.freebase\.com/ns/(.+)", rete).group(1)
                ret.add(rete)
        ret.update(revrels)
        return ret

    def hop(self, src, rel):
        if not iscollection(src):
            src = (src,)
        reverse = False
        if re.match("^reverse:(.+)$", rel):
            reverse = True
            rel = re.match("^reverse:(.+)$", rel).group(1)
        q = "SELECT DISTINCT({}) WHERE {{ ?s <{}> ?o VALUES {} {{ {} }} }} LIMIT {}"\
            .format("?o" if reverse == False else "?s",
                    "http://rdf.freebase.com/ns/{}".format(rel),
                    "?s" if reverse == False else "?o",
                    " ".join(["<http://rdf.freebase.com/ns/{}>".format(srce) for srce in src]),
                    self.limit)
        #print q
        ret = set()
        res = self._exec_query(q)
        results = res["results"]["bindings"]
        for result in results:
            rete = result["o" if reverse is False else "s"]["value"]
            retem = re.match("http://rdf\.freebase\.com/ns/(.+)", rete)
            if not retem:
                ret.add(rete)
            else:
                rete = retem.group(1)
                ret.add(rete)
        return ret

    def _exec_query(self, q):
        self.sparql.setQuery(q)
        retries = 10
        while True:
            try:
                res = self.sparql.query().convert()
                return res
            except (EndPointInternalError, EndPointNotFound), e:
                print "retrying {}".format(retries)
                retries -= 1
                if retries < 0:
                    raise e
                time.sleep(1)

    def join(self, a, b):
        return a & b

    def argmaxmin(self, src, rel, mode="max"):
        assert(iscollection(src))
        reverse = False
        if re.match("^reverse:(.+)$", rel):
            reverse = True
            rel = re.match("^reverse:(.+)$", rel).group(1)
        assert(reverse is True)
        q = "SELECT ?x WHERE {{ ?x <{}> ?v VALUES ?x {{ {} }} }} ORDER BY {}(?v) LIMIT 1"\
            .format("http://rdf.freebase.com/ns/{}".format(rel),
                    " ".join(["<http://rdf.freebase.com/ns/{}>".format(srce) for srce in src]),
                    "DESC" if mode == "max" else "ASC")
        #print q
        ret = set()
        res = self._exec_query(q)
        results = res["results"]["bindings"]
        for result in results:
            rete = result["x"]["value"]
            rete = re.match("http://rdf\.freebase\.com/ns/(.+)", rete).group(1)
            ret.add(rete)
        return ret

    def argmax(self, src, rel):
        return self.argmaxmin(src, rel, mode="max")

    def argmin(self, src, rel):
        return self.argmaxmin(src, rel, mode="min")

    def traverse_tree_subq(self, tree, entdic=None, incl_rev=True):
        """ consumes tree-lin query token by token, queries endpoint after every token
            to get intermediate result and next valid relations"""
        # higher-level code, maintains stack of partial or sub-queries
        tokens = tree.split()
        qstack = []     # stack of query subtrees (lists of tree-lin symbols)
        i = 0
        validrels = []
        inargmaxer = False
        while i < len(tokens):
            token = tokens[i]
            token = entdic[token] if token in entdic else token
            validrelses = set()
            if token[0] == ":" or token == "<RETURN>":     # relation
                qstack[-1].append(token)
                if not inargmaxer:
                    result, validrelses = self.exec_tree_and_get_relations(" ".join(qstack[-1]),
                                                    incl_reverse=incl_rev, noresult=token!="<RETURN>")
                    if token == "<RETURN>":
                        validrelses = set()
                    else:
                        if token[1:] not in validrels[-1]:
                            print "token {} not there".format(token)
                            if True:      # too many results
                                print " !!! too many results, adding {} (to {} other)".format(token[1:], len(validrelses))
                                #validrels[-1].add(token[1:])
                            else:
                                assert(token[1:] in validrels[-1])
                else:
                    result, validrelses = [], set()
            elif re.match("[a-z]+\..+", token) or token == "ARGMAX" or token == "ARGMIN":
                qstack.append([token])
                if token == "ARGMAX" or token == "ARGMIN":
                    validrelses = validrels[-1]
                    validrelses = self._revert_rels(validrelses, only_reverse=True)
                    inargmaxer = True
                    validrelses.add("reverse:None")
                else:
                    result = {token}
                    validrelses = self.get_relations_of({token}, incl_reverse=incl_rev)
            elif token == "<JOIN>":
                if inargmaxer:
                    inargmaxer = False
                qstack[-1] = qstack[-2] + qstack[-1] + [token]
                del qstack[-2]
                result, validrelses = self.exec_tree_and_get_relations(" ".join(qstack[-1]), incl_reverse=incl_rev)
            else:
                raise Exception("unrecognized token {}".format(token))
            validrels.append(validrelses)
            #print qstack, len(validrels[-1]), validrels[::-1]
            i += 1
        return result, validrels

    def traverse_tree(self, tree, entdic=None):     # DON'T use for computing answers, only for finding valid relations at every time step
        tokens = tree.split()
        ptrstack = []   # stack of pointers (sets of entity ids)
        qstack = []     # stack of query subtrees (lists of tree-lin symbols)
        i = 0
        argmaxer = None
        result = set()
        validrels = []
        while i < len(tokens):
            token = tokens[i]
            token = entdic[token] if token in entdic else token     # replace entity token by mid
            validrelses = set()
            if token[0] == ":":     # relation ==> hop
                if argmaxer is None:
                    mainpointer = self.hop(ptrstack[-1], token[1:])
                    ptrstack[-1] = mainpointer
                    qstack[-1].append(token)
                    validrelses = self.get_relations_of(ptrstack[-1], incl_reverse=True)
                else:
                    qstack[-1].append(token)
                    validrelses = self.get_relations_of(ptrstack[-1], incl_reverse=True)
            elif re.match("[a-z]+\..+", token):     # entity ==> make ptr
                ptrstack.append({token})
                qstack.append([token])
                validrelses = self.get_relations_of(ptrstack[-1], incl_reverse=True)
            elif token == "ARGMAX":     # do argmax
                assert(argmaxer is None)
                argmaxer = (tokens[i+1][1:], "max")
                #i += 1
                qstack.append([token])
                validrelses = self.get_relations_of(ptrstack[-1], incl_reverse=True)
                validrelses = self._revert_rels(validrelses, only_reverse=True)
                validrelses.add("reverse:None")
            elif token == "ARGMIN":
                assert(argmaxer is None)
                argmaxer = (tokens[i+1][1:], "min")
                #i += 1
                qstack.append([token])
                validrelses = self.get_relations_of(ptrstack[-1], incl_reverse=True)
                validrelses = self._revert_rels(validrelses, only_reverse=True)
                validrelses.add("reverse:None")
            elif token == "<JOIN>":     # join, execute argmaxers
                if argmaxer is not None:    # ignore auxptr, do argmax
                    mainpointer = self.argmaxmin(ptrstack[-1], argmaxer[0], mode=argmaxer[1])
                    ptrstack[-1] = mainpointer  # replace top of stack
                    argmaxer = None
                else:       # intersection of top-2 sets in pointer stack
                    if len(ptrstack[-1]) == self.limit == len(ptrstack[-2]):
                        pass
                    if len(ptrstack[-1]) == self.limit:     # take previous set as is, assumes top pointer is too big
                        ptrstack[-1] = ptrstack[-2]
                    elif len(ptrstack[-2]) == self.limit:   # leave top pointer as is, assumes previous pointer too big
                        pass
                    else:   # do intersection
                        mainpointer = self.join(ptrstack[-1], ptrstack[-2])
                        ptrstack[-1] = mainpointer
                    del ptrstack[-2]        # delete aux ptr (previous pointer)
                qstack[-1] = qstack[-2] + qstack[-1] + [token]
                del qstack[-2]
                validrelses = self.get_relations_of(ptrstack[-1], incl_reverse=True)
            elif token == "<RETURN>":
                result = ptrstack[-1]
            else:
                raise Exception("unsupported token: {}".format(token))
            #print [(mptr, self.name(mptr)) for mptr in mainptr]
            validrels.append(validrelses)
            i += 1
        return result, validrels

    def _revert_rels(self, rels, only_reverse=False):
        ret = set()
        for rel in rels:
            m = re.match('^reverse:(.+)$', rel)
            if m:
                if not only_reverse:
                    ret.add(m.group(1))
            else:
                ret.add("reverse:" + rel)
        return ret

    def _revert_rels_subq(self, rels, only_reverse=False):
        ret = set()
        for rel in rels:
            m = re.match('^:reverse(.+)$', rel)
            if m:
                if not only_reverse:
                    ret.add(m.group(1))
            else:
                ret.add(":reverse"+rel)
        return ret


if __name__ == "__main__":
    import sys
    t = Traverser()
    #print t.name("m.0gmm518")
    #sys.exit()
    #for x in  t.get_relations_of("m.01vsl3_", incl_reverse=True):
    #    print x
    #print " "
    #for x in t.hop(["m.01vsl3_", "m.06mt91"], "people.person.place_of_birth"):
    #    print x, t.name(x)
    #print " "
    #print t.get_relations_of(t.hop(["m.01vsl3_", "m.06mt91"], "people.person.place_of_birth"))
    #for x in t.argmax(["m.01vsl3_", "m.06mt91"], "people.person.date_of_birth"):
    #    print x, t.name(x)
    #sys.exit()
    #res = t.traverse_tree("<E0> :reverse:film.performance.film <E1> :film.actor.film <JOIN> :film.performance.character <RETURN>", {'<E0>': 'm.017gm7', '<E1>': 'm.02fgm7'})
    #res = t.traverse_tree("<E0> :location.country.currency_used <RETURN>", {'<E0>': 'm.0160w'})
    #q = t.exec_tree_and_get_relations("m.0dnv87 :base.bioventurist.product.sales :reverse:a.b.c.d m.0dnv87 :reverse:base.bioven.a.b ARGMAX :reverse:base.bioven.pop.ul <JOIN> <JOIN> :c.d.e <RETURN>")
    q = t.traverse_tree_subq("<E0> :base.bioventurist.product.sales :base.bioventurist.sales.year ARGMIN :reverse:None <JOIN> <RETURN>", entdic={"<E0>": "m.0dnv87"})
    print q
    sys.exit()
    res, validrels = t.traverse_tree("<E0> :sports.sports_team.championships ARGMAX :reverse:time.event.start_date <JOIN> <RETURN>", {'<E0>': 'm.01yjl'})
    #res, validrels = t.traverse_tree("<E0> :people.person.sibling_s :people.sibling_relationship.sibling m.05zppz :reverse:people.person.gender <JOIN> <RETURN>", {'<E0>': 'm.06w2sn5'})
    for r in res:
        print r, t.name(r)
    for vr in validrels:
        print len(vr), vr
