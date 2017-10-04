from __future__ import print_function
import qelos as q
import json, re, codecs


def run(p="../../../../datasets/lcquad/"):
    data = json.load(open(p+"data_set.json"))
    maxentities = 0
    # for question in data:
        # print("{}\n\t{}".format(question["corrected_question"].encode("utf-8"),
        #                         question["sparql_query"].encode("utf-8")))
    graphs = sparql2graph()
    print("DONE")
    # q.embed()


def sparql2graph(p="../../../../datasets/lcquad/", outp="lcquad.multilin"):
    data = json.load(open(p+"data_set.json"))
    # templates = json.load(open(p+"templates.txt"))
    entitiesused = set()
    relationsused = set()
    variablesused = set()
    qcounter = 1
    with codecs.open(p+outp, "w", "utf-8-sig") as f:
        for question in data:
            print(question["corrected_question"])
            lins, entsused, relsused, varsused = _get_graph_from_sparql(question["sparql_query"])
            entitiesused |= entsused
            relationsused |= relsused
            variablesused |= varsused
            towrite = u""
            towrite += u"Q{}: ".format(qcounter)
            towrite += question["corrected_question"] + u"\n"
            parsecounter = 1
            for lin in lins:
                towrite += u"Q{}.P{}: ".format(qcounter, parsecounter)
                towrite += lin + u"\n"
                parsecounter += 1
            towrite += u"\n"
            # towrite = re.sub("\s+", u" ", towrite)
            qcounter += 1
            f.write(towrite)

    print("{} relations in dataset".format(len(relationsused)))
    print("{} entities in dataset".format(len(entitiesused)))
    print("{} vars used in dataset".format(str(variablesused)))


def _get_graph_from_sparql(sparql):
    header, body = sparql.split("WHERE {")
    header = header.strip()
    body = body.strip()[:-1]    # remove trailing "}"
    bgp = [re.sub("\s+", " ", x.strip()) for x in re.split("(?:\.\s|\s\.)", body)]

    varre = re.compile("\?(\w+)")
    resre = re.compile("<([^>]+)>")

    entitiesused = set()
    relsused = set()
    varsused = set()

    print(header, bgp)

    triples = []

    qtype, qretinfo = _get_query_type(header)
    assert (qtype is not None)

    for rawtriple in bgp:
        if len(rawtriple) > 0:
            s, p, o = rawtriple.split(" ")
            if resre.match(s):
                entitiesused.add(resre.match(s).group(1))
            if resre.match(o):
                entitiesused.add(resre.match(o).group(1))
            if resre.match(p):
                relsused.add(resre.match(p).group(1))
            else:
                raise q.SumTingWongException()
            if varre.match(s):
                varsused.add(s)
                if s == "?uri":
                    s = "OUT"
                else:
                    assert(s == "?x")
                    s = "?x"
            if varre.match(o):
                varsused.add(o)
                if o == "?uri":
                    o = "OUT"
                else:
                    assert(o == "?x")
                    o = "?x"
            triples.append((s, p, o))

    assert(len(header) > 0 and len(bgp) > 0)

    ###### getting linearizations from sparql
    # print(triples)
    lins = []
    if qtype == "ASK":
        assert(len(triples) == 1)
        s, p, o = triples[0]
        cor = u"{} {} <<EQUALS>> {} <<RETURN>>".format(s, u":"+p, o)
        lins.append(cor)
        cor = u"{} {} <<EQUALS>> {} <<RETURN>>".format(o, u":-"+p, s)
        lins.append(cor)
        print(cor)

    else:
        for curent in entitiesused:
            if re.match("http://dbpedia\.org/ontology/.+", curent):
                continue    # don't start from types
            curent = u"<{}>".format(curent)
            topicent = curent
            curtriples = filter(lambda (s, p, o): curent in (s, o), triples)
            assert(len(curtriples) == 1)
            # triples = filter(lambda (s, p, o): not (curent in (s, o)), triples)
            s, p, o = curtriples[0]
            core = [curent]
            if curent == s:
                core.append(":"+p)
                core.append(o)
                curent = o
            elif curent == o:
                core.append(":-"+p)
                core.append(s)
                curent = s
            if not curent == "OUT":
                curtriples = filter(lambda (s, p, o): curent == s and o == "OUT" or curent == o and s == "OUT", triples)
                assert(len(curtriples) == 1)
                # triples = filter(lambda (s, p, o): not (curent == s and o == "OUT" or curent == o and s == "OUT"), triples)
                s, p, o = curtriples[0]
                if curent == s:
                    core.append(":"+p)
                    core.append(o)
                elif curent == o:
                    core.append(":-"+p)
                    core.append(s)
            # get triples attached to ?x or OUT

            xtriples = filter(lambda (s, p, o): (s == "?x" and not o in ("OUT", topicent)) or (o == "?x" and not s in ("OUT", topicent)), triples)
            otriples = filter(lambda (s, p, o): (s == "OUT" and not o in ("?x", topicent)) or (o == "OUT" and not s in ("?x", topicent)), triples)

            xconstraints = []
            for s, p, o in xtriples:
                if "?x" == o:
                    p = u"-"+p
                    o = s
                toadd = u"<<BRANCH>> :{} {} <<JOIN>>".format(p, o)
                xconstraints.append(toadd)
            oconstraints = []
            for s, p, o in otriples:
                if "OUT" == o:
                    p = u"-"+p
                    o = s
                toadd = u"<<BRANCH>> :{} {} <<JOIN>>".format(p, o)
                oconstraints.append(toadd)

            cor = u" ".join(core)

            xcon = u""
            if len(xconstraints) > 0:
                xcon = u" ".join(xconstraints)
            cor = cor.replace("?x", xcon)
            ocon = u""
            if len(oconstraints) > 0:
                ocon = u" ".join(oconstraints)
            cor = cor.replace("OUT", ocon)

            if qtype == "URI":
                cor += u" <<RETURN>>"
            elif qtype == "CNT":
                cor += u" <<COUNT>> <<RETURN>>"

            lins.append(cor)

            print(cor)

    return lins, entitiesused, relsused, varsused


def _rec_build_lin(curent, triples):
    curtriples = []
    for i, (s, p, o) in enumerate(triples):
        if s == curent or o == curent:
            curtriples.append((s, p, o))


def _get_query_type(head):
    if head == "ASK":
        return "ASK", {}
    elif re.match("SELECT\sDISTINCT\s\?uri", head):
        return "URI", {"get": "?uri"}
    elif re.match("SELECT\sDISTINCT\sCOUNT\(\?uri\)", head):
        return "CNT", {"cnt": "?uri"}
    else:
        return None


if __name__ == "__main__":
    q.argprun(run)