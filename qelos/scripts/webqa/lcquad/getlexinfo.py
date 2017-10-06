from qelos.scripts.webqa.lcquad.buildvnt import Querier, category, ENT, REL, TYP
import qelos as q
import re, sys, dill as pickle, unidecode


def get_lex_info_for_vnt(p="../../../../datasets/lcquad/lcquad.multilin.vnt",
                         outp="../../../../datasets/lcquad/lcquad.multilin.lex",
                         batsize=100):
    tt = q.ticktock("lexinfogetter")
    tt.tick("loading")
    vntses = pickle.load(open(p))
    tt.tock("loaded").tick("extracting")
    que = Querier()
    entities = set()
    types = set()
    relations = set()
    for qpid, vnts in vntses.items():
        for vnt in vnts:
            for vnte in vnt:
                if category(vnte) == REL:
                    relations.add(vnte)
                elif category(vnte) == ENT:
                    entities.add(vnte)
                elif category(vnte) == TYP:
                    types.add(vnte)
    tt.msg("{} entities, {} relations and {} types".format(len(entities), len(relations), len(types)))
    tt.tock("extracted").tick("getting types")
    typesofents = {}
    i = 0
    entities = list(entities)
    while i < len(entities):
        j = min(i+batsize, len(entities))
        entstoget = entities[i:j]
        gottypes = que.get_types_of_id(entstoget)
        typesofents.update(gottypes)
        for ent, typesofent in gottypes.items():
            types.update(typesofent)
        print(i)
        # break
        i += batsize
    tt.tock("got types").tick("idf-ing types")
    typecounts = {k: 0 for k in types}
    for ent, typeses in typesofents.items():
        for typese in typeses:
            typecounts[typese] += 1
    besttypesofents = {}
    mostbesttypesofents = {}
    for ent in typesofents:
        typesofent = list(typesofents[ent])
        typesofent = sorted(typesofent, key=lambda x: typecounts[x], reverse=False)
        besttypesofents[ent] = typesofent[0:min(min(max(1, len(typesofent) // 2), 3), len(typesofents))]
        mostbesttypesofents[ent] = typesofent[0]
    tt.tock("idf-ed types")
    uniquerels = set()
    for rel in relations:
        if re.match(":-.+", rel):
            rel = rel[0:1] + rel[2:]
        uniquerels.add(rel)
    tt.tick("extracting labels")
    labels = {}
    for rel in uniquerels:
        if rel == ":<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>":
            label = "type"
        else:
            m = re.match(":<http://dbpedia.org/(?:property|ontology)/([^>]+)>", rel)
            label = m.group(1)
            label = re.sub("([a-z])([A-Z])", "\g<1> \g<2>", label)
        labels[rel] = label.lower()
    # q.embed()
    for typ in types:
        m = re.match("<http://dbpedia\.org/ontology/([^>]+)>", typ)
        if not m:
            print("{} didn't match".format(typ))
            break
        label = m.group(1)
        label = re.sub("([a-z])([A-Z])", "\g<1> \g<2>", label)
        labels[typ] = label.lower()
    # q.embed()
    for ent in entities:
        m = re.match("<http://dbpedia.org/resource/([^>]+)>", ent)
        label = m.group(1)
        origlabel = label
        label = re.sub("_", " ", label)
        origlabel = label
        m = re.match("([^\(]+)(?:\([^\)]+\))?", label)
        if m:
            label = m.group(1)
        label = unidecode.unidecode(label)
        label = " ".join(q.tokenize(label))
        labels[ent] = (label.strip(), origlabel)
    tt.tock("extracted labels from uris")

    tt.tick("dumping")
    pickle.dump({"labels": labels, "besttypes": besttypesofents, "verybesttypes": mostbesttypesofents, "alltypes": typesofents},
                open(outp, "w"))
    tt.tock("dumped").tick("reloading")
    reloaded = pickle.load(open(outp))
    tt.tock("reloaded")

    q.embed()


if __name__ == "__main__":
    q.argprun(get_lex_info_for_vnt)



