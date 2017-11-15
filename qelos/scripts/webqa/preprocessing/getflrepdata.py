import qelos as q
import re, time, pickle
from SPARQLWrapper import SPARQLWrapper, JSON
from SPARQLWrapper.SPARQLExceptions import EndPointNotFound, EndPointInternalError
from qelos.scripts.webqa.preprocessing.buildvnt import category, ARGOPT, VAR, RETURN, JOIN, ENT, REL, BRANCH, fbfy, unfbfy


############################################################
# Get data to represent tokens from target formal language #
############################################################

class Querier(object):
    def __init__(self, address="http://drogon:9890/sparql", **kw):
        # address = "http://localhost:9890/sparql"        # TODO remote testing
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

    def get_triples_of(self, entity, language=None):
        query = """SELECT DISTINCT ?p ?o WHERE {{
            {} ?p ?o .
            {}
        }}""".format(entity, "FILTER (!isLiteral(?o) || lang(?o) = \"\" || langMatches(lang(?o), '{}'))".format(language) if language is not None else "")
        res = self._exec_query(query)
        results = res["results"]["bindings"]
        alltriples = []
        literaltriples = []
        uritriples = []
        for result in results:
            p = result["p"]["value"]
            o = result["o"]["value"]
            o_type = result["o"]["type"]
            if o_type == "literal":
                literaltriples.append((entity, p, o))
            elif o_type == "typed-literal":
                literaltriples.append((entity, p, o, result["o"]["datatype"]))
            elif o_type == "uri":
                uritriples.append((entity, p, o))
            else:
                raise q.SumTingWongException("o type: {}".format(o_type))
            alltriples.append((entity, p, o))
        return alltriples, uritriples, literaltriples

    def get_entity_property(self, entities, property, language=None):
        if not q.issequence(entities):
            entities = [entities]
        entities = [fbfy(entity) for entity in entities]
        propertychain = [fbfy(p) for p in property.strip().split()]
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


def run(basep="../../../../datasets/webqsp/webqsp.",
        files=("core.test.butd.vnt", "core.train.butd.vnt",
               "test.butd.vnt", "train.butd.vnt",
               "train.time.butd.vnt", "test.time.butd.vnt"),
        outfile="all.butd.vnt.info"):
    """
     Collect all entities and relations in files and get their info.
     For entities, get title, aliases, types, names of types, notable types.
     For relations, get URL, domain, range, domain name, range name.
    """
    que = Querier()
    tt = q.ticktock("datagetter")
    # 1. collect all relations and entities from all files
    entities = set()
    relations = set()
    numfiles = 0
    numquestions = 0
    tt.tick("collecting entities and relations from all files")
    for file in files:
        numfiles += 1
        filep = basep+file
        vnt_for_dataset = pickle.load(open(filep))
        for qid, vnt_for_question in vnt_for_dataset.items():
            numquestions += 1
            for vnt_for_timestep in vnt_for_question:
                for token in vnt_for_timestep:
                    if category(token) == REL:
                        relations.add(token)
                    elif category(token) == ENT:
                        entities.add(token)
                    else:
                        pass
    tt.tock("{} entities and {} relations collected from {} files ({} questions)".format(len(entities), len(relations), numfiles, numquestions))

    # 2. get info
    batsize = 100
    # 2.a get entity info
    entityinfo = {}
    i = 0
    entities = list(entities)
    tt.tick("getting entity info")
    while i < len(entities):
        j = min(i+batsize, len(entities))
        batch = entities[i:j]
        names = que.get_entity_property(batch, "type.object.name", language="en")
        types = que.get_entity_property(batch, "type.object.type")
        typesnames = que.get_entity_property(batch, "type.object.type type.object.name", language="en")
        aliases = que.get_entity_property(batch, "common.topic.alias", language="en")
        notabletypes = que.get_entity_property(batch, "common.topic.notable_types")
        notabletypesnames = que.get_entity_property(batch, "common.topic.notable_types type.object.name", language="en")
        for entity in batch:
            entityinfo[entity] = {"name": names[entity] if entity in names else None,
                                  "types": types[entity] if entity in types else None,
                                  "typenames": typesnames[entity] if entity in typesnames else None,
                                  "aliases": aliases[entity] if entity in aliases else None,
                                  "notabletypes": notabletypes[entity] if entity in notabletypes else None,
                                  "notabletypenames": notabletypesnames[entity] if entity in notabletypesnames else None}
        tt.msg("batch {}-{}".format(i, j))
        i = j
    tt.tock("entity info got")

    # 2.b get relation info
    tt.tick("getting relation info")
    relations = list(relations)
    relationinfo = {}
    i = 0
    while i < len(relations):
        j = min(i+batsize, len(relations))
        batch = relations[i:j]
        names = que.get_entity_property(batch, "type.object.name", language="en")
        domains = que.get_entity_property(batch, "type.property.schema")
        domainsnames = que.get_entity_property(batch, "type.property.schema type.object.name", language="en")
        ranges = que.get_entity_property(batch, "type.property.expected_type")
        rangesnames = que.get_entity_property(batch, "type.property.expected_type type.object.name", language="en")
        for x in batch:
            x = x[1:]
            relationinfo[":"+x] = {"name": names[x] if x in names else None,
                                   "domain": domains[x] if x in domains else None,
                                   "range": ranges[x] if x in ranges else None,
                                   "domainname": domainsnames[x] if x in domainsnames else None,
                                   "rangename": rangesnames[x] if x in rangesnames else None,}
        tt.msg("batch {}-{}".format(i, j))
        i = j
    tt.tock("relation info got")

    # 3. merge and store
    tt.tick("saving")
    outdic = {}
    outdic.update(entityinfo)
    outdic.update(relationinfo)
    pickle.dump(outdic, open(basep+outfile, "w"))
    tt.tock("saved")

    tt.tick("reloading saved")
    reloaded = pickle.load(open(basep+outfile))
    tt.tock("reloaded")


if __name__ == "__main__":
    que = Querier()
    alltriples, uritriples, literaltriples = que.get_triples_of("<http://rdf.freebase.com/ns/m.08966>", language="en")
    print("URI TRIPLES OF {}".format("given entity"))
    for triple in uritriples:
        print(triple)
    print("LITERAL TRIPLES OF {}".format("given entity"))
    for triple in literaltriples:
        print(triple)
    # print(que.get_entity_property("government.government_position_held.office_holder", "type.property.expected_type type.object.name", language="en"))
    # q.argprun(run)