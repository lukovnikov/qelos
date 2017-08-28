import qelos as q
import pickle
from unidecode import unidecode
from qelos.scripts.webqa.preprocessing.buildvnt import REL, ENT, category, fbfy, unfbfy


################################################
#  Prepare data for representation building.   #
#  Uses .info produced by getflrepdata.py      #
################################################

# Need to build separate dictionaries for entities and relations
# Use pretrained word embedding dictionary as id's for words in matrices

def run(p="../../../../datasets/webqsp/webqsp.all.butd.vnt.info",
        outp="../../../../datasets/webqsp/flmats/"):
    tt = q.ticktock("matrix builder")

    # load info file
    tt.tick("loading info")
    info = pickle.load(open(p))
    tt.tock("info loaded")

    # separate
    entityinfo = {}
    relationinfo = {}
    for key, val in info.items():
        if category(key) == ENT:
            entityinfo[key] = val
        elif category(key) == REL:
            relationinfo[key] = val
        else:
            raise q.SumTingWongException()

    # build and save entity matrices
    edic, names, nameschars, aliases, typenames, notabletypenames, types \
        = build_entity_matrices(entityinfo)
    pickle.dump(edic, open(outp+"webqsp.entity.dic", "w"))
    names.save(outp+"webqsp.entity.names.sm")
    nameschars.save(outp+"webqsp.entity.names.char.sm")
    # aliases.save(outp+"webqsp.entity.aliases.sm")
    typenames.save(outp+"webqsp.entity.typenames.sm")
    notabletypenames.save(outp+"webqsp.entity.notabletypes.sm")
    types.save(outp+"webqsp.entity.types.sm")

    # build and save relation matrices
    rdic, names, domains, ranges, domainids, rangeids, urlwords, urltokens \
        = build_relation_matrices(relationinfo)
    pickle.dump(rdic, open(outp+"webqsp.relation.dic", "w"))
    basep = outp+"webqsp.relation."
    names.save(basep+"names.sm")
    domains.save(basep+"domains.sm")
    ranges.save(basep+"ranges.sm")
    domainids.save(basep+"domainids.sm")
    rangeids.save(basep+"rangeids.sm")
    urlwords.save(basep+"urlwords.sm")
    urltokens.save(basep+"urltokens.sm")

    # reload
    tt.tick("reloading")
    enamesreloaded = q.StringMatrix.load(outp+"webqsp.entity.typenames.sm")
    tt.tock("reloaded")


def build_entity_matrices(info):
    tt = q.ticktock("entity matrix builder")
    tt.tick("building")
    # build
    ids = []
    names = q.StringMatrix()
    names.tokenize = lambda x: q.tokenize(x) if x != "<RARE>" else [x]
    nameschars = q.StringMatrix()
    aliases = q.StringMatrix()
    aliases.tokenize = lambda x: q.tokenize(x) if x != "<RARE>" else [x]
    nameschars.tokenize = lambda x: " ".join(q.tokenize(x)) if x != "<RARE>" else [x]
    typenames = q.StringMatrix()
    typenames.tokenize = lambda x: q.tokenize(x, preserve_patterns=['<[A-Z]+>']) if x != "<RARE>" else [x]
    types = q.StringMatrix()
    types.tokenize = lambda x: x
    notabletypenames = q.StringMatrix()
    notabletypenames.tokenize = lambda x: q.tokenize(x) if x != "<RARE>" else [x]
    for key, val in info.items():
        ids.append(key)
        name = list(val["name"])[0] if val["name"] is not None else "<RARE>"
        names.add(name)
        nameschars.add(name)
        alias = " <SEP> ".join(list(val["aliases"])) if val["aliases"] is not None else "<RARE>"
        aliases.add(alias)
        typename = " <SEP> ".join(list(val["typenames"])) if val["typenames"] is not None else "<RARE>"
        typenames.add(typename)
        typ = list(val["types"]) if val["types"] is not None else ["<RARE>"]
        types.add(typ)
        notabletypename = list(val["notabletypenames"])[0] if val["notabletypenames"] is not None else "<RARE>"
        notabletypenames.add(notabletypename)
    tt.tock("built")

    tt.tick("finalizing")
    names.finalize()
    nameschars.finalize()
    aliases.finalize()
    typenames.finalize()
    notabletypenames.finalize()
    types.finalize()
    tt.tock("finalized")

    edic = dict(zip(ids, range(len(ids))))

    return edic, names, nameschars, aliases, typenames, notabletypenames, types


def build_relation_matrices(info):
    tt = q.ticktock("relation matrix builder")
    tt.tick("building")

    # build
    ids = []
    names = q.StringMatrix()
    names.tokenize = lambda x: q.tokenize(x) if x != "<RARE>" else [x]
    domains = q.StringMatrix()
    domains.tokenize = lambda x: q.tokenize(x) if x != "<RARE>" else [x]
    ranges = q.StringMatrix()
    ranges.tokenize = lambda x: q.tokenize(x) if x != "<RARE>" else [x]
    urlwords = q.StringMatrix()
    urlwords.tokenize = lambda x: q.tokenize(x, preserve_patterns=['<[A-Z]+>']) if x != "<RARE>" else [x]
    urltokens = q.StringMatrix()
    urltokens.tokenize = lambda x: x
    domainids = q.StringMatrix()
    domainids.tokenize = lambda x: x
    rangeids = q.StringMatrix()
    rangeids.tokenize = lambda x: x

    for key, val in info.items():
        ids.append(key)
        name = list(val["name"])[0] if val["name"] is not None else "<RARE>"
        names.add(name)
        domain = list(val["domainname"])[0] if val["domainname"] is not None else "<RARE>"
        domains.add(domain)
        rangename = list(val["rangename"])[0] if val["rangename"] is not None else "<RARE>"
        ranges.add(rangename)
        rangeid = list(val["range"]) if val["range"] is not None else ["<RARE>"]
        rangeids.add(rangeid)
        domainid = list(val["domain"]) if val["domain"] is not None else ["<RARE>"]
        domainids.add(domainid)
        splits = key[1:].split(".")
        if splits[0] == "user":
            splits = splits[2:]
        while len(splits) < 3:
            splits = ["default"] + splits
        url = ".".join(splits)
        urlword = " <SEP> ".join(splits)
        urlwords.add(urlword)
        urltoken = [".".join(splits[:-2]),
                    splits[-2],
                    splits[-1]]
        urltokens.add(urltoken)
    tt.tock("built")

    tt.tick("finalizing")
    names.finalize()
    domains.finalize()
    ranges.finalize()
    rangeids.finalize()
    domainids.finalize()
    urlwords.finalize()
    urltokens.finalize()
    tt.tock("finalized")

    rdic = dict(zip(ids, range(len(ids))))

    return rdic, names, domains, ranges, domainids, rangeids, urlwords, urltokens


if __name__ == "__main__":
    q.argprun(run)

