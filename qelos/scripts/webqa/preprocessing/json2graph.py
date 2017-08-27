from __future__ import print_function
from __future__ import print_function
import qelos as q
import json, re
from nltk.corpus import stopwords
from IPython import embed


def run(trainp="../../../../datasets/webqsp/webqsp.train.json",
        testp="../../../../datasets/webqsp/webqsp.test.json",
        corechains_only=False,
        train_entity_linking=None,
        test_entity_linking=None):
    traind = json.load(open(trainp))
    testd = json.load(open(testp))
    tq2p, traingraphs = buildgraphs(traind,
                                    no_complex_order=True,
                                    no_value_constraints=True,
                                    corechains_only=corechains_only)
    xq2p, testgraphs  = buildgraphs(testd,
                                    no_complex_order=True,
                                    no_value_constraints=True,
                                    corechains_only=corechains_only)
    trainquestions = {}
    testquestions = {}
    for traindi in traind["Questions"]:
        trainquestions[traindi["QuestionId"]] = traindi
    for testdi in testd["Questions"]:
        testquestions[testdi["QuestionId"]] = testdi
    print_forms(tq2p, trainquestions, traingraphs, tofile="../../../../datasets/webqsp/webqsp.train.graph", find_partial_entity_name_matches=False)
    print_forms(xq2p, testquestions, testgraphs, tofile="../../../../datasets/webqsp/webqsp.test.graph", find_partial_entity_name_matches=False)


def print_forms(tq2p, trainquestions, traingraphs, tofile=None, find_partial_entity_name_matches=False):
    missing_entity_names = []
    total = 0
    if tofile is not None:
        tofile = open(tofile, "w")
    try:
        for qid, pids in sorted(tq2p.items(), key=lambda (x, y): int(re.match(r'[^\d]+(\d+)$', x).group(1))):
            question = trainquestions[qid]
            language_form = question["ProcessedQuestion"]
            if len(pids) > 0:
                logical_form = graphtostr(traingraphs[pids[0]])
                entity_names = set(re.findall(r"\[([^\]]+)\]", logical_form))
                entnames = set()
                for en in entity_names:
                    ens = tuple(en.split("/"))
                    if len(ens) > 1:
                        pass
                    entnames.add(ens)
                if "lord of the rings" in language_form:
                    pass
                missing_entity_name = 0
                for entitynameset in entnames:
                    entitynamesetinlanguageform = 0
                    for entitynamesetel in entitynameset:
                        if entitynamesetel in language_form:
                            logical_form = logical_form.replace("[{}]".format("/".join(entitynameset)), "[{}]".format(entitynamesetel))
                            entitynamesetinlanguageform += 1
                            break
                        elif find_partial_entity_name_matches:   # try to find substring match
                            partialname = find_partial(entitynamesetel, language_form)
                            if partialname is not None:
                                torep = "[{}]".format("/".join(entitynameset))
                                logical_form = logical_form.replace(torep, "[{}]".format(partialname))
                                entitynamesetinlanguageform += 1
                                break
                            else:
                                pass
                    if entitynamesetinlanguageform == 0:
                        missing_entity_name = 1
                        #print out
                out = u"{}\t{}\t{}\t{}".format(qid, language_form, missing_entity_name, logical_form)
                if missing_entity_name > 0:
                    missing_entity_names.append(out)
                total += 1
            else:
                out = u"{}\t{}".format(qid, language_form)
                total += 1
            if tofile is None:
                print (out)
            else:
                tofile.write("{}\n".format(out))
        for x in missing_entity_names:
            print (x)
            pass
    except Exception as e:
        if tofile is not None:
            tofile.close()
        print(question)
        raise e

    print ("{} out of {} questions have non-matching entity labels -> {}%".format(len(missing_entity_names), total, (
    total - len(missing_entity_names)) / (1. * total)))
    #print entity_names
    #embed()


def find_partial(key, hay):
    if "patriots" in hay:
        pass
    haywords = q.tokenize(hay); keywords = q.tokenize(key)
    if "".join(keywords) in haywords:
        return "".join(keywords)
    partial = []
    for i in range(len(haywords)):
        breakouter = False
        for j in range(len(keywords)):
            if (haywords[i] == keywords[j] or haywords[i] == keywords[j] + "s" or haywords[i]+"s" == keywords[j]) and haywords[i] not in set(stopwords.words("english")):  # start
                partial.append(haywords[i])
                for k in range(1, min(len(keywords) - j, len(haywords) - i )):
                    if haywords[i + k] == keywords[j+k]:
                        partial.append(haywords[i+k])
                breakouter = True
                break
            else:
                pass
        if breakouter:
            break
    if len(partial) > 0:
        return " ".join(partial)
    else:
        return None


def buildgraphs(d,
                no_value_constraints=False,
                no_entity_constraints=False,
                no_order=False,
                no_complex_order=True,
                corechains_only=False):
    # iterate over questions and their parses, output dictionary from question id to parse ids and dictionary of parses
    q2p = {}
    parsegraphs = {}
    multipleparsescount = 0
    withoutchaincount = 0
    withentconstraintcount = 0
    onlyentconstraintcount = 0
    withvalconstraintcount = 0
    argoptcount = 0
    otherordercount = 0
    onlychaincount = 0
    withordercount = 0
    numquestions = 0
    for q in d["Questions"]:
        numquestions += 1
        qid = q["QuestionId"]
        parses = q["Parses"]
        if len(parses) > 1:
            multipleparsescount += 1
            parses = [parses[0]]
        parseids = []
        for parse in parses:
            parseid = parse["ParseId"]
            parsegraph, hases, topicmid = buildgraph(parse, corechains_only=corechains_only)
            entcont, valcont, order, argopt, chain = hases
            withoutchaincount += 1 if chain is False else 0
            withentconstraintcount += 1 if entcont is True else 0
            onlyentconstraintcount += 1 if entcont is True and valcont is False and order is False else 0
            withvalconstraintcount += 1 if valcont is True else 0
            argoptcount += 1 if argopt is True else 0
            otherordercount += 1 if order is True and argopt is False else 0
            withordercount += 1 if order is True else 0
            onlychaincount += 1 if order is False and entcont is False and valcont is False else 0
            if chain and \
                    (not (valcont and no_value_constraints or
                          order and no_order or
                          entcont and no_entity_constraints or
                          order and not argopt and no_complex_order)):
                parsegraphs[parseid] = parsegraph
                parseids.append(parseid)
        q2p[qid] = parseids
    print ("number of questions: {}".format(numquestions))
    print ("number of questions with multiple parses: {}".format(multipleparsescount))
    print ("number of questions with only chain: {}".format(onlychaincount))
    print ("number of questions without chain: {}".format(withoutchaincount))
    print (
    "number of questions with:\n\tentity constraints: {} ({})\n\tvalue constraints: {}\n\torder: {} ({} of which argmax, {} others)"
    .format(withentconstraintcount, onlyentconstraintcount, withvalconstraintcount, withordercount, argoptcount,
            otherordercount))
    return q2p, parsegraphs


def buildgraph(parse, corechains_only=False):
    ret, hases, topicmid = buildgraph_from_fish(parse, corechains_only=corechains_only)
    return ret, hases, topicmid


def buildgraph_from_fish(parse, corechains_only=False):
    haschain = False
    hasentityconstraints = False
    hasvalueconstraints = False
    hasorder = False
    hasargopt = False
    haschain = False
    # fish head and tail
    qnode = OutputNode()        # tail
    #topicentity = EntityNode(parse["TopicEntityMid"], parse["TopicEntityName"]) #head
    topicentity = EntityNode(parse["TopicEntityMid"], parse["PotentialTopicEntityMention"], topicentity=True) #head
    # fish spine
    cnode = topicentity
    spinenodes = []
    if parse["InferentialChain"] is not None:
        haschain = True
        for i, rel in enumerate(parse["InferentialChain"]):
            tonode = VariableNode() if i < len(parse["InferentialChain"]) - 1 else qnode
            spinenodes.append(tonode)
            cnode.add_edge(rel, tonode)
            cnode = tonode
    # if len(parse["Sparql"]) >= len("#MANUAL SPARQL") and \
    #     parse["Sparql"][:len("#MANUAL SPARQL")] == "#MANUAL SPARQL":
    #     haschain = False
    # time
    # TODO
    # constraints
    if not corechains_only:
        for constraint in parse["Constraints"]:
            operator, argtype, arg, name, pos, pred, valtype = constraint["Operator"], constraint["ArgumentType"], constraint["Argument"], constraint["EntityName"], constraint["SourceNodeIndex"], constraint["NodePredicate"], constraint["ValueType"]
            if argtype == "Entity":
                hasentityconstraints = True
                assert(operator == "Equal")
                assert(valtype == "String")
                ent = EntityNode(arg, name)
                try:
                    edge = RelationEdge(spinenodes[pos], ent, pred)
                except IndexError as e:
                    print(parse["ParseId"])
                    break
                    #raise e
                spinenodes[pos].append_edge(edge)
                ent.append_edge(edge)
            elif argtype == "Value":
                hasvalueconstraints = True
                assert(name == "" or name == None)
                intervar = VariableNode()
                try:
                    edge = RelationEdge(spinenodes[pos], intervar, pred)
                except IndexError as e:
                    print(parse["ParseId"])
                    break
                spinenodes[pos].append_edge(edge)
                intervar.append_edge(edge)
                if operator == "LessOrEqual":
                    rel = "<="
                elif operator == "GreaterOrEqual":
                    rel = ">="
                elif operator == "Equal":
                    rel = "=="
                else:
                    raise Exception("unknown operator")
                val = ValueNode(arg, valuetype=valtype)
                edge = MathEdge(intervar, val, rel)
                intervar.append_edge(edge)
                val.append_edge(edge)
        # order
        if parse["Order"] is not None:
            hasorder = True
            orderinfo = parse["Order"]
            hasargopt = orderinfo["Count"] == 1 and orderinfo["Start"] == 0
            if hasargopt:
                argoptnode = ArgMaxNode() if orderinfo["SortOrder"] == "Descending" \
                    else ArgMinNode() if orderinfo["SortOrder"] == "Ascending" else None
                pos = orderinfo["SourceNodeIndex"]
                pred = orderinfo["NodePredicate"]
                edge = RelationEdge(spinenodes[pos], argoptnode, pred)
                spinenodes[pos].append_edge(edge)
                argoptnode.append_edge(edge)
            # TODO
    return qnode, (hasentityconstraints, hasvalueconstraints, hasorder, hasargopt, haschain), parse["TopicEntityMid"]


def graphtostr(outputnode):
    ret = ""
    return u"{1}\t{2}\t{3}\t{0}".format(*_tostr_edge_based(outputnode))
    return _tostr_rec(outputnode, "")


def _tostr_edge_based(node, top=True):
    # collect all edges:
    edges = set()
    for edge in node.edges:
        if not edge.visited:
            edges.add(edge)
            edge.visited = True
            othernodeedges = _tostr_edge_based(edge.src if edge.tgt == node else edge.tgt, top=False)
            edges.update(othernodeedges)
    if top:
        ret = ""
        varnames = {0,}
        valueconstraints = 0
        for edge in edges:
            # print edge
            if isinstance(edge.src, VariableNode):
                if edge.src._name is None:
                    edge.src._name = u"var{}".format(max(varnames) + 1)
                    varnames.add(max(varnames) + 1)
            if isinstance(edge.tgt, VariableNode):
                if edge.tgt._name is None:
                    edge.tgt._name = u"var{}".format(max(varnames) + 1)
                    varnames.add(max(varnames) + 1)
            edgestr = u"{} {} {} ; ".format(
                edge.src.value,
                edge.lbl,
                edge.tgt.value,
            )
            if isinstance(edge.src, ValueNode) or isinstance(edge.tgt, ValueNode):
                valueconstraints = 1
            # if edge.lbl == "people.marriage.type_of_union":
            #     pass
            else:
                ret += edgestr
        numedges = len(edges)
        numvars = len(varnames)-1
        return ret, numedges, numvars, valueconstraints
    else:
        return edges


def _tostr_rec(node, acc=""):
    parts = []
    isleaf = True
    for edge in node.edges:
        if isinstance(edge, VariableNode):
            pass
        if not edge.visited:
            isleaf = False
            edge.visited = True
            othernodestr = _tostr_rec(edge.src if edge.tgt == node else edge.tgt)
            edge.visited = False
            part = "({} {})".format(othernodestr, edge.lbl)
            parts.append(part)
    if isleaf:
        return node.value
    else:
        if len(parts) > 1:
            ret = "(and {})".format(" ".join(parts))
        else:
            ret = parts[0]
        return ret




class Graph(object):
    def __init__(self):
        self._nodes = []


class Node(object):
    def __init__(self):
        self._edges = []

    @property
    def edges(self):
        return self._edges

    def add_edge(self, rel, tonode):
        edg = Edge(self, tonode, rel)
        self._edges.append(edg)
        tonode._edges.append(edg)

    def append_edge(self, edge):
        self._edges.append(edge)

    @property
    def value(self):
        raise NotImplementedError()


class VariableNode(Node):
    def __init__(self, name=None):
        super(VariableNode, self).__init__()
        self._name = name

    @property
    def value(self):
        return self._name


class OrderNode(Node):
    def __init__(self, sort, start, count):
        super(OrderNode, self).__init__()
        self._sort = sort
        self._start = start
        self._count = count

    @property
    def value(self):
        return None


class ArgMaxNode(Node):
    @property
    def value(self):
        return "ARGMAX"


class ArgMinNode(Node):
    @property
    def value(self):
        return "ARGMIN"


class OutputNode(Node):
    @property
    def value(self):
        return "OUT"


class EntityNode(Node):
    def __init__(self, id, name, topicentity=False):
        super(EntityNode, self).__init__()
        self._id = id
        self._name = name
        self._topicentity = topicentity
        if self._id == "m.01m9":
            self._name += "/cities/towns/villages"
        elif self._id == "m.01mp":
            self._name += "/countries"
        elif self._id == "m.01mp":
            self._name += "/british"
        elif self._id == "m.0jzc":
            self._name += "/arabic"
        elif self._id == "m.02m29p":
            self._name += "/stewie"
        elif self._id == "m.060d2":
            self._name += "/president"
        elif self._id == "m.034qd2":
            self._name += "/hawkeye"
        #elif self._id == "m.04ztj":
        #    self._name += "/marry/married/husband/wife/wives/husbands"
        #elif self._id == "m.05zppz":
        #    self._name += "/son/father/dad/husband/brother"
        #elif self._id == "m.02zsn":
        #    self._name += "/daughter/mother/mom/wife/sister"
        #elif self._id == "m.0ddt_":
        #    self._name += "/star wars"
        #elif self._id == "m.06x5s":
        #    self._name += "/superbowl"
        #elif self._id == "m.01xljv1":
        #    self._name += "/superbowl"
        elif self._id == "m.05jxkf":
            self._name += "/college/university"
        elif self._id == "m.0m4mb":
            self._name += "/prep school/school"
        elif self._id == "m.02lk60":
            self._name += "/shrek"

    @property
    def value(self):
        try:
            return u"{}[{}]{}".format(self._id, self._name.lower(), "*" if self._topicentity else "")
        except UnicodeEncodeError, e:
            raise e


class ValueNode(Node):
    def __init__(self, value, valuetype=None):
        super(ValueNode, self).__init__()
        self._value = value
        self._valuetype = valuetype

    @property
    def value(self):
        return self._value.lower().replace(" ", "_")


class Edge(object):
    def __init__(self, src, tgt, label):
        self._src = src
        self._tgt = tgt
        self._lbl = label
        self.visited = False

    @property
    def src(self):
        return self._src

    @property
    def tgt(self):
        return self._tgt

    @property
    def lbl(self):
        return self._lbl


class RelationEdge(Edge):
    pass


class MathEdge(Edge):
    pass


class CountEdge(Edge):
    pass


if __name__ == "__main__":
    q.argprun(run)