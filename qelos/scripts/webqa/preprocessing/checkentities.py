from __future__ import print_function
import csv, re
import qelos as q
import dill as pickle


#####################################################################
## CHECK WHAT ENTITY LINKING FILE COVERS FROM IDEAL ENTITY LINKING ##
#####################################################################
TOTALTEST = 1639

def run(graphp="../../../../datasets/webqsp/webqsp.test.graph",
        elp="../../../../datasets/webqsp/webqsp.test.el.tsv",
        outp="../../../../datasets/webqsp/welllinked_qids.pkl"):
    numq = 0
    toplinkings = {}
    linkings = {}
    topicents = {}
    allents = {}
    with open(elp) as elf:
        curq = None
        prevq = None
        elf_reader = csv.reader(elf, delimiter="\t")
        for q, match, start, length, ent, title, score in elf_reader:
            ent = ent[1:].replace("/", ".")
            if q != curq:
                linkings[q] = set() if q not in linkings else linkings[q]
                # ent = "{}.{}".format(ent.group(1), ent.group(2))
                toplinkings[q] = (match, start, length, ent, score)
                curq = q
            linkings[q].add(ent)
                # print (q, match, start, length, ent, score)
    with open(graphp) as graphf:
        for line in graphf:
            splits = line.split("\t")
            numq += 1
            if len(splits) > 2:
                q, question, _, _, _, _, query = splits
                topicent = re.findall(r'([a-z])\.([^\[\s]+)\[([^\]]+)\]\*', query)[0]
                allent = re.findall(r'([a-z])\.([^\[\s]+)\[([^\]]+)\]', query)
                allents[q] = set()
                for allente in allent:
                    allents[q].add("{}.{}".format(allente[0], allente[1]))
                topicents[q] = ("{}.{}".format(topicent[0], topicent[1]), topicent[2])

    qids = set(toplinkings.keys()) & set(topicents.keys())
    print(len(qids), len(toplinkings), len(topicents))
    not_in_top_linking = set()
    not_in_linking = set()
    top_not_in_question = set()
    wellinked_qids = set()
    for qid in qids:
        if toplinkings[qid][3] != topicents[qid][0]:
            not_in_top_linking.add(qid)
            print (qid, toplinkings[qid], topicents[qid])
        else:
            wellinked_qids.add(qid)
        if topicents[qid][0] not in linkings[qid]:
            not_in_linking.add(qid)
            print (qid, linkings[qid], topicents[qid])
        if toplinkings[qid][3] not in allents[qid]:
            top_not_in_question.add(qid)

    print("{} questions with links ({} in file)".format(len(qids), numq))
    print("{} given topic entity is not top linking".format(len(not_in_top_linking)))
    print("{} given topic entity is not in linking candidates".format(len(not_in_linking)))
    print("{} top linked is not in question".format(len(top_not_in_question)))
    print("{} well-linked ones".format(len(wellinked_qids)))

    pickle.dump(wellinked_qids, open(outp, "w"))


    pass


if __name__ == "__main__":
    q.argprun(run)