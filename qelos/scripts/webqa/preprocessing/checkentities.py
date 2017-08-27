from __future__ import print_function
import csv, re
import qelos as q


#####################################################################
## CHECK WHAT ENTITY LINKING FILE COVERS FROM IDEAL ENTITY LINKING ##
#####################################################################


def run(graphp="../../../../datasets/webqsp/webqsp.test.lin",
        elp="../../../../datasets/webqsp/webqsp.test.el.tsv"):
    toplinkings = {}
    linkings = {}
    topicents = {}
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
            if len(splits) > 2:
                q, question, _, _, _, _, query = splits
                topicent = re.findall(r'([a-z])\.([^\[\s]+)\[([^\]]+)\]\*', query)[0]
                topicents[q] = ("{}.{}".format(topicent[0], topicent[1]), topicent[2])

    qids = set(toplinkings.keys()) & set(topicents.keys())
    print(len(qids), len(toplinkings), len(topicents))
    not_in_top_linking = set()
    not_in_linking = set()
    for qid in qids:
        if toplinkings[qid][3] != topicents[qid][0]:
            not_in_top_linking.add(qid)
            print (qid, toplinkings[qid], topicents[qid])
        if topicents[qid][0] not in linkings[qid]:
            not_in_linking.add(qid)
            print (qid, linkings[qid], topicents[qid])

    print(len(not_in_top_linking))
    print(len(not_in_linking))


    pass


if __name__ == "__main__":
    q.argprun(run)