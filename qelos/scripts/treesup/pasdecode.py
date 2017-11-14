import torch
import qelos as q
from qelos.scripts.treesup.pastrees import GroupTracker, generate_random_trees, Tree, BinaryTree, UnaryTree, LeafTree
import numpy as np
import re


OPT_LR = 0.1


def run(lr=OPT_LR,
        ):
    # load data
    # TODO: put last sibling dict etc in trackers
    ism = q.StringMatrix()
    ism.tokenize = lambda x: x.split()
    osm = q.StringMatrix()
    osm.tokenize = lambda x: x.split()
    numtrees = 1000
    trees = generate_random_trees(numtrees)
    for tree in trees:
        treestring = tree.pp(with_parentheses=False, arbitrary=True)
        ism.add(treestring)
        osm.add(treestring)     # only need dictionary from here
    ism.finalize()      # ism provides source sequences
    # tree decoding tracker
    osm.finalize()
    oidic = {}      # dic for target symbols at input
    oodic = {}      # dic for target symbols at output
    oidic.update(osm.D)
    oodic.update(osm.D)
    newidx = max(osm.D.values()) + 1
    oidic["<START>"] = newidx
    oodic["<START>"] = newidx
    newidx += 1
    for k, v in osm.D.items():
        if not re.match("<[^>]+>", k):
            oodic[v + "-LS"] = newidx
            newidx += 1



if __name__ == "__main__":
    q.argprun(run)