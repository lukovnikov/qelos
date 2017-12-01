from __future__ import print_function
import qelos as q
import torch
import numpy as np
import random
import re
import tqdm
import sys
from collections import OrderedDict

"""
PAS trees.
"""
_accept_overcomplete_trees = True


class Tree(object):
    def __init__(self, label, **kw):
        super(Tree, self).__init__(**kw)
        self.label = label

    @property
    def is_leaf(self):
        return False

    @property
    def is_unary(self):
        return False

    @property
    def is_binary(self):
        return False

    def __str__(self):
        return self.pp()

    def __repr__(self):
        return self.pp()

    def pp(self, with_parentheses=True, with_structure_annotation=False, arbitrary=False, _last_sibling=False, _root=True):
        raise q.WiNoDoException()

    def __eq__(self, other):
        raise q.WiNoDoException()

    @classmethod
    def parse(cls, x, _toprec=True):
        """
        :param x: printed tree
        :return: tree structure
        """
        x = x.replace("<STOP>", "")
        x = x.replace("<MASK>", "")
        while len(x) > 0:
            if x[0] in [")", ",", " ", "("]:
                x = x[1:]
            else:
                break
        splits = re.split("\s", x, 1)
        if len(splits) == 2:
            label, remainder = splits
        else:
            label, remainder = splits[0], ""
        label = label.split("*")[0]
        if label[:3] == "BIN":
            firstarg, remainder = Tree.parse(remainder, _toprec=False)
            secondarg, remainder = Tree.parse(remainder, _toprec=False)
            ret = BinaryTree(label, firstarg, secondarg)
        elif label[:3] == "UNI":
            arg, remainder = Tree.parse(remainder, _toprec=False)
            ret = UnaryTree(label, arg)
        elif label[:3] == "LEA":
            ret = LeafTree(label)
        else:
            raise q.SumTingWongException("unsupported label")
        while len(remainder) > 0:
            if remainder[0] in [")", ",", " ", "("]:
                remainder = remainder[1:]
            else:
                break
        if _toprec:
            if len(remainder) > 0:
                if _accept_overcomplete_trees:
                    # print("nonzero remainder but overcomplete trees supported")
                    pass
                else:
                    raise Exception("nonzero remainder at top level while parsing tree")
            return ret
        else:
            return ret, remainder

    def track(self):
        tracker = Tracker(self)
        return tracker


class LeafTree(Tree):
    def __init__(self, label, **kw):
        super(LeafTree, self).__init__(label, **kw)

    @property
    def is_leaf(self):
        return True

    def pp(self, with_parentheses=True, with_structure_annotation=False, arbitrary=False, _last_sibling=False, _root=True):
        label = self.label
        if with_structure_annotation and (_last_sibling or _root):
            label = label + "*LS"
        return label

    def __eq__(self, other):
        return other.label == self.label


class UnaryTree(Tree):
    def __init__(self, label, child, **kw):
        super(UnaryTree, self).__init__(label, **kw)
        self.child = child

    @property
    def is_unary(self):
        return True

    def pp(self, with_parentheses=True, with_structure_annotation=False, arbitrary=False, _last_sibling=False, _root=True):
        assert(not (with_parentheses and with_structure_annotation))
        if with_parentheses:
            formatstr = "{} ( {} )"
        else:
            formatstr = "{} {}"
        label = self.label
        if with_structure_annotation and (_last_sibling or _root):
            label = label + "*LS"
        return formatstr.format(label, self.child.pp(with_parentheses=with_parentheses,
                                                     with_structure_annotation=with_structure_annotation,
                                                     arbitrary=arbitrary,
                                                     _last_sibling=True,
                                                     _root=False))

    def __eq__(self, other):
        return self.label == other.label and self.child == other.child


class BinaryTree(Tree):
    def __init__(self, label, child1, child2, **kw):
        super(BinaryTree, self).__init__(label, **kw)
        self.children = [child1, child2]

    @property
    def is_binary(self):
        return True

    def pp(self, with_parentheses=True, with_structure_annotation=False, arbitrary=False, _last_sibling=False, _root=True):
        args = [0, 1]
        if arbitrary:
            random.shuffle(args)
        assert(not (with_parentheses and with_structure_annotation))
        if with_parentheses:
            formatstr = "{} ( {} , {} )"
        else:
            formatstr = "{} {} {}"
        label = self.label
        if with_structure_annotation and (_last_sibling or _root):
            label = label + "*LS"
        return formatstr.format(label,
                                self.children[args[0]].pp(with_parentheses=with_parentheses,
                                                          with_structure_annotation=with_structure_annotation,
                                                          arbitrary=arbitrary,
                                                          _last_sibling=False,
                                                          _root=False),
                                self.children[args[1]].pp(with_parentheses=with_parentheses,
                                                          with_structure_annotation=with_structure_annotation,
                                                          arbitrary=arbitrary,
                                                          _last_sibling=True,
                                                          _root=False))

    def __eq__(self, other):
        if not isinstance(other, BinaryTree):
            return False
        same = self.label == other.label
        same &= (self.children[0] == other.children[0]
                    and self.children[1] == other.children[1]) \
                or (self.children[0] == other.children[1]
                    and self.children[1] == other.children[0])
        return same


class Tracker(object):
    """
    Tracks the tree decoding process and provides different paths for a single tree.
    Made for top-down depth-first decoding of predicate-argument structure.
    Keeps track of multiple possible positions in the tree when ambiguous.
    ===
    If doing exploration, and a token is generated by the decoder that is not in the
    valid next tokens determined by this tracker, the tracker looks at the alternative
    token and if that is not provided (None), randomly chooses one of the valid tokens.
    """
    def __init__(self, root, last_sibling_suffix="LS"):
        # !!! node name suffixes starting with * are ignored by Tree.parse()
        super(Tracker, self).__init__()
        self.root = root
        self.last_sibling_suffix = last_sibling_suffix
        self.terminated = True         # set to False to expect a <STOP> at the end of sequence
        self.current = self.root
        self.stack = [[self.root]]
        self.possible_paths = []    # list of stacks
        self._nvt = None
        self.start()

    def reset(self):
        self.current = self.root
        self.stack = [[self.root]]
        self.possible_paths = []    # list of stacks
        self._nvt = None
        self.start()

    def start(self):
        self.possible_paths.append([[self.root]])
        allnvts = {self.root.label + "*" + self.last_sibling_suffix}
        self._nvt = allnvts
        return allnvts

    def is_terminated(self):
        return len(self._nvt) == 0

    def nxt(self, x, alt_x=None):       # x is a string or nothing
        if len(self.possible_paths) == 0:   # starting (or ended so restarting)
            return None
        else:
            if x not in self._nvt and alt_x is not None:
                x = alt_x
            if x not in self._nvt:
                assert(self._nvt is not None and len(self._nvt) > 0)
                x = random.sample(self._nvt, 1)[0]
            j = 0
            xsplits = x.split("*")
            x_islastsibling = len(xsplits) > 1 and self.last_sibling_suffix in xsplits[1:]
            if x_islastsibling:
                x = xsplits[0]
            allnewpossiblepaths = []
            while j < len(self.possible_paths):     # for every possible path
                possible_path = self.possible_paths[j]
                # topstack = possible_path[-1]
                i = 0
                newpossibilities = []   # will replace possible_path, can get more possible paths from one
                node_islastsibling = len(possible_path[-1]) == 1
                while i < len(possible_path[-1]):
                    # check every node at top of possible path stack
                    # if node label is x, then clone
                    #       (1) remove node from top of stack (and top of stack too if empty)
                    #       (2) add all children of x on top of stack
                    #       (3) add this new stack as a new possible path
                    node = possible_path[-1][i]
                    if x == node.label \
                            and (x_islastsibling == node_islastsibling):
                        newpossibility = possible_path + []    # copy
                        newpossibility[-1] = possible_path[-1] + [] # replace top with copy
                        del newpossibility[-1][i]   # delete matched node from top

                        if len(newpossibility[-1]) == 0:    # if top of stack empty, remove
                            del newpossibility[-1]

                        # push children
                        if node.is_unary:
                            newpossibility.append([node.child])
                        elif node.is_binary:
                            newpossibility.append(node.children)

                        if len(newpossibility) > 0:
                            newpossibilities.append(newpossibility)
                    i += 1
                allnewpossiblepaths += newpossibilities
                if len(newpossibilities) == 0:
                    del self.possible_paths[j]
                else:
                    j += 1
            self.possible_paths = allnewpossiblepaths
        allnvts = set()
        for possible_path in self.possible_paths:
            if len(possible_path[-1]) == 1:
                topnode = possible_path[-1][0]
                allnvts.add(topnode.label + "*" + self.last_sibling_suffix)
            else:
                for topnode in possible_path[-1]:
                    allnvts.add(topnode.label)
        self._nvt = allnvts
        return allnvts


def build_dic_from_trees(trees, suffixes=["*LS"]):
    alltokens = set()
    for tree in trees:
        treestr = tree.pp(with_parentheses=False)       # take one linearization
        treetokens = set(treestr.split())
        alltokens.update(treetokens)
    indic = OrderedDict([("<MASK>", 0), ("<START>", 1), ("<STOP>", 2)])
    outdic = OrderedDict()
    outdic.update(indic)
    offset = len(indic)
    alltokens = sorted(list(alltokens))
    alltokens_for_indic = ["<RARE>"] + alltokens
    alltokens_for_outdic = ["<RARE>", "<RARE>*NC"] + alltokens
    numtokens = len(alltokens_for_indic)
    newidx = 0
    for token in alltokens_for_indic:
        indic[token] = newidx + offset
        newidx += 1
    numtokens = len(alltokens_for_outdic)
    newidx = 0
    for token in alltokens_for_outdic:
        outdic[token] = newidx + offset
        for i, suffix in enumerate(suffixes):
            outdic[token + suffix] = newidx + offset + (i + 1) * numtokens
        newidx += 1
    return indic, outdic


class GroupTracker(object):

    def __init__(self, trees):
        super(GroupTracker, self).__init__()
        self.trees = trees
        indic, outdic = build_dic_from_trees(trees)
        self.dic = outdic
        self.rdic = {v: k for k, v in self.dic.items()}
        self.trackers = []
        self.D = outdic
        self.D_in = indic
        for tree in self.trees:
            tracker = tree.track()
            self.trackers.append(tracker)

    def get_valid_next(self, eid):
        tracker = self.trackers[eid]
        nvt = tracker._nvt
        if len(nvt) == 0:   # done
            if not tracker.terminated:
                nvt = {"<STOP>"}
            else:
                nvt = {"<MASK>"}
        nvt = map(lambda x: self.dic[x], nvt)
        return nvt

    def update(self, eid, x, alt_x=None):
        tracker = self.trackers[eid]
        nvt = tracker._nvt
        if len(nvt) == 0:   # done
            if not tracker.terminated:
                tracker.terminated = True
            # don't update tracker
        else:
            x = self.rdic[x]
            tracker.nxt(x, alt_x=self.rdic[alt_x] if alt_x is not None else None)

    def is_terminated(self, eid):
        return self.trackers[eid].is_terminated()

    def pp(self, x):
        xs = [self.rdic[xe] for xe in x if xe != self.dic["<MASK>"]]
        xstring = " ".join(xs)
        return xstring

    def reset(self):
        for tracker in self.trackers:
            tracker.reset()


def generate_random_trees(n=1000, maxleaves=6, maxunaries=2,
                          numbinaries=10, numunaries=10, numleaves=20,
                          seed=None):
    if seed is not None:
        random.seed(seed)
    # one tree = list of tokens and branches, branches are lists of tokens and branches
    unique_leaves = ["LEAF{}".format(i) for i in range(numleaves)]
    unique_unaries = ["UNI{}".format(i) for i in range(numunaries)]
    unique_binaries = ["BIN{}".format(i) for i in range(numbinaries)]
    trees = []
    for i in range(n):
        # sample number of leaves:
        numleaves = random.randint(2, maxleaves)
        leaves = random.sample(unique_leaves, numleaves)
        random.shuffle(leaves)
        subtrees = [LeafTree(leaf) for leaf in leaves]
        while len(subtrees) > 1:
            j = 0
            while j < len(subtrees):
                choiceset = []
                numunaries = 0
                current_unary_check = subtrees[j]
                while current_unary_check.is_unary:
                    numunaries += 1
                    current_unary_check = current_unary_check.child
                if numunaries < maxunaries:
                    choiceset += unique_unaries
                if j < len(subtrees) - 1:
                    choiceset += unique_binaries
                if len(choiceset) == 0:
                    j += 1
                    continue
                action = random.choice(choiceset)
                if action[:3] == "BIN":
                    # must join two top branches
                    newsubtree = BinaryTree(action, subtrees[j], subtrees[j+1])
                    subtrees[j] = newsubtree
                    del subtrees[j+1]
                else:
                    newsubtree = UnaryTree(action, subtrees[j])
                    subtrees[j] = newsubtree
                j += 1
        trees.append(subtrees[0])
    return trees


def get_trees():
    trees = generate_random_trees()
    for i, tree in enumerate(trees):
        print("tree {}".format(i))
        print("\t"+tree.pp(with_parentheses=True, arbitrary=False))
        print("\t"+tree.pp(with_parentheses=True, arbitrary=True))
        print("\t"+tree.pp(with_parentheses=False, arbitrary=True))
        parse = Tree.parse(tree.pp(with_parentheses=True))
        assert(parse == tree)
        parse = Tree.parse(tree.pp(with_parentheses=False, arbitrary=True))
        assert(parse == tree)
        tracker = parse.track()
        acc = []
        chosen = []
        nvt = list(tracker.start())
        acc.append(nvt)
        while len(nvt) > 0:
            nxt = random.choice(list(nvt))
            chosen.append(nxt)
            nvt = tracker.nxt(nxt)
            acc.append(nvt)
        print(acc)
        print(chosen)
    return trees


def test_tracker():
    futree = "BINX BIN1 UNI1 LEAF1 UNI1 LEAF2 BIN1 LEAF3 LEAF4"
    futree = Tree.parse(futree)
    uniques = set()
    for i in range(100):
        acc = []
        chosen = []
        tracker = futree.track()
        nvt = tracker.start()
        acc.append(nvt)
        while len(nvt) > 0:
            nxt = random.choice(list(nvt))
            chosen.append(nxt)
            nvt = tracker.nxt(nxt)
            acc.append(nvt)
        # print(chosen)
        rectree = Tree.parse(" ".join(chosen))
        print(futree)
        print(rectree)
        print(acc)
        print(" ")
        uniques.add(rectree.pp())
        assert(rectree == futree)
    print(len(uniques))
    for uniqueone in uniques:
        print(uniqueone)


def test_dic_builder():
    trees = get_trees()
    indic, outdic = build_dic_from_trees(trees, ["*LS"])
    for k, v in indic.items():
        assert(outdic[k] == v)
        if v > 3:
            assert(outdic[k+"*LS"] - len(indic) + 4 == v)
    print("DIC BUILDER TESTED")


def test_parse_overcomplete_trees():
    treestr = "BIN1 BIN2 LEAF1 LEAF2 UNI1 LEAF3 LEAF4 <STOP>"
    tree = Tree.parse(treestr)
    print(tree.pp(with_structure_annotation=True, with_parentheses=False))


def run(lr=0.1):
    test_parse_overcomplete_trees()
    sys.exit()
    test_dic_builder()
    test_tracker()
    trees = get_trees()
    build_dic_from_trees(trees)


if __name__ == "__main__":
    q.argprun(run)