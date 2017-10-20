from __future__ import print_function
import qelos as q
import torch
import numpy as np
import random
import re


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

    def pp(self, with_parentheses=True, arbitrary=False):
        raise q.WiNoDoException()

    def __eq__(self, other):
        raise q.WiNoDoException()

    @classmethod
    def parse(cls, x, _toprec=True):
        """
        :param x: printed tree
        :return: tree structure
        """
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
        if label[:3] == "BIN":
            firstarg, remainder = Tree.parse(remainder, _toprec=False)
            secondarg, remainder = Tree.parse(remainder, _toprec=False)
            ret = BinaryTree(label, firstarg, secondarg)
        elif label[:3] == "UNI":
            arg, remainder = Tree.parse(remainder, _toprec=False)
            ret = UnaryTree(label, arg)
        elif label[:3] == "LEA":
            ret = LeafTree(label)
        while len(remainder) > 0:
            if remainder[0] in [")", ",", " ", "("]:
                remainder = remainder[1:]
            else:
                break
        if _toprec:
            assert(len(remainder) == 0)
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

    def pp(self, with_parentheses=True, arbitrary=False):
        return self.label

    def __eq__(self, other):
        return other.label == self.label


class UnaryTree(Tree):
    def __init__(self, label, child, **kw):
        super(UnaryTree, self).__init__(label, **kw)
        self.child = child

    @property
    def is_unary(self):
        return True

    def pp(self, with_parentheses=True, arbitrary=False):
        if with_parentheses:
            formatstr = "{} ( {} )"
        else:
            formatstr = "{} {}"
        return formatstr.format(self.label, self.child.pp(with_parentheses=with_parentheses, arbitrary=arbitrary))

    def __eq__(self, other):
        return self.label == other.label and self.child == other.child


class BinaryTree(Tree):
    def __init__(self, label, child1, child2, **kw):
        super(BinaryTree, self).__init__(label, **kw)
        self.children = [child1, child2]

    @property
    def is_binary(self):
        return True

    def pp(self, with_parentheses=True, arbitrary=False):
        args = [0, 1]
        if arbitrary:
            random.shuffle(args)
        if with_parentheses:
            formatstr = "{} ( {} , {} )"
        else:
            formatstr = "{} {} {}"
        return formatstr.format(self.label,
                                self.children[args[0]].pp(with_parentheses=with_parentheses, arbitrary=arbitrary),
                                self.children[args[1]].pp(with_parentheses=with_parentheses, arbitrary=arbitrary))

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
    def __init__(self, root):
        super(Tracker, self).__init__()
        self.root = root
        self.current = self.root
        self.stack = [[self.root]]
        self.possible_paths = []    # list of stacks

    def start(self):
        self.possible_paths.append([[self.root]])
        return {self.root.label}

    def nxt(self, x):       # x is a string or nothing
        if len(self.possible_paths) == 0:   # starting (or ended so restarting)
            return None
        else:
            j = 0
            allnewpossiblepaths = []
            while j < len(self.possible_paths):     # for every possible path
                possible_path = self.possible_paths[j]
                # topstack = possible_path[-1]
                i = 0
                newpossibilities = []   # will replace possible_path, can get more possible paths from one
                while i < len(possible_path[-1]):
                    # check every node at top of possible path stack
                    # if node label is x, then clone
                    #       (1) remove node from top of stack (and top of stack too if empty)
                    #       (2) add all children of x on top of stack
                    #       (3) add this new stack as a new possible path
                    node = possible_path[-1][i]
                    if x == node.label:
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
            for topnode in possible_path[-1]:
                allnvts.add(topnode.label)
        return allnvts


def generate_random_trees(n=1000, maxleaves=6, maxunaries=2,
                          numbinaries=10, numunaries=10, numleaves=20):
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



def run(lr=0.1):
    test_tracker()
    trees = get_trees()


if __name__ == "__main__":
    q.argprun(run)