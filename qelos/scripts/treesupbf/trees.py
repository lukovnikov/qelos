# -*- coding: utf-8 -*-
from __future__ import print_function
import qelos as q
import torch
import numpy as np
import random
import re
import sys
from collections import OrderedDict
from unidecode import unidecode


class Tracker(object):
    def reset(self):
        raise NotImplemented("use subclass")

    def nxt(self, x, alt_x=None):
        raise NotImplemented("use subclass")

    def get_nvt(self):
        raise NotImplemented("use subclass")


class Trackable(object):
    @classmethod
    def parse(cls, inp, **kw):
        raise NotImplemented("use subclass")

    def track(self):
        raise NotImplemented("use subclass")

    @classmethod
    def build_dict_from(cls, sentences):
        raise NotImplemented("use subclass")


class Node(Trackable):
    """ !!! If mixed order children, children order is ordered children first, then unordered ones"""
    leaf_suffix = "NC"
    last_suffix = "LS"
    none_symbol = "<NONE>"
    root_symbol = "<ROOT>"

    suffix_sep = "*"
    label_sep = "/"
    order_sep = "#"

    def __init__(self, name, label=None, order=None, children=tuple(), **kw):
        super(Node, self).__init__(**kw)
        self.name = name    # name must be unique in a tree
        self._label = label
        self._order = order
        self.children = tuple(sorted(children, key=lambda x: x.order if x.order is not None else np.infty))
        self._ordered_children = len(self.children) > 0 and self.children[0].order is not None

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label = value

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        self._order = value

    def track(self):
        return NodeTracker(self)

    @classmethod
    def parse(cls, inp, _rec_arg=None, _toprec=True, _ret_remainder=False):
        """ breadth-first parse """
        tokens = inp
        if _toprec:
            tokens = tokens.replace("  ", " ").strip().split()

        otokens = tokens + []

        parents = _rec_arg + [] if _rec_arg is not None else None
        level = []
        siblings = []

        while True:
            head, tokens = tokens[0], tokens[1:]
            xsplits = head.split(cls.suffix_sep)
            isleaf, islast = cls.leaf_suffix in xsplits, cls.last_suffix in xsplits
            isleaf, islast = isleaf or head == cls.none_symbol, islast or head == cls.none_symbol
            x = xsplits[0]
            headname, headlabel = x, None
            if len(x.split(cls.label_sep)) == 2:
                headname, headlabel = x.split(cls.label_sep)
            headname, headorder = headname, None
            if len(headname.split(cls.order_sep)) == 2:
                headname, headorder = headname.split(cls.order_sep)

            newnode = Node(headname, label=headlabel, order=headorder)
            if not isleaf:
                level.append(newnode)
            siblings.append(newnode)

            if islast:
                if _toprec:     # siblings are roots <- no parents
                    break
                else:
                    parents[0].children = tuple(siblings)
                    siblings = []
                    del parents[0]
                    if len(parents) == 0:
                        break

        remainder = tokens

        if len(tokens) > 0:
            if len(level) > 0:
                remainder = cls.parse(tokens, _rec_arg=level, _toprec=False)
        else:
            assert (len(level)) == 0

        if _toprec:
            if len(siblings) == 1:
                ret = siblings[0]
                ret.delete_nones()
                if _ret_remainder:
                    return ret, (otokens, remainder)
                else:
                    return ret
            else:
                raise Exception("siblings is a list")
                return siblings
        else:
            return remainder

    def delete_nodes(self, *nodes):
        i = 0
        newchildren = []
        while i < len(self.children):
            child = self.children[i]
            for node in nodes:
                if child.name == node.name and child.label == node.label:
                    pass
                else:
                    newchildren.append(child)
            i += 1
        self.children = tuple(newchildren)
        for child in self.children:
            child.delete_nodes(*nodes)

    def delete_nones(self):
        self.delete_nodes(Node(Node.none_symbol))

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def num_children(self):
        return len(self.children)

    @property
    def size(self):
        return 1 + sum([child.size for child in self.children])

    def __str__(self):  return self.pp()
    def __repr__(self): return self.symbol(with_label=True, with_annotation=True, with_order=True)

    def symbol(self, with_label=True, with_annotation=True, with_order=True):
        ret = self.name
        if with_order:
            ret += "{}{}".format(self.order_sep, self.order) if self.order is not None else ""
        if with_label:
            ret += self.label_sep + self.label if self.label is not None else ""
        if with_annotation:
            if self.is_leaf and not self.name in [self.none_symbol, self.root_symbol]:
                ret += self.suffix_sep + self.leaf_suffix
        return ret

    def pptree(self, arbitrary=False, _rec_arg=False, _top_rec=True):
        return self.pp(mode="tree", arbitrary=arbitrary, _rec_arg=_rec_arg, _top_rec=_top_rec)

    def ppdf(self, mode="par", arbitrary=False):
        mode = "dfpar" if mode == "par" else "dfann"
        return self.pp(mode=mode, arbitrary=arbitrary)

    def pp(self, mode="ann", arbitrary=False, _rec_arg=None, _top_rec=True, _remove_order=False):
        assert(mode in "ann tree dfpar dfann".split())
        children = list(self.children)

        if arbitrary is True:
            # randomly shuffle children while keeping children with order in positions they were in
            fillthis = [child if child._order is not None else None for child in children]
            if None in fillthis:
                pass
            children_without_order = [child for child in children if child._order is None]
            random.shuffle(children_without_order)
            for i in range(len(fillthis)):
                if fillthis[i] is None:
                    fillthis[i] = children_without_order[0]
                    children_without_order = children_without_order[1:]
            children = fillthis
        elif arbitrary in ("alphabetical", "psychical", "omegal"):    # psychical and omegal are both reverse alphabetical
            # randomly shuffle children while keeping children with order in positions they were in
            fillthis = [child if child._order is not None else None for child in children]
            if None in fillthis:
                pass
            children_without_order = [child for child in children if child._order is None]
            sortreverse = True if arbitrary in ("psychical", "omegal") else False
            children_without_order = sorted(children_without_order, key=lambda x: x.name, reverse=sortreverse)
            # random.shuffle(children_without_order)
            for i in range(len(fillthis)):
                if fillthis[i] is None:
                    fillthis[i] = children_without_order[0]
                    children_without_order = children_without_order[1:]
            children = fillthis
        elif arbitrary in ("heavy", "light"):
            # randomly shuffle children while keeping children with order in positions they were in
            fillthis = [child if child._order is not None else None for child in children]
            children_without_order = [child for child in children if child._order is None]
            sortreverse = True if arbitrary == "heavy" else False
            children_without_order = sorted(children_without_order, key=lambda x: x.size, reverse=sortreverse)
            if None in fillthis:
                pass
            # random.shuffle(children_without_order)
            for i in range(len(fillthis)):
                if fillthis[i] is None:
                    fillthis[i] = children_without_order[0]
                    children_without_order = children_without_order[1:]
            children = fillthis

        # children = sorted(children, key=lambda x: x.order if x.order is not None else np.infty)
        if mode == "dfpar":     # depth-first with parentheses
            children = [child.pp(mode=mode, arbitrary=arbitrary, _remove_order=_remove_order) for child in children]
            ret = self.symbol(with_label=True, with_annotation=False, with_order=not _remove_order) \
                  + ("" if len(children) == 0 else " ( {} )".format(" , ".join(children)))
        if mode == "dfann":
            _is_last = True if _rec_arg is None else _rec_arg
            children = [child.pp(mode=mode, arbitrary=arbitrary, _rec_arg=_is_last_child, _remove_order=_remove_order)
                        for child, _is_last_child
                        in zip(children, [False] * (len(children)-1) + [True])]
            ret = self.symbol(with_label=True, with_annotation=False, with_order=not _remove_order) \
                  + (self.suffix_sep + "NC" if len(children) == 0 else "") + (self.suffix_sep + "LS" if _is_last else "")
            ret += "" if len(children) == 0 else " " + " ".join(children)
        if mode == "ann":
            _rec_arg = True if _rec_arg is None else _rec_arg
            stacks = [self.symbol(with_annotation=True, with_label=True, with_order=not _remove_order)
                      + ((self.suffix_sep + self.last_suffix) if (_rec_arg is True and not self.name in [self.root_symbol, self.none_symbol]) else "")]
            if len(children) > 0:
                last_child = [False] * (len(children) - 1) + [True]
                children_stacks = [child.pp(mode=mode, arbitrary=arbitrary, _rec_arg=recarg, _top_rec=False, _remove_order=_remove_order)
                                   for child, recarg in zip(children, last_child)]
                for i in range(max([len(child_stack) for child_stack in children_stacks])):
                    acc = []
                    for j in range(len(children_stacks)):
                        if len(children_stacks[j]) > i:
                            acc.append(children_stacks[j][i])
                    acc = " ".join(acc)
                    stacks.append(acc)
            if not _top_rec:
                return stacks
            ret = " ".join(stacks)
        elif mode == "tree":
            direction = "root" if _top_rec else _rec_arg
            if self.num_children > 0:
                def print_children(_children, _direction):
                    _lines = []
                    _dirs = ["up"] + ["middle"] * (len(_children) - 1) if _direction == "up" \
                        else ["middle"] * (len(_children) - 1) + ["down"]
                    for elem, _dir in zip(_children, _dirs):
                        elemlines = elem.pp(mode="tree", arbitrary=arbitrary, _rec_arg=_dir, _top_rec=False, _remove_order=_remove_order)
                        _lines += elemlines
                    return _lines

                parent = self.symbol(with_label=True, with_annotation=False, with_order=not _remove_order)
                if isinstance(parent, unicode):
                    parent = unidecode(parent)
                up_children, down_children = children[:len(children)//2], children[len(children)//2:]
                up_lines = print_children(up_children, "up")
                down_lines = print_children(down_children, "down")
                uplineprefix = "│" if direction == "middle" or direction == "down" else "" if direction == "root" else " "
                lines = [uplineprefix + " " * len(parent) + up_line for up_line in up_lines]
                parentprefix = "" if direction == "root" else '┌' if direction == "up" else '└' if direction == "down" else '├' if direction == "middle" else " "
                lines.append(parentprefix + parent + '┤')
                downlineprefix = "│" if direction == "middle" or direction == "up" else "" if direction == "root" else " "
                lines += [downlineprefix + " " * len(parent) + down_line for down_line in down_lines]
            else:
                connector = '┌' if direction == "up" else '└' if direction == "down" else '├' if direction == "middle" else ""
                s = self.symbol(with_annotation=False, with_label=True, with_order=not _remove_order)
                if isinstance(s, unicode):
                    s = unidecode(s)
                lines = [connector + s]
            if not _top_rec:
                return lines
            ret = "\n".join(lines)
        return ret

    @classmethod
    def build_dict_from(cls, trees):
        """ build dictionary for collection of trees (Nodes)"""
        allnames = set()
        alllabels = set()
        suffixes = [cls.suffix_sep + cls.leaf_suffix, cls.suffix_sep + cls.last_suffix,
                    "{}{}{}{}".format(cls.suffix_sep, cls.leaf_suffix, cls.suffix_sep, cls.last_suffix)]
        for tree in trees:
            treenames, treelabels = tree._all_names_and_labels()
            allnames.update(treenames)
            alllabels.update(treelabels)
        if len(alllabels) == 1 and alllabels == {None} or len(alllabels) == 0:
            alltokens = allnames
        else:
            alltokens = set(sum([[token+"/"+label for label in alllabels] for token in allnames], []))

        indic = OrderedDict([("<MASK>", 0), ("<START>", 1), ("<STOP>", 2),
                             (cls.root_symbol, 3), (cls.none_symbol, 4)])
        outdic = OrderedDict()
        outdic.update(indic)
        offset = len(indic)
        alltokens = ["<RARE>"] + sorted(list(alltokens))
        numtokens = len(alltokens)
        newidx = 0
        for token in alltokens:
            indic[token] = newidx + offset
            newidx += 1
        numtokens = len(alltokens)
        newidx = 0
        for token in alltokens:
            outdic[token] = newidx + offset
            for i, suffix in enumerate(suffixes):
                outdic[token +
                       suffix] = newidx + offset + (i + 1) * numtokens
            newidx += 1
        return indic, outdic

    def _all_names_and_labels(self):
        names = set()
        labels = set()
        names.add(self.name)
        for child in self.children:
            childnames, childlabels = child._all_names_and_labels()
            names.update(childnames)
            labels.update(childlabels)
        return names, labels

    def __eq__(self, other):
        if other is None:
            return False
        return self._eq_rec(other)

    def _eq_rec(self, other, _self_pos_in_list=None, _other_pos_in_list=None):
        if isinstance(other, list):
            print("other is list")
        same = self.name == other.name
        same &= self.label == other.label
        # same &= self.order == other.order
        same &= _self_pos_in_list == _other_pos_in_list
        ownchildren = self.children + tuple()
        otherchildren = other.children + tuple()
        j = 0
        while j < len(ownchildren):
            child = ownchildren[j]
            order_matters = False
            if child.order is not None:
                order_matters = True
            found = False
            i = 0
            while i < len(otherchildren):
                otherchild = otherchildren[i]
                if otherchild.order is not None:
                    order_matters = True
                equality = child._eq_rec(otherchild, _self_pos_in_list=j, _other_pos_in_list=i) \
                    if order_matters else child._eq_rec(otherchild)
                if equality:
                    found = True
                    break
                i += 1
            if found:
                otherchildren = otherchildren[:i] + otherchildren[i+1:]
                ownchildren = ownchildren[:j] + ownchildren[j+1:]
            else:
                j += 1
            same &= found
        same &= len(otherchildren) == 0 and len(ownchildren) == 0   # all children must be matched
        return same


class UniqueNode(Node):
    def track(self):
        return UniqueNodeTracker(self)


class UniqueNodeTracker(Tracker):
    none_symbol = "<NONE>"

    def __init__(self, root, labels=None, **kw):
        super(UniqueNodeTracker, self).__init__(**kw)
        # store
        self.root = Node("<ROOT>", children=(root,))
        self._possible_labels = labels
        self._allnames = self.get_available_names(self.root)
        # settings
        self._enable_allowable = True
        self._projective = False        # TODO support projective-only too
        # state
        self.current_node = self.root
        self._nvt = None
        self._anvt = None
        self._available_names = self.get_available_names(self.root.children[0]) # {}
        self._lost_children = {}
        self._stack = [self.root.name]        # queue of groups of siblings to decode
        # start
        self.start()

    def get_available_names(self, node):
        acc = {}
        acc[node.name] = node
        for child in node.children:
            acc.update(self.get_available_names(child))
        return acc

    def reset(self):
        # state
        self.current_node = self.root
        self._nvt = None
        self._anvt = None
        self._available_names = self.get_available_names(self.root.children[0])  # {}
        self._lost_children = {}
        self._stack = [self.root.name]    # list of groups of parent nodes in order they have been decoded
        # start
        self.start()

    def start(self):
        return self.compute_nvts()

    def is_terminated(self):
        return len(self._anvt) == 0

    def nxt(self, inpx, **kw):
        assert(inpx in self._anvt)        # TODO reenable

        # region process input symbol
        x, x_isleaf, x_islast = inpx + "", False, False
        inpxsplits = inpx.split(self.root.suffix_sep)
        if len(inpxsplits) > 1:
            x, x_isleaf, x_islast = inpxsplits[0], self.root.leaf_suffix in inpxsplits, self.root.last_suffix in inpxsplits
        x_isnone = x == self.none_symbol
        x_islast, x_isleaf = x_islast or x_isnone, x_isleaf or x_isnone
        x_name, x_label = x, None
        xsplits = x.split(self.root.label_sep)  # get label
        if len(xsplits) > 1:
            x_name, x_label = xsplits[0], xsplits[1]
        # endregion
        assert(x_name in self._available_names or x_isnone)

        # node = self._available_names[x_name]

        # region stored lists mgmt
        if not x_isnone:
            del self._available_names[x_name]
        if x_name in self._lost_children:
            del self._lost_children[x_name]
        if x_isleaf:
            if not x_isnone:
                own_children = list(filter(lambda x: x.name in self._available_names,
                                    self._allnames[x_name].children))
                for own_child in own_children:
                    self._lost_children[own_child.name] = own_child
        else:
            self._stack.append(x_name)  # append to list
        if x_islast:
            parent_children = self._allnames[self._stack[0]].children
            parent_children = list(filter(lambda x: x.name in self._available_names,
                                          parent_children))
            for child in parent_children:
                self._lost_children[child.name] = child

            del self._stack[0]      # x terminated first parent
        # endregion

        # selection of next node
        # self.current_node = self._allnames[self._stack[0]]

        return self.compute_nvts()

    def compute_nvts(self):
        nvt = set()
        anvt = set()
        # if len(self._stack) == 0:
        #     possible_children = [self.root]
        # else:
        if len(self._stack) > 0:
            isroot = self._stack[0] == "<ROOT>"
            parent_children = list(filter(lambda x: x.name in self._available_names,
                                          self._allnames[self._stack[0]].children))
            lost_children = list(self._lost_children.values())
            possible_children = parent_children + lost_children

            available_names = list(self._available_names.keys())
            number_non_term_parents = len(self._stack)
            number_tokens_left = len(available_names)

            if len(possible_children) > 0:
                islast = False
                if len(possible_children) == 1:     # TODO not all of lost children must be here !!!
                    islast = True
                for child in possible_children:
                    own_children = list(filter(lambda x: x.name in self._available_names,
                                               child.children))

                    token = child.name
                    token += (child.label_sep + child.label) if child.label is not None else ""

                    can_be_last = len(parent_children) < 2
                    must_be_last = len(possible_children) == 1
                    can_be_leaf = len(own_children) == 0
                    must_be_leaf = len(own_children) + len(lost_children) == 0
                    can_be_lastleaf = not (number_tokens_left > 1 and number_non_term_parents == 1)
                    must_be_lastleaf = number_non_term_parents > 0 and number_tokens_left == 1

                    if not must_be_last and not must_be_leaf and not must_be_lastleaf \
                            and number_tokens_left - 1 >= number_non_term_parents + 1:
                        nvt.add(token)
                    if can_be_last and not must_be_leaf and not must_be_lastleaf :
                        nvt.add(token+child.suffix_sep+child.last_suffix)
                    if can_be_leaf and not must_be_last and not must_be_lastleaf :
                        nvt.add(token+child.suffix_sep+child.leaf_suffix)
                    if can_be_last and can_be_leaf and can_be_lastleaf:
                        nvt.add(token+child.suffix_sep+child.leaf_suffix+child.suffix_sep+child.last_suffix)
            if len(possible_children) == 0 and not (number_tokens_left > 1 and number_non_term_parents == 1):
                nvt.add(self.none_symbol)
            self._nvt = nvt

            if self._possible_labels is None:
                possible_symbols = available_names
            else:
                possible_symbols = sum([[_name + self.root.label_sep + _label for _label in self._possible_labels]
                                        for _name in available_names])

            for possible_symbol in possible_symbols:
                if isroot:
                    anvt.add(possible_symbol + self.root.suffix_sep + self.root.last_suffix)
                    continue
                # if number_tokens_left - 1 >= number_non_term_parents + 1:   # losing at least two potential terminators (will need one child and one sibling)
                if number_tokens_left > 2 and number_non_term_parents > 0:
                    anvt.add(possible_symbol)
                # if number_tokens_left - 1 >= number_non_term_parents:   # losing at least on potential terminator in any of the below scenarios
                if number_tokens_left > 1 and number_non_term_parents > 0:
                    anvt.add(possible_symbol + self.root.suffix_sep + self.root.last_suffix)
                    anvt.add(possible_symbol + self.root.suffix_sep + self.root.leaf_suffix)
                if (number_non_term_parents > 0 or number_tokens_left == 1) and not (number_tokens_left > 1 and number_non_term_parents == 1):     # must be last one left or the rest can be sent up
                    anvt.add(possible_symbol + self.root.suffix_sep + self.root.leaf_suffix + self.root.suffix_sep + self.root.last_suffix)
            if not (number_tokens_left > 1 and number_non_term_parents == 1):
                anvt.add(self.none_symbol)
            # if len(possible_symbols) == 0:
            # if not isroot:
            #     anvt.add(self.none_symbol)

            self._anvt = anvt
            # check if every nvt is in anvt:
            if not len(nvt - anvt) == 0 or (len(nvt) == 0 and len(anvt) > 0):
                print(number_tokens_left, number_non_term_parents, len(lost_children))
                print(nvt, anvt)
                raise q.SumTingWongException()
        return nvt, anvt


class NodeTracker(Tracker):
    """ nodes need not be unique """
    def __init__(self, root, **kw):
        super(NodeTracker, self).__init__(**kw)
        self.root = root
        self.possible_paths = []
        self._nvt = None
        self.start()

    def reset(self):
        self.possible_paths = []
        self._nvt = None
        self.start()

    def start(self):
        self.possible_paths.append([[[self.root]], []])
        self._nvt = self.get_nvt()
        return self._nvt

    def is_terminated(self):
        return len(self._nvt) == 0

    def nxt(self, inpx, **kw):
        if len(self.possible_paths) == 0:
            return None
        else:
            try:
                assert(inpx in self._nvt)
            except AssertionError as e:
                print(inpx, self._nvt)
                return self._nvt
            allnewpossiblepaths = []
            xsplits = inpx.split(self.root.suffix_sep)
            x, x_isleaf, x_islast = inpx, False, False
            if len(xsplits) > 1:
                x, x_isleaf, x_islast = xsplits[0], self.root.leaf_suffix in xsplits, self.root.last_suffix in xsplits
            j = 0
            while j < len(self.possible_paths):
                possible_path = self.possible_paths[j]
                new_possible_paths = []
                i = 0
                while i < len(possible_path[-2][0]):
                    node = possible_path[-2][0][i]
                    node_islast = len(possible_path[-2][0]) == 1
                    if x == node.symbol(with_label=True, with_annotation=False, with_order=False) \
                            and node_islast == x_islast and node.is_leaf == x_isleaf:
                        new_possible_path = [[possible_path_level_group + []
                                              for possible_path_level_group in possible_path_level]
                                             for possible_path_level in possible_path]

                        del new_possible_path[-2][0][i]    # delete matched parent from group
                        if len(new_possible_path[-2][0]) == 0:  # if first parent group empty
                            del new_possible_path[-2][0]        # delete first parent group
                        if len(new_possible_path[-2]) == 0:     # if parent level empty
                            del new_possible_path[-2]           # delete parent level

                        if not node.is_leaf:
                            new_possible_path[-1].append(list(node.children))

                        if len(new_possible_path) == 1:
                            new_possible_path.append([])        # start new depth level

                        if len(new_possible_path[-2]) > 0:
                            new_possible_paths.append(new_possible_path)
                    i += 1
                allnewpossiblepaths += new_possible_paths
                if len(new_possible_paths) == 0:
                    del self.possible_paths[j]
                else:
                    j += 1
            self.possible_paths = allnewpossiblepaths

        self._nvt = self.get_nvt()
        return self._nvt

    def get_nvt(self):
        allnvts = set()
        for possible_path in self.possible_paths:
            if len(possible_path[-2][0]) == 1:
                topnode = possible_path[-2][0][0]
                allnvts.add(topnode.symbol(with_label=True, with_annotation=True, with_order=False) + self.root.suffix_sep + self.root.last_suffix)
            else:
                order_used = False
                for topnode in possible_path[-2][0]:
                    if not order_used:
                        order_used = topnode.order is not None
                    else:
                        if topnode.order is not None:
                            continue
                    allnvts.add(topnode.symbol(with_label=True, with_annotation=True, with_order=False))
        return allnvts


class Sentence(Trackable):
    leaf_suffix = "NC"
    last_suffix = "LS"
    parent_token = "<PARENT>"
    root_token = "<ROOT>"

    def __init__(self, x, **kw):
        super(Sentence, self).__init__(**kw)
        self.x = x
        self.tokenizer = lambda x: x.split()
        self._gen_tree = None       # optional tree the sentence has been parsed from

    @property
    def tokens(self):
        return self.tokenizer(self.x)

    def track(self):
        return SentenceTracker(self)

    @classmethod
    def parse(cls, inp, _rec_arg=None, _toprec=True):
        """ breadth-first parse """
        tokens = inp
        if _toprec:
            tokens = tokens.replace("  ", " ").strip().split()

        parents = _rec_arg + [] if _rec_arg is not None else None
        level = []
        siblings = []

        while True:
            head, tokens = tokens[0], tokens[1:]
            xsplits = head.split("*")
            isleaf, islast = cls.leaf_suffix in xsplits, cls.last_suffix in xsplits
            x = xsplits[0]
            isleaf = isleaf or x == cls.parent_token

            newnode = Node(x, label=None)
            if not isleaf:
                level.append(newnode)
            siblings.append(newnode)

            if islast:
                if _toprec:  # siblings are roots <- no parents
                    break
                else:
                    parents[0].children = tuple(siblings)
                    siblings = []
                    del parents[0]
                    if len(parents) == 0:
                        break

        if len(tokens) > 0:
            cls.parse(tokens, _rec_arg=level, _toprec=False)
        else:
            assert (len(level)) == 0

        if _toprec:
            assert(len(siblings) == 1)
            sentencestr = cls.get_sentence_from_tree(siblings[0])
            sentence = Sentence(sentencestr)
            sentence._gen_tree = siblings[0]
            return sentence

    @classmethod
    def get_sentence_from_tree(cls, node):
        children = list(node.children)
        if len(children) == 0:
            return node.symbol(with_label=True, with_annotation=False)
        else:
            children = [cls.get_sentence_from_tree(child) for child in children]
            ret = " ".join(children)
            ret = ret.replace(cls.parent_token, node.symbol(with_label=True, with_annotation=False))
            return ret

    @classmethod
    def build_dict_from(cls, sentences):
        alltokens = set()
        for sentence in sentences:
            assert (isinstance(sentence, Sentence))
            tokens = sentence.tokens
            alltokens.update(tokens)
        indic = OrderedDict([("<MASK>", 0), ("<START>", 1), ("<STOP>", 2),
                             (cls.root_token, 3), (cls.parent_token, 4), (cls.parent_token + "*" + cls.last_suffix, 5)])
        outdic = OrderedDict()
        outdic.update(indic)
        offset = len(indic)
        alltokens = sorted(list(alltokens))
        alltokens_for_indic = ["<RARE>"] + alltokens
        alltokens_for_outdic = ["<RARE>"] + alltokens
        numtokens = len(alltokens_for_indic)
        newidx = 0
        for token in alltokens_for_indic:
            indic[token] = newidx + offset
            newidx += 1
        numtokens = len(alltokens_for_indic)
        newidx = 0
        for token in alltokens_for_outdic:
            outdic[token] = newidx + offset
            for i, suffix in enumerate(
                    ["*" + cls.leaf_suffix, "*" + cls.last_suffix, "*" + cls.leaf_suffix + "*" + cls.last_suffix]):
                outdic[token + suffix] = newidx + offset + (i + 1) * numtokens
            newidx += 1
        return indic, outdic


class PossibleSentencePath(object):
    _parent_token = "<PARENT>"
    _sep_token = "<SEP>"
    _last_suffix = "LS"
    _leaf_suffix = "NC"
    _leaf_token = "<LEAF>"

    def __init__(self, groups, futuregroups, **kw):
        super(PossibleSentencePath, self).__init__(**kw)
        self.groups = groups
        self.ptr = 0    # pointer in the current group
        # self.left_of_parent = True
        self.after_sep = False
        self.futuregroups = futuregroups

    def clone(self):
        _g = [group+[] for group in self.groups]
        _fg = [group+[] for group in self.futuregroups]
        ret = PossibleSentencePath(_g, _fg)
        ret.ptr = self.ptr
        ret.after_sep = self.after_sep
        return ret

    @property
    def current_span(self):
        ret = []
        i = 0
        if len(self.groups) > 0:
            incl_parent = self.groups[0][i + self.ptr] == self._parent_token or (not self.after_sep and self.ptr > 0)
            while i+self.ptr < len(self.groups[0]):
                if self.groups[0][i+self.ptr] == self._parent_token:
                    if incl_parent:
                        ret.append(self.groups[0][i+self.ptr])
                    break
                else:
                    ret.append(self.groups[0][i+self.ptr])
                i += 1
            return ret
        else:
            return None

    def replace_in_span(self, pos, val, leaf=False, last=False):
        i = pos + self.ptr
        if leaf is True or val == self._sep_token:
            if leaf is True:
                assert(val is None)
                assert(pos == 0)
            # val = self._leaf_token
            if i == 0:
                del self.groups[0][0]
            else:
                left, right = self.groups[0][:i], self.groups[0][i+1:]
                if len(left) > 0:
                    self.futuregroups.append(left)
                if len(right) > 0:
                    self.groups[0] = right
                else:
                    del self.groups[0]
            self.ptr = 0
            if val == self._sep_token:
                self.after_sep = True
        else:
            if val == self._sep_token:
                # assert(self.groups[0][i] == self._parent_token)
                self.after_sep = True
            self.groups[0][i] = val
            self.ptr = i+1

    def push_future(self):  # called when last sibling is encountered
        self.after_sep = False
        # check if unrealized parents in future, don't push if there are
        nopush = len(self.groups) == 0 or self.number_parents_after(0) > 0 or (len(self.groups[0]) > self.ptr and self.groups[0][self.ptr] == self._parent_token)
        if not nopush:
            if not self.is_empty_group(self.groups[0]):
                self.futuregroups.append(self.groups[0])
            del self.groups[0]
            self.ptr = 0

    def number_parents_after(self, pos):
        """ how many parents are there in the group to the right of given current span position """
        cnt = 0; i = self.ptr + pos + 1
        while len(self.groups) > 0 and i < len(self.groups[0]):
            if self.groups[0][i] == self._parent_token:
                cnt += 1
            i += 1
        return cnt

    def number_tokens_after(self, pos):
        return len(self.groups[0]) - self.ptr - pos - 1 if len(self.groups) > 0 else 0

    def number_tokens_before(self, pos):
        cnt = 0; i = self.ptr + pos - 1
        while i > 0:
            if self.groups[0][i] in [self._sep_token]:
                break
            cnt += 1
            i -= 1
        return cnt

    def number_parents_before(self, pos):
        cnt = 0; i = self.ptr + pos - 1
        while i > 0:
            if self.groups[0][i] in [self._sep_token]:
                break
            if self.groups[0][i] == self._parent_token:
                cnt += 1
            i -= 1
        return cnt

    def has_more_parents(self, pos):
        """ check if the first group is expecting more parents to the right of given span position"""
        ret = False; i = self.ptr + pos + 1
        while i < len(self.groups[0]):
            if self.groups[0][i] == self._parent_token:
                ret = True; break
            i += 1
        return ret

    def is_last_in_group(self, pos):
        return len(self.groups[0]) == self.ptr + pos + 1

    def is_empty_group(self, x):
        if len(x) == 0:
            return True
        empty = True
        for i in range(len(x)):
            if x[i] not in [self._leaf_token, self._sep_token]:
                empty = False
                break
        return empty

    def check_update(self):
        if len(self.groups) == 0:
            for group in self.futuregroups:
                i = 0
                while i < len(group):
                    if group[i] in [self._sep_token, self._leaf_token]:
                        if not self.is_empty_group(group[:i]):
                            self.groups.append(group[:i])
                        group = group[i+1:]
                        i = 0
                    else:
                        i += 1
                if not self.is_empty_group(group):
                    self.groups.append(group)
            self.futuregroups = []
            self.ptr = 0

    def non_empty(self):
        return len(self.futuregroups) > 0 or len(self.groups) > 0


class SentenceTracker(Tracker):
    sep = "<SEP>"
    oldsep = "<OLDSEP>"

    def __init__(self, sentence, **kw):
        super(SentenceTracker, self).__init__(**kw)
        self.sentence = sentence
        self._root_tokens = self.sentence.tokens
        self.possible_paths = []
        self.ptr = 0
        self._nvt = None
        self.start()

    def reset(self):
        self.possible_paths = []
        self.ptr = 0
        self._nvt = None
        self.start()

    def start(self):
        self.possible_paths.append(PossibleSentencePath([self._root_tokens], []))
        self._nvt = set([t + "*" + self.sentence.last_suffix for t in self._root_tokens])
        return self._nvt

    def is_terminated(self):
        return len(self._nvt) == 0

    def nxt(self, inp, **kw):
        if len(self.possible_paths) == 0:
            return None
        else:
            # assert(inp in self._nvt)
            allnewpossiblepaths = []
            xsplits = inp.split("*")
            x, x_isleaf, x_islast = inp, False, False
            if len(xsplits) > 1:
                x, x_isleaf, x_islast = xsplits[0], Sentence.leaf_suffix in xsplits, Sentence.last_suffix in xsplits
            x_isleaf = x_isleaf or x == self.sentence.parent_token
            j = 0
            # check every possible path
            while j < len(self.possible_paths):
                possible_path = self.possible_paths[j]
                new_possible_paths = []     # new possible paths from previous possible paths
                i = 0
                current_span = possible_path.current_span
                while current_span is not None and i < len(current_span):
                    zero_i = False
                    node = current_span[i]
                    if node == x:
                        new_possible_path = possible_path.clone()
                        if not x_isleaf:
                            new_possible_path.replace_in_span(i, self.sentence.parent_token, last=x_islast)     # replace with parent
                        elif node == self.sentence.parent_token:
                            new_possible_path.replace_in_span(i, self.sep, last=x_islast)
                        else:
                            new_possible_path.replace_in_span(i, None, leaf=True, last=x_islast)
                            zero_i = True
                        if x_islast:
                            new_possible_path.push_future()
                        new_possible_path.check_update()
                        if new_possible_path.non_empty():
                            new_possible_paths.append(new_possible_path)
                        current_span = new_possible_path.current_span
                    if zero_i:
                        i = 0
                    else:
                        i += 1

                allnewpossiblepaths += new_possible_paths
                j += 1
            self.possible_paths = allnewpossiblepaths

        self._nvt = self.get_nvt()
        return self._nvt

    def get_nvt(self):
        allnvts = set()
        for possible_path in self.possible_paths:
            spantokens = possible_path.current_span
            for i in range(len(spantokens)):
                number_parents_after = possible_path.number_parents_after(i)
                number_tokens_after = possible_path.number_tokens_after(i)
                prev_available_as_child = False

                next_is_parent = False
                if number_tokens_after > 0:
                    next_is_parent = possible_path.groups[0][possible_path.ptr+i+1] == self.sentence.parent_token

                if spantokens[i] == self.sentence.parent_token:
                    # parent must be last if it's last symbol in its group
                    # or it's followed by another parent
                    # or the next parent needs the next sibling of this parent
                    if number_tokens_after - number_parents_after * 2 <= 0 \
                            or (number_tokens_after > 0
                                and possible_path.groups[0][possible_path.ptr+i+1] == self.sentence.parent_token):
                        allnvts.add(spantokens[i] + "*" + self.sentence.last_suffix)
                    # parent must be non-last if no more parents following and there are tokens left in its group
                    # because we'll need a parent for those tokens
                    elif number_parents_after == 0 and number_tokens_after > 0:
                        allnvts.add(spantokens[i])
                    else:
                        allnvts.add(spantokens[i] + "*" + self.sentence.last_suffix)
                        allnvts.add(spantokens[i])
                else:
                    # can_be_NC = i == 0
                    # must_be_NC = not (number_parents_after - number_parents_after * 2)
                    # can_be_LS = possible_path.after_sep #and can_have_children
                    # must_be_LS = possible_path.number_tokens_after(i) - possible_path.number_parents_after(i) * 2 >= 0
                    # can_be_NCLS = can_be_NC and can_be_LS and \
                    #               (possible_path.has_more_parents(i) or possible_path.is_last_in_group(i))
                    # must_be_NCLS = must_be_NC and must_be_LS

                    # non-parent can be *NC if it's first in span
                    # and there are sufficient possible siblings for future parents left
                    if i == 0 and (not possible_path.after_sep
                                   or (number_tokens_after - number_parents_after*2 >= 1
                                       and (not next_is_parent))):
                    # if can_be_NC and not must_be_LS and not must_be_NCLS:
                        allnvts.add(spantokens[i] + "*" + self.sentence.leaf_suffix)
                    # non-parent can be *LS if it's after SEP and it can have
                    # TODO: at least one child from before or at least one later
                    if possible_path.after_sep \
                            and (number_tokens_after - (1 if i == 0 else 0)
                                     - number_parents_after*2 >= 0):
                    # if can_be_LS and not must_be_NC and not must_be_NCLS:
                        allnvts.add(spantokens[i] + "*" + self.sentence.last_suffix)
                    # non-parent can be *NC*LS if it's first in span and it's after SEP
                    # and it's either last in group or there are more parents following
                    if i == 0 and possible_path.after_sep \
                            and (number_parents_after > 0 or possible_path.is_last_in_group(i)):
                    # if can_be_LS and can_be_NC and can_be_NCLS:
                        allnvts.add(spantokens[i] + "*" + self.sentence.leaf_suffix + "*" + self.sentence.last_suffix)
                    # non-parent can be * if it can have children and can be not last
                    # will need a child (either from before or from later) and siblings
                    if possible_path.after_sep:
                        if (not next_is_parent) and number_tokens_after - (1 if i == 0 else 0) - 1 - number_parents_after*2 >= 0:
                            #possible_path.is_last_in_group(i) and can_have_children:
                            allnvts.add(spantokens[i])
                    else:
                        if len(spantokens) - spantokens.count(self.sentence.parent_token) > 1:
                            allnvts.add(spantokens[i])
        return allnvts


class GroupTracker(object):
    """ provides single access point to a collection of trackers of trees.
        Given a collection of trackables (e.g. Nodes), build dictionary based on the trackable's .build_dict_from(),
            and initializes individual trackers for every trackable from the trackable's .track()"""
    def __init__(self, trackables):
        super(GroupTracker, self).__init__()
        if trackables is not None:
            self.trackables = trackables
            indic, outdic = trackables[0].__class__.build_dict_from(trackables)
            self.dic = outdic
            self.rdic = {v: k for k, v in self.dic.items()}
            self.trackers = []
            self.D = outdic
            self.D_in = indic
            for xe in self.trackables:
                tracker = xe.track()
                self.trackers.append(tracker)

    def get_valid_next(self, eid):
        tracker = self.trackers[eid]
        nvt = tracker._nvt
        anvt = tracker._anvt if hasattr(tracker, "_anvt") else None
        if len(nvt) == 0:  # done
            nvt = {"<MASK>"}
            anvt = {"<MASK>"} if anvt is not None else None
        nvt = map(lambda x: self.dic[x], nvt)
        anvt = map(lambda x: self.dic[x], anvt) if anvt is not None else None
        if anvt is not None:
            return nvt, anvt
        else:
            return nvt

    def update(self, eid, x, alt_x=None):
        tracker = self.trackers[eid]
        nvt = tracker._nvt
        if len(nvt) == 0:  # done
            pass
        else:
            x = self.rdic[x]
            tracker.nxt(x)

    def is_terminated(self, eid):
        return self.trackers[eid].is_terminated()

    def pp(self, x):
        xs = [self.rdic[xe] for xe in x if xe != self.dic["<MASK>"]]
        xstring = " ".join(xs)
        return xstring

    def reset(self):
        for tracker in self.trackers:
            tracker.reset()

    def __getitem__(self, item):
        """ if called with a slice, returns new GroupTracker for that slice
                (all the dictionaries are shared, the same tracker objects are reused)
            if called with an int, returns the tracker at that position """
        if isinstance(item, slice):
            gt = GroupTracker(None)
            gt.trackables = self.trackables[item]
            gt.trackers = self.trackers[item]
            gt.D = self.D
            gt.D_in = self.D_in
            gt.rdic = self.rdic
            gt.dic = self.dic
            return gt
        else:
            return self.trackers[item]
            # raise Exception("GroupTracker can only slice slices to construct new GroupTracker with multiple trackables")


def generate_random_trees(n=1000, maxleaves=6, vocsize=100, skipprob=0.3, seed=None):
    if seed is not None:
        random.seed(seed)

    names = ["N{}".format(i) for i in range(vocsize)]

    trees = []
    for i in range(n):
        numleaves = random.randint(2, maxleaves)
        leaves = random.sample(names, numleaves)
        random.shuffle(leaves)
        level = [Node(leaf) for leaf in leaves]
        while len(level) > 1:
            j = 0
            while j < len(level):
                skip = random.random() < skipprob
                if not skip:
                    numchildren = random.choice([1, 2, 2, 2, 3, 3, 4])
                    numchildren = min(numchildren, len(level) - j)
                    children = level[j: j+numchildren]
                    del level[j+1: j+numchildren]
                    name = random.sample(names, 1)[0]
                    parent = Node(name, children=tuple(children))
                    level[j] = parent
                j += 1
        trees.append(level[0])
    return trees


def run_unique_node(num_samples=10000):
    a = Node("a")
    b = Node("b")
    c = Node("c")
    d = Node("d")
    ab = Node("ab", children=(a, b))
    cd = Node("cd", children=(c, d))
    abcd = Node("abcd", children=(ab, cd))

    tracker = UniqueNodeTracker(abcd)

    # tokens = "abcd*LS ab cd*LS a*NC b*NC*LS c*NC d*NC*LS".split()
    #tracker.reset()
    #
    # for token in tokens:
    #     vnt = tracker.nxt(token)

    # tokens = "abcd*LS cd*LS ab c*NC d*NC*LS a*NC b*NC*LS".split()
    #tracker.reset()
    #
    # for token in tokens:
    #     vnt = tracker.nxt(token)
    #
    # tokens = "abcd*LS ab cd*LS a*NC*LS c*NC d*NC b*NC*LS".split()
    # tracker.reset()
    #
    # for token in tokens:
    #     vnt = tracker.nxt(token)
    #
    # tokens = "abcd*LS cd*LS ab c*NC d*NC*LS a*NC b*NC*LS".split()
    # tracker.reset()
    #
    # for token in tokens:
    #     vnt = tracker.nxt(token)

    uniquetrees = set()
    for i in range(num_samples):
        tracker.reset()
        tokens = []
        cnvt = tracker._nvt
        nvt = tracker._anvt
        while len(nvt) > 0:
            token = random.sample(nvt, 1)[0]
            # print(token, cnvt, nvt)
            # print(token)
            cnvt, nvt = tracker.nxt(token)
            # print(cnvt, nvt)
            tokens.append(token)

        if i % 1000 == 0:
            print(i)
        # print(" ".join(tokens))
        uniquetrees.add(" ".join(tokens))

        tree = Node.parse(" ".join(tokens))
        # print(" ".join(tokens))
        # print(tree.pptree())
        # tree.delete_nones()
        # print(tree.pptree())
        # assert(tree == abcd)

    print("{} unique trees".format(len(uniquetrees)))


def run_sentence(num_samples=100000):
    sentence = Sentence("le chat rouge jumped on the roof after dark")
    # tokens = "jumped*LS chat <PARENT> on after*LS le*NC <PARENT> rouge*NC*LS <PARENT> roof*LS <PARENT> dark*NC*LS the*NC <PARENT>*LS"
    # tokens = "jumped*LS chat <PARENT> on roof*NC after*LS le*NC <PARENT> rouge*NC*LS <PARENT> the*NC*LS <PARENT> dark*NC*LS"
    # tokens = "dark*LS chat rouge*NC roof after*NC <PARENT>*LS le*NC <PARENT>*LS jumped*NC on*NC the*NC <PARENT>*LS"
    # tokens = "roof*LS rouge jumped on the*NC <PARENT> after*NC dark*NC*LS le*NC chat*NC <PARENT>*LS <PARENT>*LS <PARENT>*LS"
    # tokens = "rouge*LS le*NC chat*NC <PARENT> jumped*NC dark*LS on <PARENT>*LS <PARENT> the*NC roof*NC after*NC*LS"
    # tokens = "rouge*LS chat <PARENT> roof dark*LS le*NC <PARENT>*LS jumped <PARENT>*LS after*NC <PARENT>*LS <PARENT> on*LS <PARENT> the*NC*LS"
    # tokens = "jumped*LS rouge <PARENT> on after dark*NC*LS le*NC chat*NC <PARENT>*LS <PARENT> the*NC*LS roof*NC <PARENT>*LS"
    # tokens = "le*LS <PARENT> rouge jumped roof*LS chat*NC <PARENT>*LS <PARENT> the*LS <PARENT> after*NC dark*NC*LS on*NC <PARENT>*LS"
    tokens = "chat*LS le*NC <PARENT> rouge jumped roof*LS <PARENT>*LS <PARENT> on the*NC*LS <PARENT> after*NC dark*NC*LS <PARENT>*LS"
    # tokens = "dark*LS le*NC roof after*NC <PARENT>*LS rouge on <PARENT>*LS chat*NC <PARENT>"
    tracker = SentenceTracker(sentence)
    for token in tokens.split():
        # print(token)
        nvt = tracker.nxt(token)
        # print(nvt)
    recons = Sentence.parse(tokens)
    # print(recons._gen_tree.pptree())
    assert(recons.x == sentence.x)

    uniquetrees = set()
    for i in range(num_samples):
        tracker.reset()
        nvts = tracker._nvt
        tokens = []
        while len(nvts) > 0:
            token = random.sample(nvts, 1)[0]
            # print(token)
            tokens.append(token)
            tracker.nxt(token)
            nvts = tracker._nvt
            # print(nvts)
        uniquetrees.add(" ".join(tokens))

        try:
            recons = Sentence.parse(" ".join(tokens))
        except Exception as e:
            print(" ".join(tokens))
            raise e
        if i % 1000 == 0:
            print(i)

        # print(recons.x)
        assert(recons.x == sentence.x)
    print("{} unique trees for the sentence after {} samples".format(len(uniquetrees), num_samples))


def run_node(x=0):
    treestr = "A/a*LS B#1/b C#2/c*NC X#3*LS D/d*NC E/e*LS Y*NC Z*NC*LS F/f*NC*LS"
    tree = Node.parse(treestr)
    print(treestr)
    print(tree.pp())
    print(tree.pptree())

    tracker = NodeTracker(tree)
    uniquetrees = set()

    num_samples = 1000
    for i in range(num_samples):
        tracker.reset()
        nvts = tracker._nvt
        tokens = []
        while len(nvts) > 0:
            token = random.sample(nvts, 1)[0]
            # print(token)
            tokens.append(token)
            tracker.nxt(token)
            nvts = tracker._nvt
            # print(nvts)
        # print(" ".join(tokens))
        uniquetrees.add(" ".join(tokens))

        recons = Node.parse(" ".join(tokens))
        if i % 1000 == 0:
            print(i)

        print(recons.pptree())
        # print(tree.pptree())
        assert(recons == tree)

    print("{} unique trees for the sentence after {} samples".format(len(uniquetrees), num_samples))


def test_overcomplete_trees():
    treestr = "BIN*LS BIN1 BIN2*LS UNI UNI*LS LEAF1*NC LEAF2*NC*LS LEAF3*NC*LS LEAF4*NC*LS BIN LEAF <STOP>"
    tree = Node.parse(treestr)
    print(tree.pptree())


def test_treegen():
    trees = generate_random_trees(100)
    for tree in trees:
        print(tree.pptree())

    indic, outdic = trees[0].build_dict_from(trees)

    print(len(indic))
    print(len(outdic))
    print(outdic["N1"], outdic["N1*NC"], outdic["N1*LS"], outdic["N1*NC*LS"])
    print(outdic["N2"], outdic["N2*NC"], outdic["N2*LS"], outdic["N2*NC*LS"])


def headify_tree(x, headtoken="<HEAD>"):
    """ replaces every head with headtoken and pushes down original head as first sibling """
    if x.num_children > 0:  # has children --> headify
        headified_children = [headify_tree(child, headtoken=headtoken) for child in x.children]
        minorder = 0
        for headified_child in headified_children:
            if headified_child.order is not None:
                minorder = min(minorder, headified_child.order)
                headified_child.order = headified_child.order + 1
        assert(minorder == 0)
        x_aschild = Node(x.name, label=x._label, order=0)
        x_replacement = Node(headtoken, order=x._order, children=[x_aschild]+headified_children)
        return x_replacement
    else:
        return x


def unheadify_tree(x, headtoken="<HEAD>"):
    """ unheadifies tree """
    if x.num_children > 0:
        if x.name == headtoken:
            original_head = x.children[0]

            # assert(original_head.order == 0)
            original_children = x.children[1:]
            unheadified_children = [unheadify_tree(child, headtoken=headtoken) for child in original_children]
            minorder = np.infty
            for unheadified_child in unheadified_children:
                if unheadified_child.order is not None:
                    unheadified_child.order -= 1
                    minorder = min(minorder, unheadified_child.order)
            assert(minorder == 0 or minorder == np.infty)
            new_head = Node(original_head.name, label=original_head.label, order=x.order, children=unheadified_children)
            return new_head
        else:
            raise q.SumTingWongException("first child must be head token")
    else:
        return x


def test_headify():
    treestr = "BIN*LS BIN1 BIN2*LS UNI UNI*LS LEAF1*NC LEAF2*NC*LS LEAF3*NC*LS LEAF4*NC*LS BIN LEAF <STOP>"
    originaltree = Node.parse(treestr)
    print(originaltree.pp())
    print(originaltree.pptree())
    headifiedtree = headify_tree(originaltree)
    print(headifiedtree.pp())
    print(headifiedtree.pptree())
    unheadifiedtree = unheadify_tree(headifiedtree)
    unheadifiedoriginaltree = unheadify_tree(originaltree)
    assert(originaltree == unheadifiedtree)
    assert(originaltree == unheadifiedoriginaltree)


if __name__ == "__main__":
    # run_node()
    # test_treegen()
    # test_overcomplete_trees()
    # q.argprun(run_unique_node)
    test_headify()