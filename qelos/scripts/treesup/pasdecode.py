import torch
import qelos as q
from qelos.scripts.treesup.pastrees import GroupTracker, generate_random_trees, Tree, BinaryTree, UnaryTree, LeafTree
import numpy as np
import re


OPT_LR = 0.1
OPT_INPEMBDIM = 50
OPT_OUTEMBDIM = 50
OPT_LINOUTDIM = 50
OPT_JOINT_LINOUT_MODE = "sum"
_opt_test = True


def run(lr=OPT_LR,
        inpembdim=OPT_INPEMBDIM,
        outembdim=OPT_OUTEMBDIM,
        linoutdim=OPT_LINOUTDIM,
        linoutjoinmode=OPT_JOINT_LINOUT_MODE,
        ):
    tt = q.ticktock("script")
    # region load data
    ism = q.StringMatrix()
    ism.tokenize = lambda x: x.split()
    numtrees = 1000
    trees = generate_random_trees(numtrees)
    tracker = GroupTracker(trees)
    for tree in trees:
        treestring = tree.pp(with_parentheses=False, arbitrary=True)
        ism.add(treestring)
    ism.finalize()      # ism provides source sequences
    eids = np.arange(0, len(trees)).astype("int64")

    if _opt_test:       # TEST
        for eid in eids:
            assert(tracker.trackers[eid].root == Tree.parse(ism[eid]))
    # endregion

    # region build reps
    inpemb = q.WordEmb(inpembdim, worddic=ism.D)
    outemb = q.WordEmb(outembdim, worddic=tracker.D_in)

    # computed wordlinout for output symbols with topology annotations
    # dictionaries
    specialsymbols = []
    symbols = []
    symbols_core = []        # unique cores of output symbols for linout
    symbols_ls = []          # set of output linout symbols that are annotated as last sibling
    for k, v in tracker.D.items():
        if re.match("<[^>]+>", k):
            specialsymbols.append(k)
        else:
            symbols.append(k)
            ksplits = k.split("*")
            symbols_core.append(ksplits[0])
            symbols_ls.append("LS" in ksplits[1:])
    specialsymbols_dic = dict(zip(specialsymbols, range(len(specialsymbols))))
    symbols_dic = dict(zip(symbols, range(len(symbols))))
    symbols_core_dic = dict(zip(list(set(symbols_core)), range(len(set(symbols_core)))))
    symbols2cores = [0] * len(symbols_dic)
    for symbol, symbol_core in zip(symbols, symbols_core):
        symbols2cores[symbols_dic[symbol]] = symbols_core_dic[symbol_core]

    symbols2cores = np.asarray(symbols2cores, dtype="int64")
    symbols_ls = np.asarray(symbols_ls, dtype="int64")

    # linouts
    baselinout = q.ZeroWordLinout(linoutdim, worddic=tracker.D)
    specialsymbols_linout = q.WordLinout(linoutdim, worddic=specialsymbols_dic)

    linoutdim_core, linoutdim_ls = linoutdim, linoutdim

    if linoutjoinmode == "cat":
        linoutdim_core = linoutdim // 10 * 9
    symbols_core_emb = q.WordEmb(linoutdim_core, worddic=symbols_core_dic)

    class JointLinoutComputer(torch.nn.Module):
        def __init__(self, coreemb, linoutdim, mode="sum", **kw):
            super(JointLinoutComputer, self).__init__(**kw)
            self.coreemb, self.mode = coreemb, mode
            lsembdim = linoutdim if mode != "cat" else linoutdim // 10
            self.lsemb = q.WordEmb(lsembdim, worddic={"NOLS": 0, "LS": 1})
            if mode in "muladd".split():
                self.lsemb_aux = q.WordEmb(lsembdim, worddic=self.lsemb.D)
            elif mode in "gate".split():
                self.lsemb_aux = q.Forward(lsembdim, lsembdim, activation="sigmoid", use_bias=False)
            elif mode in "bilinear linear".split():
                dim2 = lsembdim if mode in "bilinear".split() else 2 if mode in "linear".split() else None
                self.lsemb_aux = torch.nn.Bilinear(lsembdim, dim2, lsembdim, bias=False)

        def forward(self, data):
            coreembs, _ = self.coreemb(data[:, 0])
            lsembs, _ = self.lsemb(data[:, 1])
            embs = None
            if self.mode in "sum".split():
                embs = coreembs + lsembs
            elif self.mode in "mul flatgate".split():
                if self.mode in "flatgate".split():
                    lsembs = torch.nn.Sigmoid()(lsembs)
                embs = coreembs * lsembs
            elif self.mode in "cat".split():
                embs = torch.cat([coreembs, lsembs], 1)
            elif self.mode in "muladd".split():
                lsembs_aux = self.lsemb_aux(data[:, 1])
                embs = coreembs * lsembs
                embs = embs * lsembs_aux
            elif self.mode in "gate".split():
                gate = self.lsemb_aux(lsembs)
                embs = coreembs * gate
            elif self.mode in "bilinear linear".split():
                if self.mode in "linear".split():
                    lsembs = torch.stack([1 - data[:, 1], data[:, 1]], 1)
                embs = self.lsemb_aux(coreembs, lsembs)
            return embs

    symbols_linout_computer = JointLinoutComputer(symbols_core_emb, linoutdim, mode=linoutjoinmode)
    symbols_linout_data = np.stack([symbols2cores, symbols_ls], axis=1)
    symbols_linout = q.ComputedWordLinout(data=symbols_linout_data,
                                          computer=symbols_linout_computer,
                                          worddic=symbols_dic)

    linout = baselinout.override(specialsymbols_linout).override(symbols_linout)

    if _opt_test:       # TEST
        tt.tick("testing linouts")
        symbols_linout_computer.lsemb.embedding.weight.data.fill_(0)
        testvecs = torch.autograd.Variable(torch.randn(3, linoutdim))
        test_linout_output = linout(testvecs).data.numpy()
        assert(np.allclose(test_linout_output[:, 4:44], test_linout_output[:, 44:]))
        symbols_linout_computer.mode = "mul"
        test_linout_output = linout(testvecs).data.numpy()
        assert(np.allclose(test_linout_output[:, 4:], np.zeros_like(test_linout_output[:, 4:])))
        tt.tock("[success]")
    # endregion


if __name__ == "__main__":
    q.argprun(run)