import qelos as q
import pickle, numpy as np
from qelos.scripts.webqa.lcquad.buildvnt import REL, ENT, TYP, category


def run(lexp="../../../../datasets/lcquad/lcquad.multilin.lex",
        outp="../../../../datasets/lcquad/lcquad.multilin.lexmats"):
    tt = q.ticktock("lex mat builder")
    tt.tick("loading lex info file")
    info = pickle.load(open(lexp))
    labels = info["labels"]
    alltypes = info["alltypes"]
    besttypes = info["besttypes"]
    verybesttypes = info["verybesttypes"]
    tt.tock("loaded")

    # q.embed()

    tt.tick("building label mats")
    entlabelsm = q.StringMatrix(maxlen=10)
    entlabelsm.tokenize = lambda x: x.lower().split()
    entlabeldic = []
    typlabelsm = q.StringMatrix()
    typlabelsm.tokenize = lambda x: x.split()
    typlabelsm.add("<RARE>")
    typlabeldic = ["<NONE>"]
    rellabelsm = q.StringMatrix()
    rellabelsm.tokenize = lambda x: x.split()
    rellabeldic = []

    for k, v in labels.items():
        if category(k) == REL:
            rellabelsm.add(v)
            rellabeldic.append(k)
        elif category(k) == ENT:
            entlabelsm.add(v[0])
            entlabeldic.append(k)
        else:
            typlabelsm.add(v)
            typlabeldic.append(k)

    entlabelsm.finalize()
    rellabelsm.finalize()
    typlabelsm.finalize()
    entdic = dict(zip(entlabeldic, range(len(entlabeldic))))
    reldic = dict(zip(rellabeldic, range(len(rellabeldic))))
    typdic = dict(zip(typlabeldic, range(len(typlabeldic))))

    tt.tock("label mats built")

    tt.tick("doing type mats for ents")
    verybesttypmat = np.zeros((len(entdic), 1), dtype="int64")
    for k, v in verybesttypes.items():
        verybesttypmat[entdic[k]] = typdic[v]

    maxlen = 3
    besttypmat = np.zeros((len(entdic), maxlen), dtype="int64")
    for k, v in besttypes.items():
        typids = [typdic[ve] for ve in v] + [0] * (3 - len(v))
        besttypmat[entdic[k], :] = typids

    tt.tock("done type mats")

    tt.tick("saving")
    pickle.dump({"entdic": entdic, "reldic": reldic, "typdic": typdic,
                 "entsm": entlabelsm, "relsm": rellabelsm, "typsm": typlabelsm,
                 "verybesttypes": verybesttypmat, "besttypes": besttypmat},
                open(outp, "w"))
    tt.tock("saved").tick("reloading")
    reloaded = pickle.load(open(outp))
    tt.tock("reloaded")

    q.embed()



if __name__ == "__main__":
    q.argprun(run)