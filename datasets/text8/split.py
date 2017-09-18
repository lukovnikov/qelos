import re, h5py, numpy as np, unidecode
from IPython import embed


def run(p="text8", outfp="text8.h5", inspect=False):
    all = ""
    print "reading file"
    c = 0
    with open(p) as f:
        for line in f:
            all = line
    print "file read, {} chars total".format(len(all))
    charvec = np.asarray(map(ord, all), dtype="uint8")
    allchars, counts = np.unique(charvec, return_counts=True)
    validchars = allchars
    print "{} unique valid chars".format(len(validchars))
    chartransmat = np.ones((max(allchars)+1,), dtype="uint8")
    c = 2
    chardiclist = ["<MASK>", "<RARE>"]
    for k in validchars:
        chartransmat[k] = c
        chardiclist.append(chr(k))
        c += 1
        assert(len(chardiclist) == c)

    charvec = chartransmat[charvec]

    chardic = dict(zip(range(len(chardiclist)), chardiclist))

    train, valid, test = charvec[:-10000000], charvec[-10000000:-5000000], charvec[-5000000:]
    print "done split"

    with h5py.File(outfp, "w") as outf:
        outf.create_dataset("train", data=train)
        outf.create_dataset("valid", data=valid)
        outf.create_dataset("test", data=test)
        outf.create_dataset("dict", (len(chardiclist), 1), "S10", chardiclist)
    print "pickled and saved: {}".format(outfp)
    if inspect:
        embed()


def loaddata(p="text8.h5"):
    with h5py.File(p, "r") as f:
        charlist = list(f["dict"][:, 0])
        chardic = dict(zip(range(len(charlist)), charlist))
        train, valid, test = f["train"][:], f["valid"][:], f["test"][:]
    return train, valid, test, chardic


if __name__ == "__main__":
    run()
    #train, valid, test, cd = loaddata()
    #embed()
