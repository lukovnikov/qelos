import re, h5py, numpy as np, unidecode
from IPython import embed


def run(p="enwik8", outfp="enwik8.h5", inspect=False, rarefreq=50):
    all = ""
    print "reading file"
    c = 0
    with open(p) as f:
        for line in f:
            line = re.sub("\n", " ", line)      # remove newlines
            line = re.sub("\s+", " ", line)     # collapse multiple consecutive spaces
            all += line
            if c % 1e5 == 0:
                print "{}k".format(c // int(1e3))
            c += 1
    print "file read, {} chars total".format(len(all))
    all = all.decode("utf-8")
    charvec = np.asarray(map(ord, all), dtype="uint16")
    allchars, counts = np.unique(charvec, return_counts=True)
    validchars = set()
    for i in range(len(allchars)):
        if counts[i] > rarefreq:
            validchars.add(allchars[i])
    print "{} unique valid chars".format(len(validchars))
    print "{} rare chars with rare freq {}".format(len(allchars) - len(validchars), rarefreq)
    chartransmat = np.ones((max(allchars)+1,), dtype="uint16")
    c = 2
    chardiclist = [u"<MASK>", u"<RARE>"]
    for k in validchars:
        chartransmat[k] = c
        chardiclist.append(unichr(k))
        c += 1
        assert(len(chardiclist) == c)

    charvec = chartransmat[charvec]

    chardic = dict(zip(range(len(chardiclist)), chardiclist))

    train, valid, test = charvec[:-10000000], charvec[-10000000:-5000000], charvec[-5000000:]
    print "done split"
    #embed()
    with h5py.File(outfp, "w") as outf:
        outf.create_dataset("train", data=train)
        outf.create_dataset("valid", data=valid)
        outf.create_dataset("test", data=test)
        chardiclist = [x.encode("unicode-escape") for x in chardiclist]
        outf.create_dataset("dict", (len(chardiclist), 1), "S10", chardiclist)
    print "pickled and saved: {}".format(outfp)
    if inspect:
        embed()


def loaddata(p="enwik8.h5"):
    with h5py.File(p, "r") as f:
        charlist = [x.decode("unicode-escape") for x in list(f["dict"][:, 0])]
        chardic = dict(zip(range(len(charlist)), charlist))
        train, valid, test = f["train"][:], f["valid"][:], f["test"][:]
    return train, valid, test, chardic


if __name__ == "__main__":
    run()
    #train, valid, test, cd = loaddata()
    #embed()
