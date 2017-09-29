from __future__ import print_function
import re, lxml
from lxml import etree
from IPython import embed


# LOAD

languages = ["English", "Esperanto"]

verses = [[] for language in languages]
supre = re.compile("<sup[^<]+</sup>([^<]+)")

hist = []
for i, language in enumerate(languages):
    with open(language + ".xml") as f:
        tree = etree.parse(f)
        segments = tree.findall(".//seg")
        for segment in segments:
            if segment.get("type") == "verse":
                _, book, part, verse = segment.get("id").split(".")
                length = 1
                if segment.text is None:
                    text = supre.match(etree.tostring(segment[0])).group(1)
                    length = segment[0].text.strip().split("-")
                    length = int(length[1]) - int(length[0])
                else:
                    text = segment.text.strip()
                verses[i].append((book, int(part), (int(verse), length), text))

print(len(verses[0]), len(verses[1]))


# ALIGN
aligned = []

i = 0
j = 0
lverses = verses[0]
rverses = verses[1]

l, r = "", ""
lc, rc = 0, 0
i = 0
j = 0
previ = 0
prevj = 0
prevlverse = None
prevrverse = None
lacc = ""
racc = ""
while i < len(verses[0]) and j < len(verses[1]):
    lbook, lpart, (lverse, llen), ltext = lverses[i]
    rbook, rpart, (rverse, rlen), rtext = rverses[j]
    if lbook == rbook and lpart == rpart and lverse == rverse:
        l = ltext
        r = rtext
        i += 1
        j += 1
        aligned.append([l, r])
        lacc = ""
        racc = ""
    elif lbook == rbook:
        if lpart > rpart:
            aligned[-1][0] += " " + rtext
            j += 1
        elif lpart < rpart:
            aligned[-1][1] += " " + ltext
            i += 1
        else:
            if lverse > rverse:
                aligned[-1][0] += " " + rtext
                j += 1
            elif lverse < rverse:
                aligned[-1][0] += " " + ltext
                i += 1
    else:
        if lverse < rverse:
            aligned[-1][0] += " " + rtext
            j += 1
        elif lverse > rverse:
            aligned[-1][0] += " " + ltext
            i += 1


outp = "-".join(languages) + ".txt"
with open(outp, "w") as outf:
    for l, r in aligned:
        string = "{}\t{}\n".format(l.encode("utf-8"), r.encode("utf-8"))
        outf.write(string)

