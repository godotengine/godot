

import sys

if (len(sys.argv) != 2):
    print("Pass me a .fnt argument!")

f = open(sys.argv[1], "rb")

name = sys.argv[1].lower().replace(".fnt", "")

l = f.readline()

font_height = 0
font_ascent = 0
font_charcount = 0
font_chars = []
font_cc = 0

while(l != ""):

    fs = l.strip().find(" ")
    if (fs == -1):
        l = f.readline()
        continue
    t = l[0:fs]

    dv = l[fs + 1:].split(" ")
    d = {}
    for x in dv:
        if (x.find("=") == -1):
            continue
        s = x.split("=")
        d[s[0]] = s[1]

    if (t == "common"):
        font_height = d["lineHeight"]
        font_ascent = d["base"]

    if (t == "char"):
        font_chars.append(d["id"])
        font_chars.append(d["x"])
        font_chars.append(d["y"])
        font_chars.append(d["width"])
        font_chars.append(d["height"])
        font_chars.append(d["xoffset"])
        font_chars.append(d["yoffset"])
        font_chars.append(d["xadvance"])
        font_cc += 1

    l = f.readline()


print("static const int _bi_font_" + name + "_height=" + str(font_height) + ";")
print("static const int _bi_font_" + name + "_ascent=" + str(font_ascent) + ";")
print("static const int _bi_font_" + name + "_charcount=" + str(font_cc) + ";")
cstr = "static const int _bi_font_" + name + "_characters={"
for i in range(len(font_chars)):

    c = font_chars[i]
    if (i > 0):
        cstr += ", "
    cstr += c

cstr += ("};")

print(cstr)
