"""Functions used to generate source files during build time"""

import os
import os.path


def make_fonts_header(target, source, env):
    dst = str(target[0])

    with open(dst, "w", encoding="utf-8", newline="\n") as g:
        g.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
        g.write("#ifndef _DEFAULT_FONTS_H\n")
        g.write("#define _DEFAULT_FONTS_H\n")

        # Saving uncompressed, since FreeType will reference from memory pointer.
        for i in range(len(source)):
            file = str(source[i])
            with open(file, "rb") as f:
                buf = f.read()

            name = os.path.splitext(os.path.basename(file))[0]

            g.write("static const int _font_" + name + "_size = " + str(len(buf)) + ";\n")
            g.write("static const unsigned char _font_" + name + "[] = {\n")
            for j in range(len(buf)):
                g.write("\t" + str(buf[j]) + ",\n")

            g.write("};\n")

        g.write("#endif")
