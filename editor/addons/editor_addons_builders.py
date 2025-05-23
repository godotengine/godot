"""Functions used to generate source files during build time"""

from io import StringIO


def create_make_builtin_addon_header(sources_dir, symbol_prefix):
    def make_builtin_addon_header(target, source, env):
        dst = str(target[0])

        with StringIO() as s:
            s.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
            s.write("#ifndef _EDITOR_ADDONS_%s_H\n" % symbol_prefix)
            s.write("#define _EDITOR_ADDONS_%s_H\n" % symbol_prefix)

            s.write("static const int EDITOR_ADDONS_{}_FILE_COUNT = {};\n".format(symbol_prefix, len(source)))

            s.write("static const char *EDITOR_ADDONS_%s_FILE_PATHS[] = {\n" % symbol_prefix)
            sources_dir_node = env.Dir("#" + sources_dir)
            for path in source:
                fpath = sources_dir_node.rel_path(path).replace("\\", "/")
                s.write('\t"{0}",\n'.format(fpath))
            s.write("};\n")

            lengths = []

            for i in range(len(source)):
                path = source[i]
                with open(str(path), "rb") as file:
                    s.write("static const unsigned char EDITOR_ADDONS_{}_FILE_CONTENT_{}[] = ".format(symbol_prefix, i))
                    s.write("{ \n")
                    buf = file.read()
                    for j in range(len(buf)):
                        s.write("\t" + str(buf[j]) + ",\n")
                    s.write("};\n")
                    lengths.append(len(buf))

            s.write("static const int EDITOR_ADDONS_%s_FILE_LENGTHS[] = {\n" % symbol_prefix)
            for length in lengths:
                s.write("\t" + str(length) + ",\n")
            s.write("};\n")

            s.write("static const unsigned char *EDITOR_ADDONS_%s_FILE_CONTENTS[] = {\n" % symbol_prefix)
            for i in range(len(source)):
                s.write("\tEDITOR_ADDONS_{}_FILE_CONTENT_{},\n".format(symbol_prefix, i))
            s.write("};\n")

            s.write("#endif\n")

            with open(dst, "w", encoding="utf-8", newline="\n") as f:
                f.write(s.getvalue())

    return make_builtin_addon_header
