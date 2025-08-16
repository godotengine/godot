#!/usr/bin/env python

import glob
import os
import zlib


def generate_doc_source(dst, source):
    g = open(dst, "w", encoding="utf-8")
    buf = ""
    docbegin = ""
    docend = ""
    for src in source:
        src_path = str(src)
        if not src_path.endswith(".xml"):
            continue
        with open(src_path, "r", encoding="utf-8") as f:
            content = f.read()
        buf += content

    buf = (docbegin + buf + docend).encode("utf-8")
    decomp_size = len(buf)

    # Use maximum zlib compression level to further reduce file size
    # (at the cost of initial build times).
    buf = zlib.compress(buf, zlib.Z_BEST_COMPRESSION)

    g.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
    g.write("\n")
    g.write("#include <godot_cpp/godot.hpp>\n")
    g.write("\n")

    g.write('static const char *_doc_data_hash = "' + str(hash(buf)) + '";\n')
    g.write("static const int _doc_data_uncompressed_size = " + str(decomp_size) + ";\n")
    g.write("static const int _doc_data_compressed_size = " + str(len(buf)) + ";\n")
    g.write("static const unsigned char _doc_data_compressed[] = {\n")
    for i in range(len(buf)):
        g.write("\t" + str(buf[i]) + ",\n")
    g.write("};\n")
    g.write("\n")

    g.write(
        "static godot::internal::DocDataRegistration _doc_data_registration(_doc_data_hash, _doc_data_uncompressed_size, _doc_data_compressed_size, _doc_data_compressed);\n"
    )
    g.write("\n")

    g.close()


def scons_generate_doc_source(target, source, env):
    generate_doc_source(str(target[0]), source)


def generate_doc_source_from_directory(target, directory):
    generate_doc_source(target, glob.glob(os.path.join(directory, "*.xml")))
