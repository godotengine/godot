#!/usr/bin/python3

import argparse
import os
import zlib


def __make_certs_header(input: str, output: str, system_certs_path: str, builtin_certs: bool):
    f = open(input, "rb")
    g = open(output, "w", encoding="utf-8")
    buf = f.read()
    decomp_size = len(buf)

    buf = zlib.compress(buf)

    g.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
    g.write("#ifndef CERTS_COMPRESSED_GEN_H\n")
    g.write("#define CERTS_COMPRESSED_GEN_H\n")

    # System certs path. Editor will use them if defined. (for package maintainers)
    g.write('#define _SYSTEM_CERTS_PATH "%s"\n' % str(system_certs_path))
    if builtin_certs:
        # Defined here and not in env so changing it does not trigger a full rebuild.
        g.write("#define BUILTIN_CERTS_ENABLED\n")
        g.write("static const int _certs_compressed_size = " +
                str(len(buf)) + ";\n")
        g.write("static const int _certs_uncompressed_size = " +
                str(decomp_size) + ";\n")
        g.write("static const unsigned char _certs_compressed[] = {\n")
        for i in range(len(buf)):
            g.write("\t" + str(buf[i]) + ",\n")
        g.write("};\n")
    g.write("#endif // CERTS_COMPRESSED_GEN_H")

    g.close()
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate the certificate header.')
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('system_certs_path', type=str)
    parser.add_argument('builtin_certs', type=bool)

    args = parser.parse_args()
    __make_certs_header(args.input, args.output,
                        args.system_certs_path, args.builtin_certs)
