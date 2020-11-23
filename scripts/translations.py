#!/usr/bin/python3

import argparse
import os.path
import zlib
import glob


def __make_translation_header(input_dir: str, project_root: str, category: str, output: str):

    po_files: [str] = glob.glob(os.path.join(project_root, input_dir, '*.po'))

    with open(output, "w", encoding="utf-8") as g:

        g.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
        g.write("#ifndef _{}_TRANSLATIONS_H\n".format(category.upper()))
        g.write("#define _{}_TRANSLATIONS_H\n".format(category.upper()))

        sorted_paths = sorted(
            po_files, key=lambda path: os.path.splitext(os.path.basename(path))[0])

        xl_names = []
        for i in range(len(sorted_paths)):
            with open(sorted_paths[i], "rb") as f:
                buf = f.read()
            decomp_size = len(buf)
            buf = zlib.compress(buf)
            name = os.path.splitext(os.path.basename(sorted_paths[i]))[0]

            g.write("static const unsigned char _{}_translation_{}_compressed[] = {{\n".format(
                category, name))
            for j in range(len(buf)):
                g.write("\t" + str(buf[j]) + ",\n")

            g.write("};\n")

            xl_names.append([name, len(buf), str(decomp_size)])

        g.write("struct {}TranslationList {{\n".format(category.capitalize()))
        g.write("\tconst char* lang;\n")
        g.write("\tint comp_size;\n")
        g.write("\tint uncomp_size;\n")
        g.write("\tconst unsigned char* data;\n")
        g.write("};\n\n")
        g.write("static {}TranslationList _{}_translations[] = {{\n".format(
            category.capitalize(), category))
        for x in xl_names:
            g.write(
                '\t{{ "{}", {}, {}, _{}_translation_{}_compressed }},\n'.format(
                    x[0], str(x[1]), str(x[2]), category, x[0])
            )
        g.write("\t{nullptr, 0, 0, nullptr}\n")
        g.write("};\n")

        g.write("#endif")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate translation headers.')

    parser.add_argument(
        'input_dir', type=str, help='The input directory of translation files'
    )
    parser.add_argument(
        'project_root', type=str, help='The project root'
    )
    parser.add_argument(
        'category', type=str, help='The doc category'
    )
    parser.add_argument(
        'output', type=str, help='The output header file'
    )

    args = parser.parse_args()

    __make_translation_header(
        args.input_dir, args.project_root, args.category, args.output)
