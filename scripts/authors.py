#!/usr/bin/python3

import argparse

# our local utils
from utils import escape_string


def __make_authors_header(input, output):
    sections = [
        "Project Founders",
        "Lead Developer",
        "Project Manager",
        "Developers",
    ]
    sections_id = [
        "AUTHORS_FOUNDERS",
        "AUTHORS_LEAD_DEVELOPERS",
        "AUTHORS_PROJECT_MANAGERS",
        "AUTHORS_DEVELOPERS",
    ]

    f = open(input, "r", encoding="utf-8")
    g = open(output, "w", encoding="utf-8")

    g.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
    g.write("#ifndef AUTHORS_GEN_H\n")
    g.write("#define AUTHORS_GEN_H\n")

    reading = False

    def close_section():
        g.write("\t0\n")
        g.write("};\n")

    for line in f:
        if reading:
            if line.startswith("    "):
                g.write('\t"' + escape_string(line.strip()) + '",\n')
                continue
        if line.startswith("## "):
            if reading:
                close_section()
                reading = False
            for section, section_id in zip(sections, sections_id):
                if line.strip().endswith(section):
                    current_section = escape_string(section_id)
                    reading = True
                    g.write("const char *const " +
                            current_section + "[] = {\n")
                    break

    if reading:
        close_section()

    g.write("#endif // AUTHORS_GEN_H\n")

    g.close()
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate the authors header.')
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)

    args = parser.parse_args()
    __make_authors_header(args.input, args.output)
