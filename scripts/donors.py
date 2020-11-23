#!/usr/bin/python3

import argparse

# our local utils
from utils import escape_string


def __make_donors_header(input, output):
    sections = [
        "Platinum sponsors",
        "Gold sponsors",
        "Silver sponsors",
        "Bronze sponsors",
        "Mini sponsors",
        "Gold donors",
        "Silver donors",
        "Bronze donors",
    ]
    sections_id = [
        "DONORS_SPONSOR_PLATINUM",
        "DONORS_SPONSOR_GOLD",
        "DONORS_SPONSOR_SILVER",
        "DONORS_SPONSOR_BRONZE",
        "DONORS_SPONSOR_MINI",
        "DONORS_GOLD",
        "DONORS_SILVER",
        "DONORS_BRONZE",
    ]

    f = open(input, "r", encoding="utf-8")
    g = open(output, "w", encoding="utf-8")

    g.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
    g.write("#ifndef DONORS_GEN_H\n")
    g.write("#define DONORS_GEN_H\n")

    reading = False

    def close_section():
        g.write("\t0\n")
        g.write("};\n")

    for line in f:
        if reading >= 0:
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

    g.write("#endif // DONORS_GEN_H\n")

    g.close()
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate the donors header.')
    parser.add_argument('input', type=str)
    parser.add_argument('output', type=str)

    args = parser.parse_args()
    __make_donors_header(args.input, args.output)
