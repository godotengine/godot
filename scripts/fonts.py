#!/usr/bin/python3

import argparse
import os.path
import glob


def __make_fonts_header(input_dir: str, project_root: str, output: str):

    font_files: [str] = glob.glob(
        os.path.join(project_root, input_dir, '*.ttf'))
    font_files += glob.glob(
        os.path.join(project_root, input_dir, '*.otf'))

    with open(output, "w", encoding="utf-8") as g:

        g.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
        g.write("#ifndef _EDITOR_FONTS_H\n")
        g.write("#define _EDITOR_FONTS_H\n")

        # saving uncompressed, since freetype will reference from memory pointer
        for font_file in font_files:
            buf: [bytes] = []
            with open(font_file, "rb") as f:
                buf = f.read()

            name = os.path.splitext(os.path.basename(font_file))[0]

            g.write("static const int _font_" + name +
                    "_size = " + str(len(buf)) + ";\n")
            g.write("static const unsigned char _font_" + name + "[] = {\n")
            for b in buf:
                g.write("\t" + str(b) + ",\n")

            g.write("};\n")

        g.write("#endif")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate the font header.')

    parser.add_argument(
        'input_dir', type=str, help='The input directory of font files'
    )
    parser.add_argument(
        'project_root', type=str, help='The project root'
    )
    parser.add_argument(
        'output', type=str, help='The output font file'
    )

    args = parser.parse_args()

    __make_fonts_header(args.input_dir, args.project_root, args.output)
