#!/usr/bin/python3

import argparse


def make_splash(input: str, output: str):
    with open(input, "rb") as f:
        buf = f.read()

    with open(output, "w") as g:
        g.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
        g.write("#ifndef BOOT_SPLASH_H\n")
        g.write("#define BOOT_SPLASH_H\n")
        g.write("#include \"core/math/color.h\"\n")
        g.write("static const Color boot_splash_bg_color = Color(0.14, 0.14, 0.14);\n")
        g.write("static const unsigned char boot_splash_png[] = {\n")
        for i in range(len(buf)):
            g.write(str(buf[i]) + ",\n")
        g.write("};\n")
        g.write("#endif")


def make_splash_editor(input: str, output: str):
    with open(input, "rb") as f:
        buf = f.read()

    with open(output, "w") as g:
        g.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
        g.write("#ifndef BOOT_SPLASH_EDITOR_H\n")
        g.write("#define BOOT_SPLASH_EDITOR_H\n")
        g.write("#include \"core/math/color.h\"\n")
        g.write(
            "static const Color boot_splash_editor_bg_color = Color(0.14, 0.14, 0.14);\n")
        g.write("static const unsigned char boot_splash_editor_png[] = {\n")
        for i in range(len(buf)):
            g.write(str(buf[i]) + ",\n")
        g.write("};\n")
        g.write("#endif")


def make_app_icon(input: str, output: str):
    with open(input, "rb") as f:
        buf = f.read()

    with open(output, "w") as g:
        g.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
        g.write("#ifndef APP_ICON_H\n")
        g.write("#define APP_ICON_H\n")
        g.write("#include \"core/math/color.h\"\n")
        g.write("static const unsigned char app_icon_png[] = {\n")
        for i in range(len(buf)):
            g.write(str(buf[i]) + ",\n")
        g.write("};\n")
        g.write("#endif")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spash art compiler.')

    subparsers = parser.add_subparsers(help='sub-command help', dest='command')

    # create the parser for raw glsl
    splash_parser = subparsers.add_parser('splash', help='splash help')
    splash_parser.add_argument(
        'input', type=str, help='The input png file.')
    splash_parser.add_argument(
        'output', type=str, help='The output c++ image binary.')

    splash_editor_parser = subparsers.add_parser(
        'splash_editor', help='splash_editor help')
    splash_editor_parser.add_argument(
        'input', type=str, help='The input png file.')
    splash_editor_parser.add_argument(
        'output', type=str, help='The output c++ image binary.')

    app_icon_parser = subparsers.add_parser(
        'app_icon', help='app_icon help')
    app_icon_parser.add_argument(
        'input', type=str, help='The input png file.')
    app_icon_parser.add_argument(
        'output', type=str, help='The output c++ image binary.')

    args = parser.parse_args()

    if args.command == 'splash':
        make_splash(args.input, args.output)
    elif args.command == 'splash_editor':
        make_splash_editor(args.input, args.output)
    elif args.command == 'app_icon':
        make_app_icon(args.input, args.output)
