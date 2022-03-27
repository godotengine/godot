#!/usr/bin/python3

"""Functions used to generate source files during build time

All such functions are invoked in a subprocess on Windows to prevent build flakiness.

"""
import os, sys



def replace_if_different(output_path_str, new_content_path_str):
    import pathlib

    output_path = pathlib.Path(output_path_str)
    new_content_path = pathlib.Path(new_content_path_str)
    if not output_path.exists():
        new_content_path.replace(output_path)
        return
    if output_path.read_bytes() == new_content_path.read_bytes():
        new_content_path.unlink()
    else:
        new_content_path.replace(output_path)

def make_splash(target, source, env):
    src = source[0]
    dst = target[0]

    tmpfile = dst + '~'
    with open(src, "rb") as f:
        buf = f.read()

    with open(tmpfile, "w") as g:
        g.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
        g.write("#ifndef BOOT_SPLASH_H\n")
        g.write("#define BOOT_SPLASH_H\n")
        # Use a neutral gray color to better fit various kinds of projects.
        g.write("static const Color boot_splash_bg_color = Color(0.14, 0.14, 0.14);\n")
        g.write("static const unsigned char boot_splash_png[] = {\n")
        for i in range(len(buf)):
            g.write(str(buf[i]) + ",\n")
        g.write("};\n")
        g.write("#endif")

    replace_if_different(dst, tmpfile)

def make_splash_editor(target, source, env):
    src = source[0]
    dst = target[0]

    with open(src, "rb") as f:
        buf = f.read()

    with open(dst, "w") as g:
        g.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
        g.write("#ifndef BOOT_SPLASH_EDITOR_H\n")
        g.write("#define BOOT_SPLASH_EDITOR_H\n")
        # The editor splash background color is taken from the default editor theme's background color.
        # This helps achieve a visually "smoother" transition between the splash screen and the editor.
        g.write("static const Color boot_splash_editor_bg_color = Color(0.125, 0.145, 0.192);\n")
        g.write("static const unsigned char boot_splash_editor_png[] = {\n")
        for i in range(len(buf)):
            g.write(str(buf[i]) + ",\n")
        g.write("};\n")
        g.write("#endif")


def make_app_icon(src, dst):
    tmpname = src + '~'
    with open(src, "rb") as f:
        buf = f.read()

    with open(tmpname, "w") as g:
        g.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
        g.write("#ifndef APP_ICON_H\n")
        g.write("#define APP_ICON_H\n")
        g.write("static const unsigned char app_icon_png[] = {\n")
        for i in range(len(buf)):
            g.write(str(buf[i]) + ",\n")
        g.write("};\n")
        g.write("#endif")

    replace_if_different(dst, tmpname)

if __name__ == "__main__":
    command = sys.argv[1]
    if command == 'make_app_icon':
        make_app_icon(sys.argv[2], sys.argv[3])
    elif command == 'make_splash':
        make_splash([sys.argv[2]], [sys.argv[3]], None)
    else:
        sys.exit('Uknonwn command')

