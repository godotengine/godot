"""Functions used to generate source files during build time"""


def make_splash(target, source, env):
    src = str(source[0])
    dst = str(target[0])

    with open(src, "rb") as f:
        buf = f.read()

    with open(dst, "w", encoding="utf-8", newline="\n") as g:
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


def make_splash_editor(target, source, env):
    src = str(source[0])
    dst = str(target[0])

    with open(src, "rb") as f:
        buf = f.read()

    with open(dst, "w", encoding="utf-8", newline="\n") as g:
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


def make_app_icon(target, source, env):
    src = str(source[0])
    dst = str(target[0])

    with open(src, "rb") as f:
        buf = f.read()

    with open(dst, "w", encoding="utf-8", newline="\n") as g:
        g.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
        g.write("#ifndef APP_ICON_H\n")
        g.write("#define APP_ICON_H\n")
        g.write("static const unsigned char app_icon_png[] = {\n")
        for i in range(len(buf)):
            g.write(str(buf[i]) + ",\n")
        g.write("};\n")
        g.write("#endif")
