"""Functions used to generate source files during build time

All such functions are invoked in a subprocess on Windows to prevent build flakiness.

"""

import os
from io import StringIO
from platform_methods import subprocess_main


# See also `editor/icons/editor_icons_builders.py`.
def make_default_theme_icons_action(target, source, env):
    dst = target[0]
    svg_icons = source

    icons_string = StringIO()

    for f in svg_icons:
        fname = str(f)

        icons_string.write('\t"')

        with open(fname, "rb") as svgf:
            b = svgf.read(1)
            while len(b) == 1:
                icons_string.write("\\" + str(hex(ord(b)))[1:])
                b = svgf.read(1)

        icons_string.write('"')
        if fname != svg_icons[-1]:
            icons_string.write(",")
        icons_string.write("\n")

    s = StringIO()
    s.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n\n")
    s.write('#include "modules/modules_enabled.gen.h"\n\n')
    s.write("#ifndef _DEFAULT_THEME_ICONS_H\n")
    s.write("#define _DEFAULT_THEME_ICONS_H\n")
    s.write("static const int default_theme_icons_count = {};\n\n".format(len(svg_icons)))
    s.write("#ifdef MODULE_SVG_ENABLED\n")
    s.write("static const char *default_theme_icons_sources[] = {\n")
    s.write(icons_string.getvalue())
    s.write("};\n")
    s.write("#endif // MODULE_SVG_ENABLED\n\n")
    s.write("static const char *default_theme_icons_names[] = {\n")

    index = 0
    for f in svg_icons:
        fname = str(f)

        # Trim the `.svg` extension from the string.
        icon_name = os.path.basename(fname)[:-4]

        s.write('\t"{0}"'.format(icon_name))

        if fname != svg_icons[-1]:
            s.write(",")
        s.write("\n")

        index += 1

    s.write("};\n")

    s.write("#endif\n")

    with open(dst, "w") as f:
        f.write(s.getvalue())

    s.close()
    icons_string.close()


if __name__ == "__main__":
    subprocess_main(globals())
