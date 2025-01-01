"""Functions used to generate source files during build time"""

import os
from io import StringIO

from methods import to_raw_cstring


# See also `editor/icons/editor_icons_builders.py`.
def make_default_theme_icons_action(target, source, env):
    dst = str(target[0])
    svg_icons = [str(x) for x in source]

    with StringIO() as icons_string, StringIO() as s:
        for svg in svg_icons:
            with open(svg, "r") as svgf:
                icons_string.write("\t%s,\n" % to_raw_cstring(svgf.read()))

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

        with open(dst, "w", encoding="utf-8", newline="\n") as f:
            f.write(s.getvalue())
