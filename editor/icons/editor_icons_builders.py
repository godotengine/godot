"""Functions used to generate source files during build time"""

import os
from io import StringIO


# See also `scene/theme/icons/default_theme_icons_builders.py`.
def make_editor_icons_action(target, source, env):
    dst = str(target[0])
    svg_icons = source

    with StringIO() as icons_string, StringIO() as s:
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

        s.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
        s.write("#ifndef _EDITOR_ICONS_H\n")
        s.write("#define _EDITOR_ICONS_H\n")
        s.write("static const int editor_icons_count = {};\n".format(len(svg_icons)))
        s.write("static const char *editor_icons_sources[] = {\n")
        s.write(icons_string.getvalue())
        s.write("};\n\n")
        s.write("static const char *editor_icons_names[] = {\n")

        # this is used to store the indices of thumbnail icons
        thumb_medium_indices = []
        thumb_big_indices = []
        index = 0
        for f in svg_icons:
            fname = str(f)

            # Trim the `.svg` extension from the string.
            icon_name = os.path.basename(fname)[:-4]
            # some special cases
            if icon_name.endswith("MediumThumb"):  # don't know a better way to handle this
                thumb_medium_indices.append(str(index))
            if icon_name.endswith("BigThumb"):  # don't know a better way to handle this
                thumb_big_indices.append(str(index))
            if icon_name.endswith("GodotFile"):  # don't know a better way to handle this
                thumb_big_indices.append(str(index))

            s.write('\t"{0}"'.format(icon_name))

            if fname != svg_icons[-1]:
                s.write(",")
            s.write("\n")

            index += 1

        s.write("};\n")

        if thumb_medium_indices:
            s.write("\n\n")
            s.write("static const int editor_md_thumbs_count = {};\n".format(len(thumb_medium_indices)))
            s.write("static const int editor_md_thumbs_indices[] = {")
            s.write(", ".join(thumb_medium_indices))
            s.write("};\n")
        if thumb_big_indices:
            s.write("\n\n")
            s.write("static const int editor_bg_thumbs_count = {};\n".format(len(thumb_big_indices)))
            s.write("static const int editor_bg_thumbs_indices[] = {")
            s.write(", ".join(thumb_big_indices))
            s.write("};\n")

        s.write("#endif\n")

        with open(dst, "w", encoding="utf-8", newline="\n") as f:
            f.write(s.getvalue())
