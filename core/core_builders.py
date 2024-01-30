"""Functions used to generate source files during build time

All such functions are invoked in a subprocess on Windows to prevent build flakiness.
"""

from platform_methods import subprocess_main
from compat import iteritems, itervalues, open_utf8, escape_string, byte_to_str


def make_certs_header(target, source, env):

    src = source[0]
    dst = target[0]
    f = open(src, "rb")
    g = open_utf8(dst, "w")
    buf = f.read()
    decomp_size = len(buf)
    import zlib

    buf = zlib.compress(buf)

    g.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
    g.write("#ifndef _CERTS_RAW_H\n")
    g.write("#define _CERTS_RAW_H\n")

    # System certs path. Editor will use them if defined. (for package maintainers)
    path = env["system_certs_path"]
    g.write('#define _SYSTEM_CERTS_PATH "%s"\n' % str(path))
    if env["builtin_certs"]:
        # Defined here and not in env so changing it does not trigger a full rebuild.
        g.write("#define BUILTIN_CERTS_ENABLED\n")
        g.write("static const int _certs_compressed_size = " + str(len(buf)) + ";\n")
        g.write("static const int _certs_uncompressed_size = " + str(decomp_size) + ";\n")
        g.write("static const unsigned char _certs_compressed[] = {\n")
        for i in range(len(buf)):
            g.write("\t" + byte_to_str(buf[i]) + ",\n")
        g.write("};\n")
    g.write("#endif")

    g.close()
    f.close()


def make_authors_header(target, source, env):
    sections = ["Project Founders", "Lead Developer", "Project Manager", "Developers"]
    sections_id = ["AUTHORS_FOUNDERS", "AUTHORS_LEAD_DEVELOPERS", "AUTHORS_PROJECT_MANAGERS", "AUTHORS_DEVELOPERS"]

    src = source[0]
    dst = target[0]
    f = open_utf8(src, "r")
    g = open_utf8(dst, "w")

    g.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
    g.write("#ifndef _EDITOR_AUTHORS_H\n")
    g.write("#define _EDITOR_AUTHORS_H\n")

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
                    g.write("const char *const " + current_section + "[] = {\n")
                    break

    if reading:
        close_section()

    g.write("#endif\n")

    g.close()
    f.close()


def make_donors_header(target, source, env):
    sections = [
        "Patrons",
        "Platinum sponsors",
        "Gold sponsors",
        "Silver sponsors",
        "Diamond members",
        "Titanium members",
        "Platinum members",
        "Gold members",
    ]
    sections_id = [
        "DONORS_PATRONS",
        "DONORS_SPONSORS_PLATINUM",
        "DONORS_SPONSORS_GOLD",
        "DONORS_SPONSORS_SILVER",
        "DONORS_MEMBERS_DIAMOND",
        "DONORS_MEMBERS_TITANIUM",
        "DONORS_MEMBERS_PLATINUM",
        "DONORS_MEMBERS_GOLD",
    ]

    src = source[0]
    dst = target[0]
    f = open_utf8(src, "r")
    g = open_utf8(dst, "w")

    g.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
    g.write("#ifndef _EDITOR_DONORS_H\n")
    g.write("#define _EDITOR_DONORS_H\n")

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
                    g.write("const char *const " + current_section + "[] = {\n")
                    break

    if reading:
        close_section()

    g.write("#endif\n")

    g.close()
    f.close()


def make_license_header(target, source, env):
    src_copyright = source[0]
    src_license = source[1]
    dst = target[0]

    class LicenseReader:
        def __init__(self, license_file):
            self._license_file = license_file
            self.line_num = 0
            self.current = self.next_line()

        def next_line(self):
            line = self._license_file.readline()
            self.line_num += 1
            while line.startswith("#"):
                line = self._license_file.readline()
                self.line_num += 1
            self.current = line
            return line

        def next_tag(self):
            if not ":" in self.current:
                return ("", [])
            tag, line = self.current.split(":", 1)
            lines = [line.strip()]
            while self.next_line() and self.current.startswith(" "):
                lines.append(self.current.strip())
            return (tag, lines)

    from collections import OrderedDict

    projects = OrderedDict()
    license_list = []

    with open_utf8(src_copyright, "r") as copyright_file:
        reader = LicenseReader(copyright_file)
        part = {}
        while reader.current:
            tag, content = reader.next_tag()
            if tag in ("Files", "Copyright", "License"):
                part[tag] = content[:]
            elif tag == "Comment":
                # attach part to named project
                projects[content[0]] = projects.get(content[0], []) + [part]

            if not tag or not reader.current:
                # end of a paragraph start a new part
                if "License" in part and not "Files" in part:
                    # no Files tag in this one, so assume standalone license
                    license_list.append(part["License"])
                part = {}
                reader.next_line()

    data_list = []
    for project in itervalues(projects):
        for part in project:
            part["file_index"] = len(data_list)
            data_list += part["Files"]
            part["copyright_index"] = len(data_list)
            data_list += part["Copyright"]

    with open_utf8(dst, "w") as f:

        f.write("/* THIS FILE IS GENERATED DO NOT EDIT */\n")
        f.write("#ifndef _EDITOR_LICENSE_H\n")
        f.write("#define _EDITOR_LICENSE_H\n")
        f.write("const char *const GODOT_LICENSE_TEXT =")

        with open_utf8(src_license, "r") as license_file:
            for line in license_file:
                escaped_string = escape_string(line.strip())
                f.write('\n\t\t"' + escaped_string + '\\n"')
        f.write(";\n\n")

        f.write(
            "struct ComponentCopyrightPart {\n"
            "\tconst char *license;\n"
            "\tconst char *const *files;\n"
            "\tconst char *const *copyright_statements;\n"
            "\tint file_count;\n"
            "\tint copyright_count;\n"
            "};\n\n"
        )

        f.write(
            "struct ComponentCopyright {\n"
            "\tconst char *name;\n"
            "\tconst ComponentCopyrightPart *parts;\n"
            "\tint part_count;\n"
            "};\n\n"
        )

        f.write("const char *const COPYRIGHT_INFO_DATA[] = {\n")
        for line in data_list:
            f.write('\t"' + escape_string(line) + '",\n')
        f.write("};\n\n")

        f.write("const ComponentCopyrightPart COPYRIGHT_PROJECT_PARTS[] = {\n")
        part_index = 0
        part_indexes = {}
        for project_name, project in iteritems(projects):
            part_indexes[project_name] = part_index
            for part in project:
                f.write(
                    '\t{ "'
                    + escape_string(part["License"][0])
                    + '", '
                    + "&COPYRIGHT_INFO_DATA["
                    + str(part["file_index"])
                    + "], "
                    + "&COPYRIGHT_INFO_DATA["
                    + str(part["copyright_index"])
                    + "], "
                    + str(len(part["Files"]))
                    + ", "
                    + str(len(part["Copyright"]))
                    + " },\n"
                )
                part_index += 1
        f.write("};\n\n")

        f.write("const int COPYRIGHT_INFO_COUNT = " + str(len(projects)) + ";\n")

        f.write("const ComponentCopyright COPYRIGHT_INFO[] = {\n")
        for project_name, project in iteritems(projects):
            f.write(
                '\t{ "'
                + escape_string(project_name)
                + '", '
                + "&COPYRIGHT_PROJECT_PARTS["
                + str(part_indexes[project_name])
                + "], "
                + str(len(project))
                + " },\n"
            )
        f.write("};\n\n")

        f.write("const int LICENSE_COUNT = " + str(len(license_list)) + ";\n")

        f.write("const char *const LICENSE_NAMES[] = {\n")
        for l in license_list:
            f.write('\t"' + escape_string(l[0]) + '",\n')
        f.write("};\n\n")

        f.write("const char *const LICENSE_BODIES[] = {\n\n")
        for l in license_list:
            for line in l[1:]:
                if line == ".":
                    f.write('\t"\\n"\n')
                else:
                    f.write('\t"' + escape_string(line) + '\\n"\n')
            f.write('\t"",\n\n')
        f.write("};\n\n")

        f.write("#endif\n")


if __name__ == "__main__":
    subprocess_main(globals())
