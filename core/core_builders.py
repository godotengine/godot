"""Functions used to generate source files during build time"""

from collections import OrderedDict
from io import TextIOWrapper

import methods


# Generate disabled classes
def disabled_class_builder(target, source, env):
    _ = env

    with methods.generated_wrapper(str(target[0])) as file:
        for c in source[0].read():
            if cs := c.strip():
                file.write(f"class {cs}; template <> struct is_class_enabled<{cs}> : std::false_type {{}};\n")


# Generate version info
def version_info_builder(target, source, env):
    _ = env

    with methods.generated_wrapper(str(target[0])) as file:
        file.write("""\
#define GODOT_VERSION_SHORT_NAME "{short_name}"
#define GODOT_VERSION_NAME "{name}"
#define GODOT_VERSION_MAJOR {major}
#define GODOT_VERSION_MINOR {minor}
#define GODOT_VERSION_PATCH {patch}
#define GODOT_VERSION_STATUS "{status}"
#define GODOT_VERSION_BUILD "{build}"
#define GODOT_VERSION_MODULE_CONFIG "{module_config}"
#define GODOT_VERSION_WEBSITE "{website}"
#define GODOT_VERSION_DOCS_BRANCH "{docs_branch}"
#define GODOT_VERSION_DOCS_URL "https://docs.godotengine.org/en/" GODOT_VERSION_DOCS_BRANCH
""".format(**source[0].read()))


def version_hash_builder(target, source, env):
    _ = env

    with methods.generated_wrapper(str(target[0])) as file:
        file.write("""\
#include "core/version.h"

const char *const GODOT_VERSION_HASH = "{git_hash}";
const unsigned long long GODOT_VERSION_TIMESTAMP = {git_timestamp};
""".format(**source[0].read()))


def encryption_key_builder(target, source, env):
    _ = env

    src = source[0].read() or "0" * 64
    try:
        buffer = bytes.fromhex(src)
        if len(buffer) != 32:
            raise ValueError
    except ValueError:
        methods.print_error(
            f'Invalid AES256 encryption key, not 64 hexadecimal characters: "{src}".\n'
            "Unset `SCRIPT_AES256_ENCRYPTION_KEY` in your environment "
            "or make sure that it contains exactly 64 hexadecimal characters."
        )
        raise

    with methods.generated_wrapper(str(target[0])) as file:
        file.write(f"""\
#include <cstdint>

uint8_t script_encryption_key[32] = {{
{methods.format_buffer(buffer, 1)}
}};""")


def make_certs_header(target, source, env):
    _ = env

    buffer = methods.get_buffer(str(source[0]))
    decomp_size = len(buffer)
    buffer = methods.compress_buffer(buffer)

    with methods.generated_wrapper(str(target[0])) as file:
        file.write('#define _SYSTEM_CERTS_PATH "{}"\n'.format(source[2].read() or ""))
        if source[1].read():
            file.write(f"""\
#define BUILTIN_CERTS_ENABLED

inline constexpr int _certs_compressed_size = {len(buffer)};
inline constexpr int _certs_uncompressed_size = {decomp_size};
inline constexpr unsigned char _certs_compressed[] = {{
{methods.format_buffer(buffer, 1)}
}};
""")


def make_authors_or_donors_header(target, source, sections):
    buffer = methods.get_buffer(str(source[0]))
    reading = False

    with methods.generated_wrapper(str(target[0])) as file:

        def close_section():
            file.write("\tnullptr,\n};\n\n")

        for line in buffer.decode().splitlines():

            if line.startswith("    "):
                if reading:
                    file.write(f'\t"{methods.to_escaped_cstring(line).strip()}",\n')

            elif line.startswith("## "):
                if reading:
                    close_section()

                section = sections.get(line[3:].strip())

                if section:
                    file.write(f"inline constexpr const char *{section}[] = {{\n")
                    reading = True
                else:
                    reading = False

        if reading:
            close_section()


def make_authors_header(target, source, env):
    _ = env

    sections = {
        "Project Founders": "AUTHORS_FOUNDERS",
        "Lead Developer": "AUTHORS_LEAD_DEVELOPERS",
        "Project Manager": "AUTHORS_PROJECT_MANAGERS",
        "Developers": "AUTHORS_DEVELOPERS",
    }

    make_authors_or_donors_header(target, source, sections)


def make_donors_header(target, source, env):
    _ = env

    sections = {
        "Patrons": "DONORS_PATRONS",
        "Platinum sponsors": "DONORS_SPONSORS_PLATINUM",
        "Gold sponsors": "DONORS_SPONSORS_GOLD",
        "Silver sponsors": "DONORS_SPONSORS_SILVER",
        "Diamond members": "DONORS_MEMBERS_DIAMOND",
        "Titanium members": "DONORS_MEMBERS_TITANIUM",
        "Platinum members": "DONORS_MEMBERS_PLATINUM",
        "Gold members": "DONORS_MEMBERS_GOLD",
    }

    make_authors_or_donors_header(target, source, sections)


class LicenseReader:
    def __init__(self, license_file: TextIOWrapper):
        self._license_file = license_file
        self.line_num = 0
        self.current = self.next_line()

    def next_line(self):
        while True:
            line = self._license_file.readline()
            self.line_num += 1

            if not line.startswith("#"):
                self.current = line
                return line

    def next_tag(self):
        if ":" not in self.current:
            return "", []

        tag, line = self.current.split(":", 1)
        lines = [line.strip()]
        self.next_line()

        while self.current and self.current.startswith(" "):
            lines.append(self.current.strip())
            self.next_line()

        return tag, lines


def make_license_header(target, source, env):
    _ = env

    src_copyright = str(source[0])
    src_license = str(source[1])

    projects = OrderedDict()
    license_list = []
    valid_tags = {"Files", "Copyright", "License"}

    with open(src_copyright, "r", encoding="utf-8") as copyright_file:
        reader = LicenseReader(copyright_file)
        part = {}
        while reader.current:
            tag, content = reader.next_tag()
            if tag in valid_tags:
                part[tag] = content[:]
            elif tag == "Comment" and part:
                projects.setdefault(content[0], []).append(part)
            if not tag or not reader.current:
                if "License" in part and "Files" not in part:
                    license_list.append(part["License"])
                part = {}
                reader.next_line()

    data_list = []

    all_parts = [part for project in projects.values() for part in project]

    for part in all_parts:
        part["file_index"] = len(data_list)
        data_list += part["Files"]

        part["copyright_index"] = len(data_list)
        data_list += part["Copyright"]

    with open(src_license, "r", encoding="utf-8") as file:
        license_text = file.read()

    with methods.generated_wrapper(str(target[0])) as file:
        file.write(f"""\
inline constexpr const char *GODOT_LICENSE_TEXT = {{
{methods.to_raw_cstring(license_text)}
}};

struct ComponentCopyrightPart {{
	const char *license;
	const char *const *files;
	const char *const *copyright_statements;
	int file_count;
	int copyright_count;
}};

struct ComponentCopyright {{
	const char *name;
	const ComponentCopyrightPart *parts;
	int part_count;
}};

""")

        file.write("inline constexpr const char *COPYRIGHT_INFO_DATA[] = {\n")
        for line in data_list:
            file.write(f'\t"{methods.to_escaped_cstring(line)}",\n')
        file.write("};\n\n")

        file.write("inline constexpr ComponentCopyrightPart COPYRIGHT_PROJECT_PARTS[] = {\n")
        part_index = 0
        part_indexes = {}

        for project_name, project in projects.items():
            part_indexes[project_name] = part_index
            for part in project:
                file.write(
                    f'\t{{ "{methods.to_escaped_cstring(part["License"][0])}", '
                    + f"&COPYRIGHT_INFO_DATA[{part['file_index']}], "
                    + f"&COPYRIGHT_INFO_DATA[{part['copyright_index']}], "
                    + f"{len(part['Files'])}, "
                    + f"{len(part['Copyright'])} }},\n"
                )
                part_index += 1
        file.write("};\n\n")

        file.write(f"inline constexpr int COPYRIGHT_INFO_COUNT = {len(projects)};\n")

        file.write("inline constexpr ComponentCopyright COPYRIGHT_INFO[] = {\n")
        for project_name, project in projects.items():
            file.write(
                f'\t{{ "{methods.to_escaped_cstring(project_name)}", '
                + f"&COPYRIGHT_PROJECT_PARTS[{part_indexes[project_name]}], "
                + f"{len(project)} }},\n"
            )
        file.write("};\n\n")

        file.write(f"inline constexpr int LICENSE_COUNT = {len(license_list)};\n")

        file.write("inline constexpr const char *LICENSE_NAMES[] = {\n")
        for license in license_list:
            file.write(f'\t"{methods.to_escaped_cstring(license[0])}",\n')
        file.write("};\n\n")

        file.write("inline constexpr const char *LICENSE_BODIES[] = {\n\n")
        for license in license_list:
            to_raw = ["" if line == "." else line for line in license[1:]]
            file.write(f"{methods.to_raw_cstring(to_raw)},\n\n")
        file.write("};\n\n")
