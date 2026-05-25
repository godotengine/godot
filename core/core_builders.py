"""Functions used to generate source files during build time"""

import argparse
import os
import sys
from collections import OrderedDict
from io import TextIOWrapper

# Add parent directory to path so we can import methods
sys.path.insert(0, root_directory := os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

import methods


# Generate disabled classes
def disabled_class_builder(target, source, env):
    with methods.generated_wrapper(str(target[0])) as file:
        for c in source[0].read():
            if cs := c.strip():
                file.write(f"class {cs}; template <> struct is_class_enabled<{cs}> : std::false_type {{}};\n")


# Generate version info
def version_info_builder(target, source):
    with methods.generated_wrapper(target) as file:
        file.write(
            f"""\
#define GODOT_VERSION_SHORT_NAME "{source[0]}"
#define GODOT_VERSION_NAME "{source[1]}"
#define GODOT_VERSION_MAJOR {source[2]}
#define GODOT_VERSION_MINOR {source[3]}
#define GODOT_VERSION_PATCH {source[4]}
#define GODOT_VERSION_STATUS "{source[5]}"
#define GODOT_VERSION_BUILD "{source[6]}"
#define GODOT_VERSION_MODULE_CONFIG "{source[7]}"
#define GODOT_VERSION_WEBSITE "{source[8]}"
#define GODOT_VERSION_DOCS_BRANCH "{source[9]}"
#define GODOT_VERSION_DOCS_URL "https://docs.godotengine.org/en/" GODOT_VERSION_DOCS_BRANCH
"""
        )


def version_hash_builder(target, source):
    with methods.generated_wrapper(target) as file:
        file.write(
            f"""\
#include "core/version.h"

const char *const GODOT_VERSION_HASH = "{source[0]}";
const unsigned long long GODOT_VERSION_TIMESTAMP = {source[1]};
"""
        )


def encryption_key_builder(target, source):
    # source[0] is the encryption key, or None
    if source[0] == "None":
        src = "0" * 64
    else:
        src = source[0]
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

    with methods.generated_wrapper(target) as file:
        file.write(
            f"""\
#include <cstdint>

uint8_t script_encryption_key[32] = {{
{methods.format_buffer(buffer, 1)}
}};"""
        )


def make_certs_header(target, source):
    buffer = methods.get_buffer(str(source[0]))
    decomp_size = len(buffer)
    buffer = methods.compress_buffer(buffer)

    with methods.generated_wrapper(target) as file:
        # System certs path. Editor will use them if defined. (for package maintainers)
        file.write('#define _SYSTEM_CERTS_PATH "{}"\n'.format(source[2] or ""))
        if source[1] == "True":
            # Defined here and not in env so changing it does not trigger a full rebuild.
            file.write(f"""\
#define BUILTIN_CERTS_ENABLED

inline constexpr int _certs_compressed_size = {len(buffer)};
inline constexpr int _certs_uncompressed_size = {decomp_size};
inline constexpr unsigned char _certs_compressed[] = {{
{methods.format_buffer(buffer, 1)}
}};
""")


def make_authors_header(target, source):
    SECTIONS = {
        "Project Founders": "AUTHORS_FOUNDERS",
        "Lead Developer": "AUTHORS_LEAD_DEVELOPERS",
        "Project Manager": "AUTHORS_PROJECT_MANAGERS",
        "Developers": "AUTHORS_DEVELOPERS",
    }
    buffer = methods.get_buffer(str(source[0]))
    reading = False

    with methods.generated_wrapper(target) as file:

        def close_section():
            file.write("\tnullptr,\n};\n\n")

        for line in buffer.decode().splitlines():
            if line.startswith("    ") and reading:
                file.write(f'\t"{methods.to_escaped_cstring(line).strip()}",\n')
            elif line.startswith("## "):
                if reading:
                    close_section()
                    reading = False
                section = SECTIONS[line[3:].strip()]
                if section:
                    file.write(f"inline constexpr const char *{section}[] = {{\n")
                    reading = True

        if reading:
            close_section()


def make_donors_header(target, source):
    SECTIONS = {
        "Patrons": "DONORS_PATRONS",
        "Platinum sponsors": "DONORS_SPONSORS_PLATINUM",
        "Gold sponsors": "DONORS_SPONSORS_GOLD",
        "Silver sponsors": "DONORS_SPONSORS_SILVER",
        "Diamond members": "DONORS_MEMBERS_DIAMOND",
        "Titanium members": "DONORS_MEMBERS_TITANIUM",
        "Platinum members": "DONORS_MEMBERS_PLATINUM",
        "Gold members": "DONORS_MEMBERS_GOLD",
    }
    buffer = methods.get_buffer(str(source[0]))
    reading = False

    with methods.generated_wrapper(target) as file:

        def close_section():
            file.write("\tnullptr,\n};\n\n")

        for line in buffer.decode().splitlines():
            if line.startswith("    ") and reading:
                file.write(f'\t"{methods.to_escaped_cstring(line).strip()}",\n')
            elif line.startswith("## "):
                if reading:
                    close_section()
                    reading = False
                section = SECTIONS.get(line[3:].strip())
                if section:
                    file.write(f"inline constexpr const char *{section}[] = {{\n")
                    reading = True

        if reading:
            close_section()


def make_license_header(target, source, env):
    src_copyright = str(source[0])
    src_license = str(source[1])

    class LicenseReader:
        def __init__(self, license_file: TextIOWrapper):
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
            if ":" not in self.current:
                return ("", [])
            tag, line = self.current.split(":", 1)
            lines = [line.strip()]
            while self.next_line() and self.current.startswith(" "):
                lines.append(self.current.strip())
            return (tag, lines)

    projects = OrderedDict()
    license_list = []

    with open(src_copyright, "r", encoding="utf-8") as copyright_file:
        reader = LicenseReader(copyright_file)
        part = {}
        while reader.current:
            tag, content = reader.next_tag()
            if tag in ("Files", "Copyright", "License"):
                part[tag] = content[:]
            elif tag == "Comment" and part:
                # attach non-empty part to named project
                projects[content[0]] = projects.get(content[0], []) + [part]

            if not tag or not reader.current:
                # end of a paragraph start a new part
                if "License" in part and "Files" not in part:
                    # no Files tag in this one, so assume standalone license
                    license_list.append(part["License"])
                part = {}
                reader.next_line()

    data_list = []
    for project in iter(projects.values()):
        for part in project:
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
        for project_name, project in iter(projects.items()):
            part_indexes[project_name] = part_index
            for part in project:
                file.write(
                    f'\t{{ "{methods.to_escaped_cstring(part["License"][0])}", '
                    + f"&COPYRIGHT_INFO_DATA[{part['file_index']}], "
                    + f"&COPYRIGHT_INFO_DATA[{part['copyright_index']}], "
                    + f"{len(part['Files'])}, {len(part['Copyright'])} }},\n"
                )
                part_index += 1
        file.write("};\n\n")

        file.write(f"inline constexpr int COPYRIGHT_INFO_COUNT = {len(projects)};\n")

        file.write("inline constexpr ComponentCopyright COPYRIGHT_INFO[] = {\n")
        for project_name, project in iter(projects.items()):
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
            to_raw = []
            for line in license[1:]:
                if line == ".":
                    to_raw += [""]
                else:
                    to_raw += [line]
            file.write(f"{methods.to_raw_cstring(to_raw)},\n\n")
        file.write("};\n\n")


def main():
    # Parse initial arguments to check for argfile
    initial_parser = argparse.ArgumentParser(add_help=False)
    initial_parser.add_argument("--argfile", help="File containing additional arguments")
    initial_args, remaining_args = initial_parser.parse_known_args()

    # If argfile is provided, read arguments from it
    if initial_args.argfile:
        file_args = methods.read_args_from_file(initial_args.argfile)
        # Combine file arguments with remaining command line arguments
        sys.argv = [sys.argv[0]] + file_args + remaining_args

        # Print arguments to stdout if --verbose is present
        if "--verbose" in sys.argv:
            print("Arguments read from file:", initial_args.argfile)
            print("Combined arguments:", " ".join(file_args + remaining_args))

    # Parse all arguments
    parser = argparse.ArgumentParser(description="Core build tools", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--method",
        required=True,
        choices=[
            "make_authors_header",
            "make_donors_header",
            "encryption_key_builder",
            "make_certs_header",
            "version_info_builder",
            "version_hash_builder",
            "make_license_header",
        ],
        help="""Builder method to execute.
- make_authors_header:      Source: AUTHORS.md
- make_donors_header:       Source: DONORS.md
- encryption_key_builder:   Source: encryption key
- make_certs_header:        Source: ca-bundle.crt, builtin_certs, system_certs_path
- version_info_builder:     Source: short_name, name, major, minor, patch, status, build, module_config, website, docs_branch
- version_hash_builder:     Source: git_hash, git_timestamp
- make_license_header:      Source: COPYRIGHT.txt, LICENSE.txt""",
    )
    parser.add_argument("--target", required=True, help="Target file")
    parser.add_argument("--source", nargs="+", required=True, help="Source file(s)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    target = args.target
    source = args.source

    if args.method == "make_authors_header":
        make_authors_header(target, source)
    elif args.method == "make_donors_header":
        make_donors_header(target, source)
    elif args.method == "encryption_key_builder":
        encryption_key_builder(target, source)
    elif args.method == "make_certs_header":
        make_certs_header(target, source)
    elif args.method == "version_info_builder":
        version_info_builder(target, source)
    elif args.method == "version_hash_builder":
        version_hash_builder(target, source)
    else:
        print(f"Unknown method: {args.method}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
