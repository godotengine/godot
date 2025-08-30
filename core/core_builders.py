"""Functions used to generate source files during build time"""

import argparse
import os
import sys

try:
    sys.path.insert(0, "./")
    import methods
except ImportError:
    raise SystemExit(f'Generator script "{__file__}" must be run from repository root!')


def disabled_classes(output: str, *classes: str):
    with methods.generated_wrapper(output) as file:
        for c in classes:
            if cs := c.strip():
                file.write(f"#define ClassDB_Disable_{cs} 1\n")


def version_info(output: str, version_path: str, module: str, build: str, status: str):
    values = {}

    if os.path.exists(version_path):
        dirname = os.path.split(os.path.abspath(version_path))[0]
        if dirname:
            sys.path.insert(0, dirname)
        try:
            values["__name__"] = version_path
            with open(version_path) as file:
                contents = file.read()
            exec(contents, {}, values)
        finally:
            if dirname:
                sys.path.remove(dirname)
            del values["__name__"]

    if module:
        values["module"] = module
    if build:
        values["build"] = build
    if status:
        values["status"] = status

    with methods.generated_wrapper(output) as file:
        file.write(f"""\
#define GODOT_VERSION_SHORT_NAME "{values.get("short_name", "")}"
#define GODOT_VERSION_NAME "{values.get("name", "")}"
#define GODOT_VERSION_MAJOR {values.get("major", 0)}
#define GODOT_VERSION_MINOR {values.get("minor", 0)}
#define GODOT_VERSION_PATCH {values.get("patch", 0)}
#define GODOT_VERSION_STATUS "{values.get("status", "")}"
#define GODOT_VERSION_BUILD "{values.get("build", "custom_build")}"
#define GODOT_VERSION_MODULE_CONFIG "{values.get("module_config", "") + values.get("module", "")}"
#define GODOT_VERSION_WEBSITE "{values.get("website", "")}"
#define GODOT_VERSION_DOCS_BRANCH "{values.get("docs", "")}"
#define GODOT_VERSION_DOCS_URL "https://docs.godotengine.org/en/" GODOT_VERSION_DOCS_BRANCH
""")


def git_info(output: str, hash: str, timestamp: int):
    with methods.generated_wrapper(output) as file:
        file.write(f"""\
#include "core/version.h"

const char *const GODOT_VERSION_HASH = "{hash}";
const uint64_t GODOT_VERSION_TIMESTAMP = {timestamp};
""")


def encryption_key(output: str, key: bytes):
    with methods.generated_wrapper(output) as file:
        file.write(f"""\
#include "core/config/project_settings.h"

uint8_t script_encryption_key[32] = {{
	{methods.format_buffer(key, 1)}
}};
""")


def certs(output: str, certs: str, builtin: bool, system_certs: str):
    import zlib

    with methods.generated_wrapper(output) as file:
        # System certs path. Editor will use them if defined. (for package maintainers)
        file.write(f'#define _SYSTEM_CERTS_PATH "{system_certs}"\n')

        if builtin:
            buffer = methods.get_buffer(certs)
            decomp_size = len(buffer)
            buffer = zlib.compress(buffer, zlib.Z_BEST_COMPRESSION)

            # Defined here and not in env so changing it does not trigger a full rebuild.
            file.write(f"""\
#define BUILTIN_CERTS_ENABLED

inline constexpr int _certs_compressed_size = {len(buffer)};
inline constexpr int _certs_uncompressed_size = {decomp_size};
inline constexpr unsigned char _certs_compressed[] = {{
	{methods.format_buffer(buffer, 1)}
}};
""")


def authors(output: str, authors: str):
    SECTIONS = {
        "Project Founders": "AUTHORS_FOUNDERS",
        "Lead Developer": "AUTHORS_LEAD_DEVELOPERS",
        "Project Manager": "AUTHORS_PROJECT_MANAGERS",
        "Developers": "AUTHORS_DEVELOPERS",
    }
    with open(authors, "rb") as authors_file:
        buffer = authors_file.read()
    reading = False

    with methods.generated_wrapper(output) as file:

        def close_section():
            file.write("\tnullptr,\n};\n\n")

        for line in buffer.decode().splitlines():
            if line.startswith("    ") and reading:
                file.write(f'\tR"<!>({line.strip()})<!>",\n')
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


def donors(output: str, donors: str):
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
    buffer = methods.get_buffer(donors)
    reading = False

    with methods.generated_wrapper(output) as file, open(donors):

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


def license(output: str, license: str, copyright: str):
    from collections import OrderedDict
    from io import TextIOWrapper

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

    projects = OrderedDict()  # type: ignore[var-annotated]
    license_list = []

    with open(copyright, "r", encoding="utf-8") as copyright_file:
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

    data_list = []  # type: ignore[var-annotated]
    for project in iter(projects.values()):
        for part in project:
            part["file_index"] = len(data_list)
            data_list += part["Files"]
            part["copyright_index"] = len(data_list)
            data_list += part["Copyright"]

    with open(license, "r", encoding="utf-8") as file_license:
        license_text = file_license.read()

    with methods.generated_wrapper(output) as file:
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
        for item in license_list:
            file.write(f'\t"{methods.to_escaped_cstring(item[0])}",\n')
        file.write("};\n\n")

        file.write("inline constexpr const char *LICENSE_BODIES[] = {\n\n")
        for item in license_list:
            to_raw = []
            for line in item[1:]:
                if line == ".":
                    to_raw += [""]
                else:
                    to_raw += [line]
            file.write(f"{methods.to_raw_cstring(to_raw)},\n\n")
        file.write("};\n\n")


# Command-Line Arguments


def aes256_type(input: str):
    try:
        buffer = bytes.fromhex(input)
        if len(buffer) != 32:
            raise ValueError
        return buffer
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid AES256: {input}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    subparsers = parser.add_subparsers(dest="command", required=True)

    disabled_classes_parser = subparsers.add_parser("disabled_classes")
    disabled_classes_parser.add_argument("classes", nargs="*")

    version_info_parser = subparsers.add_parser("version_info")
    version_info_parser.add_argument("version_path")
    version_info_parser.add_argument("--module", default="")
    version_info_parser.add_argument("--build", default="")
    version_info_parser.add_argument("--status", default="")

    git_info_parser = subparsers.add_parser("git_info")
    git_info_parser.add_argument("--hash", default="")
    git_info_parser.add_argument("--timestamp", type=int, default=0)

    encryption_key_parser = subparsers.add_parser("encryption_key")
    encryption_key_parser.add_argument("--key", type=aes256_type, default="0" * 64)

    certs_parser = subparsers.add_parser("certs")
    certs_parser.add_argument("certs")
    certs_parser.add_argument("builtin", type=bool)
    certs_parser.add_argument("--system", default="")

    authors_parser = subparsers.add_parser("authors")
    authors_parser.add_argument("authors")

    donors_parser = subparsers.add_parser("donors")
    donors_parser.add_argument("donors")

    license_parser = subparsers.add_parser("license")
    license_parser.add_argument("license")
    license_parser.add_argument("copyright")

    args = parser.parse_args()
    if args.command == "disabled_classes":
        disabled_classes(args.output, *args.classes)
    elif args.command == "version_info":
        version_info(args.output, args.version_path, args.module, args.build, args.status)
    elif args.command == "git_info":
        git_info(args.output, args.hash, args.timestamp)
    elif args.command == "encryption_key":
        encryption_key(args.output, args.key)
    elif args.command == "certs":
        certs(args.output, args.certs, args.builtin, args.system)
    elif args.command == "authors":
        authors(args.output, args.authors)
    elif args.command == "donors":
        donors(args.output, args.donors)
    elif args.command == "license":
        license(args.output, args.license, args.copyright)


if __name__ == "__main__":
    raise SystemExit(main())
