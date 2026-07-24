"""Functions used to generate source files during build time"""

import argparse
import sys

try:
    sys.path.insert(0, "./")
    import methods
except ImportError:
    raise SystemExit(f'Generator script "{__file__}" must be run from repository root!')


def disabled_classes(target: str, classes: list[str]) -> None:
    with methods.generated_wrapper(target) as file:
        for c in classes:
            if cs := c.strip():
                file.write(f"class {cs}; template <> struct is_class_enabled<{cs}> : std::false_type {{}};\n")


def version_info(
    target: str,
    short_name: str,
    name: str,
    major: int,
    minor: int,
    patch: int,
    status: str,
    build: str,
    module_config: str,
    website: str,
    docs_branch: str,
) -> None:
    with methods.generated_wrapper(target) as file:
        file.write(f"""\
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
""")


def git_info(target: str, hash: str, timestamp: int) -> None:
    with methods.generated_wrapper(target) as file:
        file.write(f"""\
#include "core/version.h"

const char *const GODOT_VERSION_HASH = "{hash}";
const unsigned long long GODOT_VERSION_TIMESTAMP = {timestamp};
""")


def encryption_key(target: str, key: bytes) -> None:
    with methods.generated_wrapper(target) as file:
        file.write(f"""\
#include "core/config/project_settings.h"

uint8_t script_encryption_key[32] = {{
{methods.format_buffer(key, 1)}
}};
""")


def certs(target: str, certs: str, builtin: bool, system_certs: str) -> None:
    import zlib

    with methods.generated_wrapper(target) as file:
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


def authors(target: str, authors: str) -> None:
    SECTIONS = {
        "Project Founders": "AUTHORS_FOUNDERS",
        "Lead Developer": "AUTHORS_LEAD_DEVELOPERS",
        "Project Manager": "AUTHORS_PROJECT_MANAGERS",
        "Developers": "AUTHORS_DEVELOPERS",
    }
    with open(authors, "rb") as authors_file:
        buffer = authors_file.read()
    reading = False

    with methods.generated_wrapper(target) as file:

        def close_section() -> None:
            file.write("\tnullptr,\n};\n\n")

        for line in buffer.decode().splitlines():
            if line.startswith("    ") and reading:
                file.write(f"\t{methods.to_raw_cstring(line.strip())},\n")
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


def donors(target: str, donors: str) -> None:
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

    with methods.generated_wrapper(target) as file, open(donors):

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


def license(target: str, license: str, copyright: str) -> None:
    from typing import TextIO

    class LicenseReader:
        def __init__(self, license_file: TextIO) -> None:
            self._license_file = license_file
            self.line_num = 0
            self.current = self.next_line()

        def next_line(self) -> str:
            line = self._license_file.readline()
            self.line_num += 1
            while line.startswith("#"):
                line = self._license_file.readline()
                self.line_num += 1
            self.current = line
            return line

        def next_tag(self) -> tuple[str, list[str]]:
            if ":" not in self.current:
                return ("", [])
            tag, line = self.current.split(":", 1)
            lines = [line.strip()]
            while self.next_line() and self.current.startswith(" "):
                lines.append(self.current.strip())
            return (tag, lines)

    projects = {}  # type: ignore[var-annotated]
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

    with methods.generated_wrapper(target) as file:
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


def aes256_type(input: str) -> bytes:
    try:
        buffer = bytes.fromhex(input)
        if len(buffer) != 32:
            raise ValueError
        return buffer
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid AES256: {input}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    disabled_classes_parser = subparsers.add_parser("disabled_classes")
    disabled_classes_parser.add_argument("target")
    disabled_classes_parser.add_argument("classes", nargs="*")

    version_info_parser = subparsers.add_parser("version_info")
    version_info_parser.add_argument("target")
    version_info_parser.add_argument("--short_name", default="")
    version_info_parser.add_argument("--name", default="")
    version_info_parser.add_argument("--major", type=int, default=0)
    version_info_parser.add_argument("--minor", type=int, default=0)
    version_info_parser.add_argument("--patch", type=int, default=0)
    version_info_parser.add_argument("--status", default="")
    version_info_parser.add_argument("--build", default="")
    version_info_parser.add_argument("--module_config", default="")
    version_info_parser.add_argument("--website", default="")
    version_info_parser.add_argument("--docs_branch", default="")

    git_info_parser = subparsers.add_parser("git_info")
    git_info_parser.add_argument("target")
    git_info_parser.add_argument("--hash", default="")
    git_info_parser.add_argument("--timestamp", type=int, default=0)

    encryption_key_parser = subparsers.add_parser("encryption_key")
    encryption_key_parser.add_argument("target")
    encryption_key_parser.add_argument("--key", type=aes256_type, default="0" * 64)

    certs_parser = subparsers.add_parser("certs")
    certs_parser.add_argument("target")
    certs_parser.add_argument("certs")
    certs_parser.add_argument("builtin", type=bool)
    certs_parser.add_argument("--system", default="")

    authors_parser = subparsers.add_parser("authors")
    authors_parser.add_argument("target")
    authors_parser.add_argument("authors")

    donors_parser = subparsers.add_parser("donors")
    donors_parser.add_argument("target")
    donors_parser.add_argument("donors")

    license_parser = subparsers.add_parser("license")
    license_parser.add_argument("target")
    license_parser.add_argument("license")
    license_parser.add_argument("copyright")

    args = parser.parse_args()
    if args.command == "disabled_classes":
        disabled_classes(args.target, args.classes)
    elif args.command == "version_info":
        version_info(
            args.target,
            args.short_name,
            args.name,
            args.major,
            args.minor,
            args.patch,
            args.status,
            args.build,
            args.module_config,
            args.website,
            args.docs_branch,
        )
    elif args.command == "git_info":
        git_info(args.target, args.hash, args.timestamp)
    elif args.command == "encryption_key":
        encryption_key(args.target, args.key)
    elif args.command == "certs":
        certs(args.target, args.certs, args.builtin, args.system)
    elif args.command == "authors":
        authors(args.target, args.authors)
    elif args.command == "donors":
        donors(args.target, args.donors)
    elif args.command == "license":
        license(args.target, args.license, args.copyright)


if __name__ == "__main__":
    main()
