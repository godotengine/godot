"""Functions used to generate source files during build time"""

from collections import OrderedDict

import methods


def make_default_controller_mappings(target, source, env):
    with methods.generated_wrapper(str(target[0])) as file:
        file.write("""\
#include "core/input/default_controller_mappings.h"

#include "core/typedefs.h"

""")

        PLATFORM_VARIABLES = {
            "Linux": "LINUXBSD",
            "Windows": "WINDOWS",
            "Mac OS X": "MACOS",
            "Android": "ANDROID",
            "iOS": "APPLE_EMBEDDED",
            "Web": "WEB",
        }

        # ensure mappings have a consistent order
        platform_mappings = OrderedDict()
        for src_path in map(str, source):
            with open(src_path, "r", encoding="utf-8") as f:
                mapping_file_lines = f.readlines()

            current_platform = None
            for line in mapping_file_lines:
                if not line:
                    continue
                line = line.strip()
                if len(line) == 0:
                    continue
                if line[0] == "#":
                    platform_or_header = line[1:].strip()
                    if platform_or_header not in PLATFORM_VARIABLES:
                        continue  # Header
                    current_platform = platform_or_header
                    if current_platform not in platform_mappings:
                        platform_mappings[current_platform] = {}
                elif current_platform:
                    line_parts = line.split(",")
                    guid = line_parts[0]
                    if guid in platform_mappings[current_platform]:
                        file.write(
                            "// WARNING: DATABASE {} OVERWROTE PRIOR MAPPING: {} {}\n".format(
                                src_path, current_platform, platform_mappings[current_platform][guid]
                            )
                        )
                    platform_mappings[current_platform][guid] = line

        file.write("const char *DefaultControllerMappings::mappings[] = {\n")
        for platform, mappings in platform_mappings.items():
            variable = PLATFORM_VARIABLES[platform]
            file.write(f"#ifdef {variable}_ENABLED\n")
            for mapping in mappings.values():
                file.write(f'\t"{mapping}",\n')
            file.write(f"#endif // {variable}_ENABLED\n")

        file.write("\tnullptr\n};\n")
