"""Functions used to generate source files during build time"""

from pathlib import Path

import methods


def export_icon_builder(target, source, env):
    src_path = Path(str(source[0]))
    src_name = src_path.stem
    platform = src_path.parent.parent.stem

    with open(str(source[0]), "r") as file:
        svg = file.read()

    with methods.generated_wrapper(str(target[0])) as file:
        file.write(
            f"""\
inline constexpr const char *_{platform}_{src_name}_svg = {methods.to_raw_cstring(svg)};
"""
        )


def register_platform_apis_builder(target, source, env):
    platforms = source[0].read()
    api_inc = "\n".join([f'#include "{p}/api/api.h"' for p in platforms])
    api_reg = "\n\t".join([f"register_{p}_api();" for p in platforms])
    api_unreg = "\n\t".join([f"unregister_{p}_api();" for p in platforms])
    with methods.generated_wrapper(str(target[0])) as file:
        file.write(
            f"""\
#include "register_platform_apis.h"

{api_inc}

void register_platform_apis() {{
	{api_reg}
}}

void unregister_platform_apis() {{
	{api_unreg}
}}
"""
        )
