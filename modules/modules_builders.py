"""Functions used to generate source files during build time"""

import os

import methods


def modules_enabled_builder(target, source, env):
    modules = sorted(source[0].read())
    with methods.generated_wrapper(str(target[0])) as file:
        for module in modules:
            file.write(f"#define MODULE_{module.upper()}_ENABLED\n")


def register_module_types_builder(target, source, env):
    modules = source[0].read()
    mod_inc = "\n".join([f'#include "{value}/register_types.h"' for value in modules.values()])
    mod_init = "\n".join(
        [
            f"""\
#ifdef MODULE_{key.upper()}_ENABLED
	initialize_{key}_module(p_level);
#endif"""
            for key in modules.keys()
        ]
    )
    mod_uninit = "\n".join(
        [
            f"""\
#ifdef MODULE_{key.upper()}_ENABLED
	uninitialize_{key}_module(p_level);
#endif"""
            for key in modules.keys()
        ]
    )
    with methods.generated_wrapper(str(target[0])) as file:
        file.write(
            f"""\
#include "register_module_types.h"

#include "modules/modules_enabled.gen.h"

{mod_inc}

void initialize_modules(ModuleInitializationLevel p_level) {{
{mod_init}
}}

void uninitialize_modules(ModuleInitializationLevel p_level) {{
{mod_uninit}
}}
"""
        )


def modules_tests_builder(target, source, env):
    headers = sorted([os.path.relpath(src.path, methods.base_folder).replace("\\", "/") for src in source])
    with methods.generated_wrapper(str(target[0])) as file:
        for header in headers:
            file.write(f'#include "{header}"\n')
