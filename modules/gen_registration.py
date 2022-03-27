#!/usr/bin/env python3

import sys, os

header = '''// register_module_types.gen.cpp
/* THIS FILE IS GENERATED DO NOT EDIT */
#include "modules/register_module_types.h"

#include "modules/modules_enabled.gen.h"

'''

def replace_if_different(output_path_str, new_content_path_str):
    import pathlib

    output_path = pathlib.Path(output_path_str)
    new_content_path = pathlib.Path(new_content_path_str)
    if not output_path.exists():
        new_content_path.replace(output_path)
        return
    if output_path.read_bytes() == new_content_path.read_bytes():
        new_content_path.unlink()
    else:
        new_content_path.replace(output_path)

def create_register_source(ofilename, modules):
    tmpfilename = ofilename + '~'
    with open(tmpfilename, 'w') as ofile:
        ofile.write(header)
        for module in modules:
            ofile.write(f'#include "modules/{module}/register_types.h"\n\n')
        ofile.write('\nvoid initialize_modules(ModuleInitializationLevel p_level) {\n')
        for module in modules:
            ofile.write(f'        initialize_{module}_module(p_level);\n')
        ofile.write('}\n')
        ofile.write('\nvoid uninitialize_modules(ModuleInitializationLevel p_level) {\n')
        for module in modules:
            ofile.write(f'        uninitialize_{module}_module(p_level);\n')
        ofile.write('}\n')
    replace_if_different(ofilename, tmpfilename)

def create_enabled_header(ofilename, modules):
    tmpfilename = ofilename + '~'
    with open(tmpfilename, 'w') as ofile:
        ofile.write('#pragma once\n\n')
        for m in modules:
            mup = m.upper()
            ofile.write(f'#define MODULE_{mup}_ENABLED\n')
    replace_if_different(ofilename, tmpfilename)

if __name__ == "__main__":
    registerfilename = sys.argv[1]
    modules = sys.argv[2:]
    create_register_source(registerfilename, modules)
    enabledfilename = os.path.join(os.path.split(registerfilename)[0], 'modules_enabled.gen.h')
    create_enabled_header(enabledfilename, modules)
