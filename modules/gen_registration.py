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

def create_source(ofilename, modules):
    tmpfilename = ofilename + '~'
    with open(tmpfilename, 'w') as ofile:
        ofile.write(header)
        for module in modules:
            ofile.write(f'#include "modules/{module}/register_types.h"\n\n')
        ofile.write('void preregister_module_types() {\n')
        for module in modules:
            modupper = module.upper()
            ofile.write(f'#ifdef MODULE_{modupper}_HAS_PREREGISTER\n')
            ofile.write(f'        preregister_{module}_types();\n')
            ofile.write('#endif\n')
        ofile.write('}\n\n')
        ofile.write('\nvoid register_module_types() {\n')
        for module in modules:
            ofile.write(f'        register_{module}_types();\n')
        ofile.write('}\n')
        ofile.write('\nvoid unregister_module_types() {\n')
        for module in modules:
            ofile.write(f'        unregister_{module}_types();\n')
        ofile.write('}\n')
    replace_if_different(ofilename, tmpfilename)

if __name__ == "__main__":
    ofilename = sys.argv[1]
    modules = sys.argv[2:]
    create_source(ofilename, modules)
