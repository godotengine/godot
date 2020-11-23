#!/usr/bin/python3

import module_db

import argparse
import os
#import glob


def __make_modules_enabled_header(module_db_file: dict, output: str):
    mdb = module_db.load_db(module_db_file)
    modules_enabled: [str] = mdb.get_modules_enabled_names()

    with open(output, "w") as f:
        f.write('#ifndef MODULE_GUARD_DEFINES\n')
        f.write('#define MODULE_GUARD_DEFINES\n\n')
        for module in modules_enabled:
            f.write("#define %s\n" % ("MODULE_" + module.upper() + "_ENABLED"))
        f.write('\n#endif\n')


def __make_register_module_types_cpp(module_db_file: dict, project_root: str, output: str):
    includes_cpp = ""
    preregister_cpp = ""
    register_cpp = ""
    unregister_cpp = ""

    mdb = module_db.load_db(module_db_file)

    for module in mdb.get_modules():
        name = module.name
        path = module.path
        try:
            with open(os.path.join(project_root, path, "register_types.h")):
                includes_cpp += "#ifdef MODULE_" + name.upper() + "_ENABLED\n"
                includes_cpp += '#include "' + path + '/register_types.h"\n'
                includes_cpp += "#endif\n"
                preregister_cpp += "#ifdef MODULE_" + name.upper() + "_ENABLED\n"
                preregister_cpp += "#ifdef MODULE_" + name.upper() + "_HAS_PREREGISTER\n"
                preregister_cpp += "\tpreregister_" + name + "_types();\n"
                preregister_cpp += "#endif\n"
                preregister_cpp += "#endif\n"
                register_cpp += "#ifdef MODULE_" + name.upper() + "_ENABLED\n"
                register_cpp += "\tregister_" + name + "_types();\n"
                register_cpp += "#endif\n"
                unregister_cpp += "#ifdef MODULE_" + name.upper() + "_ENABLED\n"
                unregister_cpp += "\tunregister_" + name + "_types();\n"
                unregister_cpp += "#endif\n"
        except OSError:
            pass

    modules_cpp = """// register_module_types.gen.cpp
/* THIS FILE IS GENERATED DO NOT EDIT */
#include "modules/register_module_types.h"

#include "modules/modules_enabled.gen.h"

%s

void preregister_module_types() {
%s
}

void register_module_types() {
%s
}

void unregister_module_types() {
%s
}
""" % (
        includes_cpp,
        preregister_cpp,
        register_cpp,
        unregister_cpp,
    )

    # NOTE: It is safe to generate this file here, since this is still executed serially
    with open(output, "w") as f:
        f.write(modules_cpp)

    return


# TODO: I dont like this...
# We should register tests into the module db instead of globbing.
# def __make_modules_tests(module_db_file: str, output: str):

#     module_db_data = module_db.load_module_db(module_db_file)

#     with open(output, 'w') as f:
#         for module_data in module_db['modules'].values():
#             if

#         for name, path in env.module_list.items():
#             headers = glob.glob(os.path.join(path, "tests", "*.h"))
#             for h in headers:
#                 f.write('#include "%s"\n' % (os.path.normpath(h)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Module file generators')

    subparsers = parser.add_subparsers(help="sub-command help", dest='command')

    # module enabled
    module_enabled_parser = subparsers.add_parser(
        'modules_enabled', help='Generate the modules_enabled file')
    module_enabled_parser.add_argument(
        'module_db_file', type=str, help='The module db json file.'
    )
    module_enabled_parser.add_argument(
        'output', type=str, help='The output header file.'
    )

    # register module type
    register_module_type_parser = subparsers.add_parser(
        'register_module_types', help='Generate the register_module_types file'
    )
    register_module_type_parser.add_argument(
        'module_db_file', type=str, help='The module db json file.'
    )
    register_module_type_parser.add_argument(
        'project_root', type=str, help='The project source root'
    )
    register_module_type_parser.add_argument(
        'output', type=str, help='The output cpp file.'
    )

    args = parser.parse_args()

    if args.command == 'modules_enabled':
        __make_modules_enabled_header(args.module_db_file, args.output)
    elif args.command == 'register_module_types':
        __make_register_module_types_cpp(
            args.module_db_file, args.project_root, args.output)
