#!/usr/bin/python3

import argparse


def __make_register_platform_apis(platforms: [str], output: str):
    # Register platform-exclusive APIs
    reg_apis_inc = '#include "platform/register_platform_apis.h"\n'
    reg_apis = "void register_platform_apis() {\n"
    unreg_apis = "void unregister_platform_apis() {\n"
    for platform in platforms:
        reg_apis += "\tregister_" + platform + "_api();\n"
        unreg_apis += "\tunregister_" + platform + "_api();\n"
        reg_apis_inc += '#include "platform/' + platform + '/api/api.h"\n'

    reg_apis_inc += "\n"
    reg_apis += "}\n\n"
    unreg_apis += "}\n"

    with open(output, "w", encoding="utf-8") as f:
        f.write(reg_apis_inc)
        f.write(reg_apis)
        f.write(unreg_apis)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate the platform apis.')

    parser.add_argument('platform', type=str, nargs='+',
                        help='The platforms whose API\'s we must register')
    parser.add_argument('output', type=str, help='The generated api file.')

    args = parser.parse_args()

    __make_register_platform_apis(args.platform, args.output)
