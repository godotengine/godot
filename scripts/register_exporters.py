#!/usr/bin/python3

import argparse


def __make_register_exporters_cpp(platforms: [str], output: str):
    reg_exporters_inc = '#include "editor/register_exporters.h"\n\n'
    reg_exporters = 'void register_exporters() {\n'
    for platform in platforms:
        reg_exporters += '\tregister_' + platform + '_exporter();\n'
        reg_exporters_inc += '#include "platform/' + platform + '/export/export.h"\n'
    reg_exporters_inc += '\n'
    reg_exporters += '}\n'

    # NOTE: It is safe to generate this file here, since this is still executed serially
    with open(output, 'w', encoding='utf-8') as f:
        f.write(reg_exporters_inc)
        f.write(reg_exporters)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate regester_exporters.')

    parser.add_argument(
        'platform', nargs='+', type=str, help='The platforms that have exporters.'
    )
    parser.add_argument(
        'output', type=str, help='The output generated file.'
    )

    args = parser.parse_args()

    __make_register_exporters_cpp(args.platform, args.output)
