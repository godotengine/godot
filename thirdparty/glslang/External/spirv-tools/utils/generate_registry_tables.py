#!/usr/bin/env python3
# Copyright (c) 2016 Google Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generates the vendor tool table from the SPIR-V XML registry."""

import errno
import io
import os.path
from xml.etree.ElementTree import XML, XMLParser, TreeBuilder


def mkdir_p(directory):
    """Make the directory, and all its ancestors as required.  Any of the
    directories are allowed to already exist.
    This is compatible with Python down to 3.0.
    """

    if directory == "":
        # We're being asked to make the current directory.
        return

    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(directory):
            pass
        else:
            raise


def generate_vendor_table(registry):
    """Returns a list of C style initializers for the registered vendors
    and their tools.

    Args:
      registry: The SPIR-V XMLregistry as an xml.ElementTree
    """

    lines = []
    for ids in registry.iter('ids'):
        if 'vendor' == ids.attrib['type']:
            for an_id in ids.iter('id'):
                value = an_id.attrib['value']
                vendor = an_id.attrib['vendor']
                if 'tool' in an_id.attrib:
                    tool = an_id.attrib['tool']
                    vendor_tool = vendor + ' ' + tool
                else:
                    tool = ''
                    vendor_tool = vendor
                line = '{' + '{}, "{}", "{}", "{}"'.format(value,
                                                           vendor,
                                                           tool,
                                                           vendor_tool) + '},'
                lines.append(line)
    return '\n'.join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=
                                     'Generate tables from SPIR-V XML registry')
    parser.add_argument('--xml', metavar='<path>',
                        type=str, required=True,
                        help='SPIR-V XML Registry file')
    parser.add_argument('--generator-output', metavar='<path>',
                        type=str, required=True,
                        help='output file for SPIR-V generators table')
    args = parser.parse_args()

    with io.open(args.xml, encoding='utf-8') as xml_in:
      parser = XMLParser(target=TreeBuilder(), encoding='utf-8')
      registry = XML(xml_in.read(), parser=parser)

    mkdir_p(os.path.dirname(args.generator_output))
    with open(args.generator_output, 'w') as f:
      f.write(generate_vendor_table(registry))


if __name__ == '__main__':
    main()
