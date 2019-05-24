#!/usr/bin/env python
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

from __future__ import print_function

import distutils.dir_util
import os.path
import xml.etree.ElementTree


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

    with open(args.xml) as xml_in:
       registry = xml.etree.ElementTree.fromstring(xml_in.read())

    distutils.dir_util.mkpath(os.path.dirname(args.generator_output))
    print(generate_vendor_table(registry), file=open(args.generator_output, 'w'))


if __name__ == '__main__':
    main()
