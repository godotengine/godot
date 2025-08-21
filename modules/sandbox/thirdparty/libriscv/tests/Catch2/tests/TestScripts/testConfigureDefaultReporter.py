#!/usr/bin/env python3

#              Copyright Catch2 Authors
# Distributed under the Boost Software License, Version 1.0.
#   (See accompanying file LICENSE.txt or copy at
#        https://www.boost.org/LICENSE_1_0.txt)

# SPDX-License-Identifier: BSL-1.0

from ConfigureTestsCommon import configure_and_build, run_and_return_output

import os
import re
import sys

"""
Tests the CMake configure option for CATCH_CONFIG_DEFAULT_REPORTER

Requires 2 arguments, path folder where the Catch2's main CMakeLists.txt
exists, and path to where the output files should be stored.
"""

if len(sys.argv) != 3:
    print('Wrong number of arguments: {}'.format(len(sys.argv)))
    print('Usage: {} catch2-top-level-dir base-build-output-dir'.format(sys.argv[0]))
    exit(1)

catch2_source_path = os.path.abspath(sys.argv[1])
build_dir_path = os.path.join(os.path.abspath(sys.argv[2]), 'CMakeConfigTests', 'DefaultReporter')

output_file = f"{build_dir_path}/foo.xml"
# We need to escape backslashes in Windows paths, because otherwise they
# are interpreted as escape characters in strings, and cause compilation
# error.
escaped_output_file = output_file.replace('\\', '\\\\')
configure_and_build(catch2_source_path,
                    build_dir_path,
                    [("CATCH_CONFIG_DEFAULT_REPORTER", f"xml::out={escaped_output_file}")])

stdout, _ = run_and_return_output(os.path.join(build_dir_path, 'tests'), 'SelfTest', ['[approx][custom]'])

if not os.path.exists(output_file):
    print(f'Did not find the {output_file} file')
    exit(2)

xml_tag = '</Catch2TestRun>'
with open(output_file, 'r', encoding='utf-8') as file:
    if xml_tag not in file.read():
        print(f"Could not find '{xml_tag}' in the file")
        exit(3)
