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
Tests the CMake configure option for CATCH_CONFIG_EXPERIMENTAL_REDIRECT

Requires 2 arguments, path folder where the Catch2's main CMakeLists.txt
exists, and path to where the output files should be stored.
"""

if len(sys.argv) != 3:
    print('Wrong number of arguments: {}'.format(len(sys.argv)))
    print('Usage: {} catch2-top-level-dir base-build-output-dir'.format(sys.argv[0]))
    exit(1)

catch2_source_path = os.path.abspath(sys.argv[1])
build_dir_path = os.path.join(os.path.abspath(sys.argv[2]), 'CMakeConfigTests', 'ExperimentalRedirect')

configure_and_build(catch2_source_path,
                    build_dir_path,
                    [("CATCH_CONFIG_EXPERIMENTAL_REDIRECT", "ON")])

stdout, _ = run_and_return_output(os.path.join(build_dir_path, 'tests'),
                                  'SelfTest',
                                  ['-r', 'xml', '"has printf"'])


# The print from printf must be within the XML's reporter stdout tag.
required_output = '''\
      <StdOut>
loose text artifact
      </StdOut>
'''
if not required_output in stdout:
    print("Could not find '{}' in the stdout".format(required_output))
    print('stdout: "{}"'.format(stdout))
    exit(2)
