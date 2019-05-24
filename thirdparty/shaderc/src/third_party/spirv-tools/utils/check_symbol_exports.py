#!/usr/bin/env python
# Copyright (c) 2017 Google Inc.

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
"""Checks names of global exports from a library."""

from __future__ import print_function

import os.path
import re
import subprocess
import sys


PROG = 'check_symbol_exports'


def command_output(cmd, directory):
    """Runs a command in a directory and returns its standard output stream.

    Captures the standard error stream.

    Raises a RuntimeError if the command fails to launch or otherwise fails.
    """
    p = subprocess.Popen(cmd,
                         cwd=directory,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         universal_newlines=True)
    (stdout, _) = p.communicate()
    if p.returncode != 0:
        raise RuntimeError('Failed to run %s in %s' % (cmd, directory))
    return stdout


def check_library(library):
    """Scans the given library file for global exports.  If all such
    exports are namespaced or begin with spv (in either C or C++ styles)
    then return 0.  Otherwise emit a message and return 1."""

    # The pattern for a global symbol record
    symbol_pattern = re.compile(r'^[0-aA-Fa-f]+ g *F \.text.*[0-9A-Fa-f]+ +(.*)')

    # Ok patterns are as follows, assuming Itanium name mangling:
    #   spv[A-Z]          :  extern "C" symbol starting with spv
    #   _ZN               :  something in a namespace
    #   _Z[0-9]+spv[A-Z_] :  C++ symbol starting with spv[A-Z_]
    symbol_ok_pattern = re.compile(r'^(spv[A-Z]|_ZN|_Z[0-9]+spv[A-Z_])')
    seen = set()
    result = 0
    for line in command_output(['objdump', '-t', library], '.').split('\n'):
        match = symbol_pattern.search(line)
        if match:
            symbol = match.group(1)
            if symbol not in seen:
                seen.add(symbol)
                #print("look at '{}'".format(symbol))
                if not symbol_ok_pattern.match(symbol):
                    print('{}: error: Unescaped exported symbol: {}'.format(PROG, symbol))
                    result = 1
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Check global names exported from a library')
    parser.add_argument('library', help='The static library to examine')
    args = parser.parse_args()

    if not os.path.isfile(args.library):
        print('{}: error: {} does not exist'.format(PROG, args.library))
        sys.exit(1)

    if os.name is 'posix':
        status = check_library(args.library)
        sys.exit(status)
    else:
        print('Passing test since not on Posix')
        sys.exit(0)


if __name__ == '__main__':
    main()
