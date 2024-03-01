#!/usr/bin/env python3
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
"""Ensures that all externally visible functions in the library have an appropriate name

Appropriate function names are:
  - names starting with spv,
  - anything in a namespace,
  - functions added by the protobuf compiler,
  - and weak definitions of new and delete."""

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

    # The pattern for an externally visible symbol record
    symbol_pattern = re.compile(r'^[0-aA-Fa-f]+ +([wg]) *F \.text.*[0-9A-Fa-f]+ +(.*)')

    # Ok patterns are as follows, assuming Itanium name mangling:
    #   spv[A-Z]          :  extern "C" symbol starting with spv
    #   _ZN               :  something in a namespace
    #   _ZSt              :  something in the standard namespace
    #   _ZZN              :  something in a local scope and namespace
    #   _Z[0-9]+spv[A-Z_] :  C++ symbol starting with spv[A-Z_]
    symbol_ok_pattern = re.compile(r'^(spv[A-Z]|_ZN|_ZSt|_ZZN|_Z[0-9]+spv[A-Z_])')

    # In addition, the following pattern allowlists global functions that are added
    # by the protobuf compiler:
    #   - AddDescriptors_spvtoolsfuzz_2eproto()
    #   - InitDefaults_spvtoolsfuzz_2eproto()
    symbol_allowlist_pattern = re.compile(r'_Z[0-9]+.*spvtoolsfuzz_2eproto.*')

    symbol_is_new_or_delete = re.compile(r'^(_Zna|_Znw|_Zdl|_Zda)')
    # Compilaion for Arm has various thunks for constructors, destructors, vtables.
    # They are weak.
    symbol_is_thunk = re.compile(r'^_ZT')

    # This occurs in NDK builds.
    symbol_is_hidden = re.compile(r'^\.hidden ')

    seen = set()
    result = 0
    for line in command_output(['objdump', '-t', library], '.').split('\n'):
        match = symbol_pattern.search(line)
        if match:
            linkage = match.group(1)
            symbol = match.group(2)
            if symbol not in seen:
                seen.add(symbol)
                #print("look at '{}'".format(symbol))
                if not (symbol_is_new_or_delete.match(symbol) and linkage == 'w'):
                    if not (symbol_is_thunk.match(symbol) and linkage == 'w'):
                        if not (symbol_allowlist_pattern.match(symbol) or
                                symbol_ok_pattern.match(symbol) or
                                symbol_is_hidden.match(symbol)):
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

    if os.name == 'posix':
        status = check_library(args.library)
        sys.exit(status)
    else:
        print('Passing test since not on Posix')
        sys.exit(0)


if __name__ == '__main__':
    main()
