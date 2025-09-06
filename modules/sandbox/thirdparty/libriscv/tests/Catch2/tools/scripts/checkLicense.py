#!/usr/bin/env python3

import sys
import glob

correct_licence = """\

//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
"""

def check_licence_in_file(filename: str) -> bool:
    with open(filename, 'r') as f:
        file_preamble = ''.join(f.readlines()[:7])

    if correct_licence != file_preamble:
        print('File {} does not have proper licence'.format(filename))
        return False
    return True

def check_licences_in_path(path: str) -> int:
    failed = 0
    files_to_check = glob.glob(path + '/**/*.cpp', recursive=True) \
                   + glob.glob(path + '/**/*.hpp', recursive=True)
    for file in files_to_check:
        if not check_licence_in_file(file):
            failed += 1
    return failed

def check_licences():
    failed = 0
    # Add 'extras' after the amalgamted files are regenerated with the new script (past 3.4.0)
    roots = ['src/catch2', 'tests', 'examples', 'fuzzing']
    for root in roots:
        failed += check_licences_in_path(root)
    
    if failed:
        print('{} files are missing licence'.format(failed))
        sys.exit(1)

if __name__ == "__main__":
    check_licences()
