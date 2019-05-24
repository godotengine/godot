#!/usr/bin/env python
# Copyright 2015 The Shaderc Authors. All rights reserved.
#
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
"""Adds copyright notices to all the files that need them under the
current directory.

usage: add_copyright.py [--check]

With --check, prints out all the files missing the copyright notice and exits
with status 1 if any such files are found, 0 if none.
"""

from __future__ import print_function

import fileinput
import fnmatch
import os
import re
import sys

COPYRIGHTRE = re.compile(
    r'Copyright \d+ The Shaderc Authors. All rights reserved.')
COPYRIGHT = 'Copyright 2016 The Shaderc Authors. All rights reserved.'
LICENSED = """
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""


def find(top, filename_glob, skip_glob_list):
    """Returns files in the tree rooted at top matching filename_glob but not
    in directories matching skip_glob_list."""

    file_list = []
    for path, dirs, files in os.walk(top):
        for glob in skip_glob_list:
            for match in fnmatch.filter(dirs, glob):
                dirs.remove(match)
        for filename in fnmatch.filter(files, filename_glob):
            file_list.append(os.path.join(path, filename))
    return file_list


def filtered_descendants(glob):
    """Returns glob-matching filenames under the current directory, but skips
    some irrelevant paths."""
    return find('.', glob, ['third_party', 'external', 'build*', 'out*',
                            'CompilerIdCXX'])


def skip(line):
    """Returns true if line is all whitespace or shebang."""
    stripped = line.lstrip()
    return stripped == '' or stripped.startswith('#!')


def comment(text, prefix):
    """Returns commented-out text.

    Each line of text will be prefixed by prefix and a space character.  Any
    trailing whitespace will be trimmed.
    """
    accum = []
    for line in text.split('\n'):
        accum.append((prefix + ' ' + line).rstrip())
    return '\n'.join(accum)


def insert_copyright(glob, comment_prefix):
    """Finds all glob-matching files under the current directory and inserts the
    copyright message into them unless they already have it or are empty.

    The copyright message goes into the first non-whitespace, non-shebang line
    in a file.  It is prefixed on each line by comment_prefix and a space.
    """
    copyright = comment(COPYRIGHT, comment_prefix) + '\n'
    licensed = comment(LICENSED, comment_prefix) + '\n\n'
    for file in filtered_descendants(glob):
        has_copyright = False
        for line in fileinput.input(file, inplace=1):
            has_copyright = has_copyright or COPYRIGHTRE.search(line)
            if not has_copyright and not skip(line):
                sys.stdout.write(copyright)
                sys.stdout.write(licensed)
                has_copyright = True
            sys.stdout.write(line)
        if not has_copyright:
            open(file, 'a').write(copyright + licensed)


def alert_if_no_copyright(glob, comment_prefix):
    """Prints names of all files missing a copyright message.

    Finds all glob-matching files under the current directory and checks if they
    contain the copyright message.  Prints the names of all the files that
    don't.

    Returns the total number of file names printed.
    """
    printed_count = 0
    for file in filtered_descendants(glob):
        has_copyright = False
        with open(file) as contents:
            for line in contents:
                if COPYRIGHTRE.search(line):
                    has_copyright = True
                    break
        if not has_copyright:
            print(file, ' has no copyright message.')
            printed_count += 1
    return printed_count


def main():
    glob_comment_pairs = [('*.h', '//'), ('*.hpp', '//'), ('*.cc', '//'),
                          ('*.py', '#'), ('*.cpp', '//')]
    if '--check' in sys.argv:
        count = 0
        for pair in glob_comment_pairs:
            count += alert_if_no_copyright(*pair)
        sys.exit(count > 0)
    else:
        for pair in glob_comment_pairs:
            insert_copyright(*pair)


if __name__ == '__main__':
    main()
