#!/usr/bin/env python
# Copyright (c) 2016 Google Inc.
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
"""Checks for copyright notices in all the files that need them under the
current directory.  Optionally insert them.  When inserting, replaces
an MIT or Khronos free use license with Apache 2.
"""
from __future__ import print_function

import argparse
import fileinput
import fnmatch
import inspect
import os
import re
import sys

# List of designated copyright owners.
AUTHORS = ['The Khronos Group Inc.',
           'LunarG Inc.',
           'Google Inc.',
           'Google LLC',
           'Pierre Moreau']
CURRENT_YEAR='2019'

YEARS = '(2014-2016|2015-2016|2016|2016-2017|2017|2018|2019)'
COPYRIGHT_RE = re.compile(
        'Copyright \(c\) {} ({})'.format(YEARS, '|'.join(AUTHORS)))

MIT_BEGIN_RE = re.compile('Permission is hereby granted, '
                          'free of charge, to any person obtaining a')
MIT_END_RE = re.compile('MATERIALS OR THE USE OR OTHER DEALINGS IN '
                        'THE MATERIALS.')
APACHE2_BEGIN_RE = re.compile('Licensed under the Apache License, '
                              'Version 2.0 \(the "License"\);')
APACHE2_END_RE = re.compile('limitations under the License.')

LICENSED = """Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""
LICENSED_LEN = 10 # Number of lines in LICENSED


def find(top, filename_glob, skip_glob_dir_list, skip_glob_files_list):
    """Returns files in the tree rooted at top matching filename_glob but not
    in directories matching skip_glob_dir_list nor files matching
    skip_glob_dir_list."""

    file_list = []
    for path, dirs, files in os.walk(top):
        for glob in skip_glob_dir_list:
            for match in fnmatch.filter(dirs, glob):
                dirs.remove(match)
        for filename in fnmatch.filter(files, filename_glob):
            full_file = os.path.join(path, filename)
            if full_file not in skip_glob_files_list:
                file_list.append(full_file)
    return file_list


def filtered_descendants(glob):
    """Returns glob-matching filenames under the current directory, but skips
    some irrelevant paths."""
    return find('.', glob, ['third_party', 'external', 'CompilerIdCXX',
        'build*', 'out*'], ['./utils/clang-format-diff.py'])


def skip(line):
    """Returns true if line is all whitespace or shebang."""
    stripped = line.lstrip()
    return stripped == '' or stripped.startswith('#!')


def comment(text, prefix):
    """Returns commented-out text.

    Each line of text will be prefixed by prefix and a space character.  Any
    trailing whitespace will be trimmed.
    """
    accum = ['{} {}'.format(prefix, line).rstrip() for line in text.split('\n')]
    return '\n'.join(accum)


def insert_copyright(author, glob, comment_prefix):
    """Finds all glob-matching files under the current directory and inserts the
    copyright message, and license notice.  An MIT license or Khronos free
    use license (modified MIT) is replaced with an Apache 2 license.

    The copyright message goes into the first non-whitespace, non-shebang line
    in a file.  The license notice follows it.  Both are prefixed on each line
    by comment_prefix and a space.
    """

    copyright = comment('Copyright (c) {} {}'.format(CURRENT_YEAR, author),
                        comment_prefix) + '\n\n'
    licensed = comment(LICENSED, comment_prefix) + '\n\n'
    for file in filtered_descendants(glob):
        # Parsing states are:
        #   0 Initial: Have not seen a copyright declaration.
        #   1 Seen a copyright line and no other interesting lines
        #   2 In the middle of an MIT or Khronos free use license
        #   9 Exited any of the above
        state = 0
        update_file = False
        for line in fileinput.input(file, inplace=1):
            emit = True
            if state is 0:
                if COPYRIGHT_RE.search(line):
                    state = 1
                elif skip(line):
                    pass
                else:
                    # Didn't see a copyright. Inject copyright and license.
                    sys.stdout.write(copyright)
                    sys.stdout.write(licensed)
                    # Assume there isn't a previous license notice.
                    state = 1
            elif state is 1:
                if MIT_BEGIN_RE.search(line):
                    state = 2
                    emit = False
                elif APACHE2_BEGIN_RE.search(line):
                    # Assume an Apache license is preceded by a copyright
                    # notice.  So just emit it like the rest of the file.
                    state = 9
            elif state is 2:
                # Replace the MIT license with Apache 2
                emit = False
                if MIT_END_RE.search(line):
                    state = 9
                    sys.stdout.write(licensed)
            if emit:
                sys.stdout.write(line)


def alert_if_no_copyright(glob, comment_prefix):
    """Prints names of all files missing either a copyright or Apache 2 license.

    Finds all glob-matching files under the current directory and checks if they
    contain the copyright message and license notice.  Prints the names of all the
    files that don't meet both criteria.

    Returns the total number of file names printed.
    """
    printed_count = 0
    for file in filtered_descendants(glob):
        has_copyright = False
        has_apache2 = False
        line_num = 0
        apache_expected_end = 0
        with open(file) as contents:
            for line in contents:
                line_num += 1
                if COPYRIGHT_RE.search(line):
                    has_copyright = True
                if APACHE2_BEGIN_RE.search(line):
                    apache_expected_end = line_num + LICENSED_LEN
                if (line_num is apache_expected_end) and APACHE2_END_RE.search(line):
                    has_apache2 = True
        if not (has_copyright and has_apache2):
            message = file
            if not has_copyright:
                message += ' has no copyright'
            if not has_apache2:
                message += ' has no Apache 2 license notice'
            print(message)
            printed_count += 1
    return printed_count


class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__(
                description=inspect.getdoc(sys.modules[__name__]))
        self.add_argument('--update', dest='author', action='store',
                          help='For files missing a copyright notice, insert '
                               'one for the given author, and add a license '
                               'notice.  The author must be in the AUTHORS '
                               'list in the script.')


def main():
    glob_comment_pairs = [('*.h', '//'), ('*.hpp', '//'), ('*.sh', '#'),
                          ('*.py', '#'), ('*.cpp', '//'),
                          ('CMakeLists.txt', '#')]
    argparser = ArgParser()
    args = argparser.parse_args()

    if args.author:
        if args.author not in AUTHORS:
            print('error: --update argument must be in the AUTHORS list in '
                  'check_copyright.py: {}'.format(AUTHORS))
            sys.exit(1)
        for pair in glob_comment_pairs:
            insert_copyright(args.author, *pair)
        sys.exit(0)
    else:
        count = sum([alert_if_no_copyright(*p) for p in glob_comment_pairs])
        sys.exit(count > 0)


if __name__ == '__main__':
    main()
