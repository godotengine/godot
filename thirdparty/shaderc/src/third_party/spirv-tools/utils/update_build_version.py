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

# Updates an output file with version info unless the new content is the same
# as the existing content.
#
# Args: <spirv-tools_dir> <output-file>
#
# The output file will contain a line of text consisting of two C source syntax
# string literals separated by a comma:
#  - The software version deduced from the CHANGES file in the given directory.
#  - A longer string with the project name, the software version number, and
#    git commit information for the directory.  The commit information
#    is the output of "git describe" if that succeeds, or "git rev-parse HEAD"
#    if that succeeds, or otherwise a message containing the phrase
#    "unknown hash".
# The string contents are escaped as necessary.

from __future__ import print_function

import datetime
import errno
import os
import os.path
import re
import subprocess
import sys
import time


def mkdir_p(directory):
    """Make the directory, and all its ancestors as required.  Any of the
    directories are allowed to already exist."""

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


def command_output(cmd, directory):
    """Runs a command in a directory and returns its standard output stream.

    Captures the standard error stream.

    Raises a RuntimeError if the command fails to launch or otherwise fails.
    """
    p = subprocess.Popen(cmd,
                         cwd=directory,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    (stdout, _) = p.communicate()
    if p.returncode != 0:
        raise RuntimeError('Failed to run %s in %s' % (cmd, directory))
    return stdout


def deduce_software_version(directory):
    """Returns a software version number parsed from the CHANGES file
    in the given directory.

    The CHANGES file describes most recent versions first.
    """

    # Match the first well-formed version-and-date line.
    # Allow trailing whitespace in the checked-out source code has
    # unexpected carriage returns on a linefeed-only system such as
    # Linux.
    pattern = re.compile(r'^(v\d+\.\d+(-dev)?) \d\d\d\d-\d\d-\d\d\s*$')
    changes_file = os.path.join(directory, 'CHANGES')
    with open(changes_file, mode='rU') as f:
        for line in f.readlines():
            match = pattern.match(line)
            if match:
                return match.group(1)
    raise Exception('No version number found in {}'.format(changes_file))


def describe(directory):
    """Returns a string describing the current Git HEAD version as descriptively
    as possible.

    Runs 'git describe', or alternately 'git rev-parse HEAD', in directory.  If
    successful, returns the output; otherwise returns 'unknown hash, <date>'."""
    try:
        # decode() is needed here for Python3 compatibility. In Python2,
        # str and bytes are the same type, but not in Python3.
        # Popen.communicate() returns a bytes instance, which needs to be
        # decoded into text data first in Python3. And this decode() won't
        # hurt Python2.
        return command_output(['git', 'describe'], directory).rstrip().decode()
    except:
        try:
            return command_output(
                ['git', 'rev-parse', 'HEAD'], directory).rstrip().decode()
        except:
            # This is the fallback case where git gives us no information,
            # e.g. because the source tree might not be in a git tree.
            # In this case, usually use a timestamp.  However, to ensure
            # reproducible builds, allow the builder to override the wall
            # clock time with enviornment variable SOURCE_DATE_EPOCH
            # containing a (presumably) fixed timestamp.
            timestamp = int(os.environ.get('SOURCE_DATE_EPOCH', time.time()))
            formatted = datetime.date.fromtimestamp(timestamp).isoformat()
            return 'unknown hash, {}'.format(formatted)


def main():
    if len(sys.argv) != 3:
        print('usage: {} <spirv-tools-dir> <output-file>'.format(sys.argv[0]))
        sys.exit(1)

    output_file = sys.argv[2]
    mkdir_p(os.path.dirname(output_file))

    software_version = deduce_software_version(sys.argv[1])
    new_content = '"{}", "SPIRV-Tools {} {}"\n'.format(
        software_version, software_version,
        describe(sys.argv[1]).replace('"', '\\"'))

    if os.path.isfile(output_file):
        with open(output_file, 'r') as f:
            if new_content == f.read():
                return

    with open(output_file, 'w') as f:
        f.write(new_content)

if __name__ == '__main__':
    main()
