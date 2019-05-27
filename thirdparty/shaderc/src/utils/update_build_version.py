#!/usr/bin/env python

# Copyright 2016 The Shaderc Authors. All rights reserved.
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

# Updates build-version.inc in the current directory, unless the update is
# identical to the existing content.
#
# Args: <shaderc-dir> <spirv-tools-dir> <glslang-dir>
#
# For each directory, there will be a line in build-version.inc containing that
# directory's "git describe" output enclosed in double quotes and appropriately
# escaped.

from __future__ import print_function

import datetime
import os.path
import re
import subprocess
import sys
import time

OUTFILE = 'build-version.inc'


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
        raise RuntimeError('Failed to run {} in {}'.format(cmd, directory))
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
    with open(changes_file) as f:
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


def get_version_string(project, directory):
    """Returns a detailed version string for a given project with its directory,
    which consists of software version string and git description string."""
    detailed_version_string_lst = [project]
    if project != 'glslang':
        detailed_version_string_lst.append(deduce_software_version(directory))
    detailed_version_string_lst.append(describe(directory).replace('"', '\\"'))
    return ' '.join(detailed_version_string_lst)


def main():
    if len(sys.argv) != 4:
        print('usage: {} <shaderc-dir> <spirv-tools-dir> <glslang-dir>'.format(
            sys.argv[0]))
        sys.exit(1)

    projects = ['shaderc', 'spirv-tools', 'glslang']
    new_content = ''.join([
        '"{}\\n"\n'.format(get_version_string(p, d))
        for (p, d) in zip(projects, sys.argv[1:])
    ])

    if os.path.isfile(OUTFILE):
        with open(OUTFILE, 'r') as f:
            if new_content == f.read():
                return
    with open(OUTFILE, 'w') as f:
        f.write(new_content)


if __name__ == '__main__':
    main()
