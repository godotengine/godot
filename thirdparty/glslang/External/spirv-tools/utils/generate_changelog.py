#!/usr/bin/env python3

# Copyright (c) 2023 Google Inc.
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

# Args: <CHANGES-file> <tag> <output-file>
# Updates an output file with changelog from the given CHANGES file and tag.
#  - search for first line matching <tag> in file <CHANGES-file>
#  - search for the next line with a tag
#  - writes all the lines in between those 2 tags into <output-file>

import errno
import os
import os.path
import re
import subprocess
import logging
import sys

# Regex to match the SPIR-V version tag.
# Example of matching tags:
#  - v2020.1
#  - v2020.1-dev
#  - v2020.1.rc1
VERSION_REGEX = re.compile(r'^(v\d+\.\d+) +[0-9]+-[0-9]+-[0-9]+$')

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

def main():
    FORMAT = '%(asctime)s %(message)s'
    logging.basicConfig(format="[%(asctime)s][%(levelname)-8s] %(message)s", datefmt="%H:%M:%S")
    if len(sys.argv) != 4:
        logging.error("usage: {} <CHANGES-path> <tag> <output-file>".format(sys.argv[0]))
        sys.exit(1)

    changes_path = sys.argv[1]
    start_tag = sys.argv[2]
    output_file_path = sys.argv[3]

    changelog = []
    has_found_start = False
    with open(changes_path, "r") as file:
      for line in file.readlines():
        m = VERSION_REGEX.match(line)
        if m:
          print(m.groups()[0])
          print(start_tag)
          if has_found_start:
            break;
          if start_tag == m.groups()[0]:
            has_found_start = True
          continue

        if has_found_start:
          changelog.append(line)

    if not has_found_start:
      logging.error("No tag matching {} found.".format(start_tag))
      sys.exit(1)

    content = "".join(changelog)
    if os.path.isfile(output_file_path):
      with open(output_file_path, 'r') as f:
        if content == f.read():
          sys.exit(0)

    mkdir_p(os.path.dirname(output_file_path))
    with open(output_file_path, 'w') as f:
        f.write(content)
    sys.exit(0)

if __name__ == '__main__':
    main()
