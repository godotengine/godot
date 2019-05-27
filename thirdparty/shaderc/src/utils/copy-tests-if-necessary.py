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

"""Copies tests from the source directory to the binary directory if something
in the source directory has changed. It also updates the path in runtests
to point to the correct binary directory.

Arguments: glslang_test_source_dir glslang_test_bin_dir [intermediate-dir]

intermediate-dir is optional, it specifies that there the additional directory
between the root, and the tests/binary.
"""

import errno
import os
import shutil
import sys


def get_modified_times(path):
    """Returns a string containing a newline-separated set of
    filename:last_modified_time pairs for all files rooted at path.
    """
    output = []
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            fullpath = os.path.join(root, filename)
            output.append(
                filename + ":" +
                str(os.path.getmtime(fullpath)) + "\n")
    return "".join(sorted(output))


def read_file(path):
    """Reads a file and returns the data as a string."""
    output = ""
    try:
        # If we could not open then we simply return "" as the output
        with open(path, "r") as content:
            output = content.read()
    except:
        pass
    return output


def write_file(path, output):
    """Writes an output string to the file located at path."""
    with open(path, "w") as content:
        content.write(output)


def substitute_file(path, substitution):
    """Substitutes all instances of substitution[0] with substitution[1] for the
    file located at path."""
    with open(path, "r") as content:
        f_input = content.read()
    if f_input:
        f_input = f_input.replace(substitution[0], substitution[1])
    with open(path, "w") as content:
        content.write(f_input)


def substitute_files(path, substitution):
    """Runs substitute_file() on all files rooted at path."""
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            substitute_file(os.path.join(root, filename), substitution)


def setup_directory(source, dest):
    """Removes the destination directory if it exists and copies the source
    directory over the destination if it exists.
    """
    try:
        shutil.rmtree(dest)
    except OSError as e:
        # shutil will throw if it could not find the directory.
        if e.errno == errno.ENOENT:
            pass
        else:
            raise
    shutil.copytree(source, dest)


def main():
    glsl_src_dir = os.path.normpath(sys.argv[1])
    glsl_bin_dir = os.path.normpath(sys.argv[2])
    intermediate_directory = None
    if (len(sys.argv) > 3):
        intermediate_directory = sys.argv[3]
    glsl_list_file = os.path.join(glsl_bin_dir, "glsl_test_list")

    src_glsl_stamp = get_modified_times(glsl_src_dir)
    old_glsl_stamp = read_file(glsl_list_file)

    target_location = "../glslang/StandAlone/"
    if intermediate_directory:
        target_location = "../" + target_location + intermediate_directory + "/"
    target_location = "EXE=" + target_location

    if src_glsl_stamp != old_glsl_stamp:
        setup_directory(glsl_src_dir, glsl_bin_dir)
        runtests_script = os.path.join(glsl_bin_dir, "runtests")
        substitute_file(runtests_script,
                        ("EXE=../build/install/bin/", target_location))
        write_file(glsl_list_file, src_glsl_stamp)


if __name__ == "__main__":
    main()
