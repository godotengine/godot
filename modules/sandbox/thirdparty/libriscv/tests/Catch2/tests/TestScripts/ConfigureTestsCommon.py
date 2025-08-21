#!/usr/bin/env python3

#              Copyright Catch2 Authors
# Distributed under the Boost Software License, Version 1.0.
#   (See accompanying file LICENSE.txt or copy at
#        https://www.boost.org/LICENSE_1_0.txt)

# SPDX-License-Identifier: BSL-1.0

from typing import List, Tuple

import os
import subprocess

def configure_and_build(source_path: str, project_path: str, options: List[Tuple[str, str]]):
    base_configure_cmd = ['cmake',
                          '-B{}'.format(project_path),
                          '-S{}'.format(source_path),
                          '-DCMAKE_BUILD_TYPE=Debug',
                          '-DCATCH_DEVELOPMENT_BUILD=ON']
    for option, value in options:
        base_configure_cmd.append('-D{}={}'.format(option, value))
    try:
        subprocess.run(base_configure_cmd,
                       stdout = subprocess.PIPE,
                       stderr = subprocess.STDOUT,
                       check = True)
    except subprocess.SubprocessError as ex:
        print("Could not configure build to '{}' from '{}'".format(project_path, source_path))
        print("Return code: {}".format(ex.returncode))
        print("output: {}".format(ex.output))
        raise
    print('Configuring {} finished'.format(project_path))

    build_cmd = ['cmake',
                 '--build', '{}'.format(project_path),
                 # For now we assume that we only need Debug config
                 '--config', 'Debug']
    try:
        subprocess.run(build_cmd,
                       stdout = subprocess.PIPE,
                       stderr = subprocess.STDOUT,
                       check = True)
    except subprocess.SubprocessError as ex:
        print("Could not build project in '{}'".format(project_path))
        print("Return code: {}".format(ex.returncode))
        print("output: {}".format(ex.output))
        raise
    print('Building {} finished'.format(project_path))

def run_and_return_output(base_path: str, binary_name: str, other_options: List[str]) -> Tuple[str, str]:
    # For now we assume that Windows builds are done using MSBuild under
    # Debug configuration. This means that we need to add "Debug" folder
    # to the path when constructing it. On Linux, we don't add anything.
    config_path = "Debug" if os.name == 'nt' else ""
    full_path = os.path.join(base_path, config_path, binary_name)

    base_cmd = [full_path]
    base_cmd.extend(other_options)

    try:
        ret = subprocess.run(base_cmd,
                             stdout = subprocess.PIPE,
                             stderr = subprocess.PIPE,
                             check = True,
                             universal_newlines = True)
    except subprocess.SubprocessError as ex:
        print('Could not run "{}"'.format(base_cmd))
        print('Args: "{}"'.format(other_options))
        print('Return code: {}'.format(ex.returncode))
        print('stdout: {}'.format(ex.stdout))
        print('stderr: {}'.format(ex.stdout))
        raise

    return (ret.stdout, ret.stderr)
