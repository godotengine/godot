#!/usr/bin/env python3

#              Copyright Catch2 Authors
# Distributed under the Boost Software License, Version 1.0.
#   (See accompanying file LICENSE.txt or copy at
#        https://www.boost.org/LICENSE_1_0.txt)

# SPDX-License-Identifier: BSL-1.0

import os
import subprocess
import sys
import re
import json
from collections import namedtuple
from typing import List

TestInfo = namedtuple('TestInfo', ['name', 'tags'])

cmake_version_regex = re.compile(r'cmake version (\d+)\.(\d+)\.(\d+)')

def get_cmake_version():
    result = subprocess.run(['cmake', '--version'],
                            capture_output = True,
                            check = True,
                            text = True)
    version_match = cmake_version_regex.match(result.stdout)
    if not version_match:
        print('Could not find cmake version in output')
        print(f"output: '{result.stdout}'")
        exit(4)
    return (int(version_match.group(1)),
            int(version_match.group(2)),
            int(version_match.group(3)))

def build_project(sources_dir, output_base_path, catch2_path):
    build_dir = os.path.join(output_base_path, 'ctest-registration-test')
    config_cmd = ['cmake',
                  '-B', build_dir,
                  '-S', sources_dir,
                  f'-DCATCH2_PATH={catch2_path}',
                  '-DCMAKE_BUILD_TYPE=Debug']

    build_cmd = ['cmake',
                 '--build', build_dir,
                 '--config', 'Debug']

    try:
        subprocess.run(config_cmd,
                       capture_output = True,
                       check = True,
                       text = True)
        subprocess.run(build_cmd,
                       capture_output = True,
                       check = True,
                       text = True)
    except subprocess.CalledProcessError as err:
        print('Error when building the test project')
        print(f'cmd: {err.cmd}')
        print(f'stderr: {err.stderr}')
        print(f'stdout: {err.stdout}')
        exit(3)

    return build_dir



def get_test_names(build_path: str) -> List[TestInfo]:
    # For now we assume that Windows builds are done using MSBuild under
    # Debug configuration. This means that we need to add "Debug" folder
    # to the path when constructing it. On Linux, we don't add anything.
    config_path = "Debug" if os.name == 'nt' else ""
    full_path = os.path.join(build_path, config_path, 'tests')


    cmd = [full_path, '--reporter', 'json', '--list-tests']
    result = subprocess.run(cmd,
                            capture_output = True,
                            check = True,
                            text = True)

    test_listing = json.loads(result.stdout)

    assert test_listing['version'] == 1

    tests = []
    for test in test_listing['listings']['tests']:
        test_name = test['name']
        tags = test['tags']
        tests.append(TestInfo(test_name, tags))

    return tests

def get_ctest_listing(build_path):
    old_path = os.getcwd()
    os.chdir(build_path)

    cmd = ['ctest', '-C', 'debug', '--show-only=json-v1']
    result = subprocess.run(cmd,
                            capture_output = True,
                            check = True,
                            text = True)
    os.chdir(old_path)
    return result.stdout

def extract_tests_from_ctest(ctest_output) -> List[TestInfo]:
    ctest_response = json.loads(ctest_output)
    tests = ctest_response['tests']
    test_infos = []
    for test in tests:
        test_command = test['command']
        # First part of the command is the binary, second is the filter.
        # If there are less, registration has failed. If there are more,
        # registration has changed and the script needs updating.
        assert len(test_command) == 2
        test_name = test_command[1]
        labels = []
        for prop in test['properties']:
            if prop['name'] == 'LABELS':
                labels = prop['value']

        test_infos.append(TestInfo(test_name, labels))

    return test_infos

def check_DL_PATHS(ctest_output):
    ctest_response = json.loads(ctest_output)
    tests = ctest_response['tests']
    for test in tests:
        properties = test['properties']
        for property in properties:
            if property['name'] == 'ENVIRONMENT_MODIFICATION':
                assert len(property['value']) == 2, f"The test provides 2 arguments to DL_PATHS, but instead found {len(property['value'])}"

def escape_catch2_test_names(infos: List[TestInfo]):
    escaped = []
    for info in infos:
      name = info.name
      for char in ('\\', ',', '[', ']'):
          name = name.replace(char, f"\\{char}")
      escaped.append(TestInfo(name, info.tags))
    return escaped

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} path-to-catch2-cml output-path')
        exit(2)
    catch2_path = sys.argv[1]
    output_base_path = sys.argv[2]
    sources_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

    build_path = build_project(sources_dir, output_base_path, catch2_path)

    catch_test_names = escape_catch2_test_names(get_test_names(build_path))
    ctest_output = get_ctest_listing(build_path)
    ctest_test_names = extract_tests_from_ctest(ctest_output)

    mismatched = 0
    for catch_test in catch_test_names:
        if catch_test not in ctest_test_names:
            print(f"Catch2 test '{catch_test}' not found in CTest")
            mismatched += 1
    for ctest_test in ctest_test_names:
        if ctest_test not in catch_test_names:
            print(f"CTest test '{ctest_test}' not found in Catch2")
            mismatched += 1

    if mismatched:
        print(f"Found {mismatched} mismatched tests catch test names and ctest test commands!")
        exit(1)
    print(f"{len(catch_test_names)} tests matched")

    cmake_version = get_cmake_version()
    if cmake_version >= (3, 27):
        check_DL_PATHS(ctest_output)
