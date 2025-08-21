#!/usr/bin/env python3

#              Copyright Catch2 Authors
# Distributed under the Boost Software License, Version 1.0.
#   (See accompanying file LICENSE.txt or copy at
#        https://www.boost.org/LICENSE_1_0.txt)

# SPDX-License-Identifier: BSL-1.0

import os
import re
import sys
import xml.etree.ElementTree as ET
import subprocess

"""
Test that Catch2 recognizes `XML_OUTPUT_FILE` env variable and creates
a junit reporter that writes to the provided path.

Requires 2 arguments, path to Catch2 binary configured with
`CATCH_CONFIG_BAZEL_SUPPORT`, and the output directory for the output file.
"""
if len(sys.argv) != 3:
    print("Wrong number of arguments: {}".format(len(sys.argv)))
    print("Usage: {} test-bin-path output-dir".format(sys.argv[0]))
    exit(1)


bin_path = os.path.abspath(sys.argv[1])
output_dir = os.path.abspath(sys.argv[2])
xml_out_path = os.path.join(output_dir, '{}.xml'.format(os.path.basename(bin_path)))

# Ensure no file exists from previous test runs
if os.path.isfile(xml_out_path):
    os.remove(xml_out_path)

print('bin path:', bin_path)
print('xml out path:', xml_out_path)

env = os.environ.copy()
env["XML_OUTPUT_FILE"] = xml_out_path
test_passing = True

try:
    ret = subprocess.run(
        bin_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        universal_newlines=True,
        env=env
    )
    stdout = ret.stdout
except subprocess.SubprocessError as ex:
    if ex.returncode == 42:
        # The test cases are allowed to fail.
        test_passing = False
        stdout = ex.stdout
    else:
        print('Could not run "{}"'.format(bin_path))
        print("Return code: {}".format(ex.returncode))
        print("stdout: {}".format(ex.stdout))
        print("stderr: {}".format(ex.stderr))
        raise

# Check for valid XML output
try:
    tree = ET.parse(xml_out_path)
except ET.ParseError as ex:
    print("Invalid XML: '{}'".format(ex))
    raise
except FileNotFoundError as ex:
    print("Could not find '{}'".format(xml_out_path))
    raise

bin_name = os.path.basename(bin_path)
# Check for matching testsuite
if not tree.find('.//testsuite[@name="{}"]'.format(bin_name)):
    print("Could not find '{}' testsuite".format(bin_name))
    exit(2)

# Check that we haven't disabled the default reporter
summary_test_cases = re.findall(r'test cases: \d* \| \d* passed \| \d* failed', stdout)
if len(summary_test_cases) == 0:
    print("Could not find test summary in {}".format(stdout))
    exit(2)

total, passed, failed = [int(s) for s in summary_test_cases[0].split() if s.isdigit()]

if failed == 0 and not test_passing:
    print("Expected at least 1 test failure!")
    exit(2)

if len(tree.findall('.//testcase')) != total:
    print("Unexpected number of test cases!")
    exit(2)

if len(tree.findall('.//failure')) != failed:
    print("Unexpected number of test failures!")
    exit(2)

if (passed + failed) != total:
    print("Something has gone very wrong, ({} + {}) != {}".format(passed, failed, total))
    exit(2)
