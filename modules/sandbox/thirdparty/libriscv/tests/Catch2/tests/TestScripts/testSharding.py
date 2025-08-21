#!/usr/bin/env python3

#              Copyright Catch2 Authors
# Distributed under the Boost Software License, Version 1.0.
#   (See accompanying file LICENSE.txt or copy at
#        https://www.boost.org/LICENSE_1_0.txt)

# SPDX-License-Identifier: BSL-1.0

"""
This test script verifies that sharding tests does change which tests are run.
This is done by running the binary multiple times, once to list all the tests,
once per shard to list the tests for that shard, and once again per shard to
execute the tests. The sharded lists are compared to the full list to ensure
none are skipped, duplicated, and that the order remains the same.
"""

import random
import subprocess
import sys
import xml.etree.ElementTree as ET

from collections import namedtuple

from typing import List, Dict

seed = random.randint(0, 2 ** 32 - 1)
number_of_shards = 5

def make_base_commandline(self_test_exe):
    return [
        self_test_exe,
        '--reporter', 'xml',
        '--order', 'rand',
        '--rng-seed', str(seed),
        "[generators]~[benchmarks]~[.]"
    ]


def list_tests(self_test_exe: str, extra_args: List[str] = None):
    cmd = make_base_commandline(self_test_exe) + ['--list-tests']
    if extra_args:
        cmd.extend(extra_args)

    try:
        ret = subprocess.run(cmd,
                             stdout = subprocess.PIPE,
                             stderr = subprocess.PIPE,
                             timeout = 10,
                             check = True,
                             universal_newlines = True)
    except subprocess.CalledProcessError as ex:
        print('Could not list tests:\n{}'.format(ex.stderr))

    if ret.stderr:
        raise RuntimeError("Unexpected error output:\n" + ret.stderr)

    root = ET.fromstring(ret.stdout)
    result = [elem.text for elem in root.findall('./TestCase/Name')]

    if len(result) < 2:
        raise RuntimeError("Unexpectedly few tests listed (got {})".format(
            len(result)))


    return result


def execute_tests(self_test_exe: str, extra_args: List[str] = None):
    cmd = make_base_commandline(self_test_exe)
    if extra_args:
        cmd.extend(extra_args)

    try:
        ret = subprocess.run(cmd,
                             stdout = subprocess.PIPE,
                             stderr = subprocess.PIPE,
                             timeout = 10,
                             check = True,
                             universal_newlines = True)
    except subprocess.CalledProcessError as ex:
        print('Could not list tests:\n{}'.format(ex.stderr))

    if ret.stderr:
        raise RuntimeError("Unexpected error output:\n" + process.stderr)

    root = ET.fromstring(ret.stdout)
    result = [elem.attrib["name"] for elem in root.findall('./TestCase')]

    if len(result) < 2:
        raise RuntimeError("Unexpectedly few tests listed (got {})".format(
            len(result)))

    return result


def test_sharded_listing(self_test_exe: str) -> Dict[int, List[str]]:
    """
    Asks the test binary for list of all tests, and also for lists of
    tests from shards.

    The combination of shards is then checked whether it corresponds to
    the full list of all tests.

    Returns the dictionary of shard-index => listed tests for later use.
    """
    all_tests = list_tests(self_test_exe)
    big_shard_tests = list_tests(self_test_exe, ['--shard-count', '1', '--shard-index', '0'])

    assert all_tests == big_shard_tests, (
        "No-sharding test list does not match the listing of big shard:\nNo shard:\n{}\n\nWith shard:\n{}\n".format(
            '\n'.join(all_tests),
            '\n'.join(big_shard_tests)
        )
    )

    shard_listings = dict()
    for shard_idx in range(number_of_shards):
        shard_listings[shard_idx] = list_tests(self_test_exe, ['--shard-count', str(number_of_shards), '--shard-index', str(shard_idx)])

    shard_sizes = [len(v) for v in shard_listings.values()]
    assert len(all_tests) == sum(shard_sizes)

    # Check that the shards have roughly the right sizes (e.g. we don't
    # have all tests in single shard and the others are empty)
    differences = [abs(x1 - x2) for x1, x2 in zip(shard_sizes, shard_sizes[1:])]
    assert all(diff <= 1 for diff in differences), "A shard has weird size: {}".format(shard_sizes)

    combined_shards = [inner for outer in shard_listings.values() for inner in outer]
    assert all_tests == combined_shards, (
        "All tests and combined shards disagree.\nNo shard:\n{}\n\nCombined:\n{}\n\n".format(
            '\n'.join(all_tests),
            '\n'.join(combined_shards)
        )
    )
    shard_listings[-1] = all_tests

    return shard_listings


def test_sharded_execution(self_test_exe: str, listings: Dict[int, List[str]]):
    """
    Runs the test binary and checks that the executed tests match the
    previously listed tests.

    Also does this for various shard indices, and that the combination
    of all shards matches the full run/listing.
    """
    all_tests = execute_tests(self_test_exe)
    big_shard_tests = execute_tests(self_test_exe, ['--shard-count', '1', '--shard-index', '0'])
    assert all_tests == big_shard_tests

    assert listings[-1] == all_tests

    for shard_idx in range(number_of_shards):
        assert listings[shard_idx] == execute_tests(self_test_exe, ['--shard-count', str(number_of_shards), '--shard-index', str(shard_idx)])


def main():
    self_test_exe, = sys.argv[1:]
    listings = test_sharded_listing(self_test_exe)
    test_sharded_execution(self_test_exe, listings)

if __name__ == '__main__':
    sys.exit(main())
