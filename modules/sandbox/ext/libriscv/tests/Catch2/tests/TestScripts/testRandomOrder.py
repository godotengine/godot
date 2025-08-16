#!/usr/bin/env python3

#              Copyright Catch2 Authors
# Distributed under the Boost Software License, Version 1.0.
#   (See accompanying file LICENSE.txt or copy at
#        https://www.boost.org/LICENSE_1_0.txt)

# SPDX-License-Identifier: BSL-1.0

"""
This test script verifies that the random ordering of tests inside
Catch2 is invariant in regards to subsetting. This is done by running
the binary 3 times, once with all tests selected, and twice with smaller
subsets of tests selected, and verifying that the selected tests are in
the same relative order.
"""

import subprocess
import sys
import random
import xml.etree.ElementTree as ET

def none_to_empty_str(e):
    if e is None:
        return ""
    assert type(e) is str
    return e

def list_tests(self_test_exe, tags, rng_seed):
    cmd = [self_test_exe, '--reporter', 'xml', '--list-tests', '--order', 'rand',
            '--rng-seed', str(rng_seed)]
    tags_arg = ','.join('[{}]~[.]'.format(t) for t in tags)
    if tags_arg:
        cmd.append(tags_arg)
    process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if stderr:
        raise RuntimeError("Unexpected error output:\n" + process.stderr)

    root = ET.fromstring(stdout)
    result = [(none_to_empty_str(tc.find('Name').text),
               none_to_empty_str(tc.find('Tags').text),
               none_to_empty_str(tc.find('ClassName').text)) for tc in root.findall('./TestCase')]

    if len(result) < 2:
        raise RuntimeError("Unexpectedly few tests listed (got {})".format(
            len(result)))
    return result

def check_is_sublist_of(shorter, longer):
    assert len(shorter) < len(longer)
    assert len(set(longer)) == len(longer)

    indexes_in_longer = {s: i for i, s in enumerate(longer)}
    for s1, s2 in zip(shorter, shorter[1:]):
        assert indexes_in_longer[s1] < indexes_in_longer[s2], (
                '{} comes before {} in longer list.\n'
                'Longer: {}\nShorter: {}'.format(s2, s1, longer, shorter))

def main():
    self_test_exe, = sys.argv[1:]

    # We want a random seed for the test, but want to avoid 0,
    # because it has special meaning
    seed = random.randint(1, 2 ** 32 - 1)

    list_one_tag = list_tests(self_test_exe, ['generators'], seed)
    list_two_tags = list_tests(self_test_exe, ['generators', 'matchers'], seed)
    list_all = list_tests(self_test_exe, [], seed)

    # First, verify that restricting to a subset yields the same order
    check_is_sublist_of(list_two_tags, list_all)
    check_is_sublist_of(list_one_tag, list_two_tags)

if __name__ == '__main__':
    sys.exit(main())
