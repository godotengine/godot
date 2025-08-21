#!/usr/bin/env python3

#              Copyright Catch2 Authors
# Distributed under the Boost Software License, Version 1.0.
#   (See accompanying file LICENSE.txt or copy at
#        https://www.boost.org/LICENSE_1_0.txt)

# SPDX-License-Identifier: BSL-1.0

"""
This test script verifies that the testCasePartial{Starting,Ended} reporter
events fire properly. This is done by calling a test binary compiled with
reporter that reports specifically testCase* events, and verifying the
outputs match what we expect.
"""

import subprocess
import sys

expected_section_output = '''\
TestCaseStarting: section
TestCaseStartingPartial: section#0
TestCasePartialEnded: section#0
TestCaseStartingPartial: section#1
TestCasePartialEnded: section#1
TestCaseStartingPartial: section#2
TestCasePartialEnded: section#2
TestCaseStartingPartial: section#3
TestCasePartialEnded: section#3
TestCaseEnded: section
'''

expected_generator_output = '''\
TestCaseStarting: generator
TestCaseStartingPartial: generator#0
TestCasePartialEnded: generator#0
TestCaseStartingPartial: generator#1
TestCasePartialEnded: generator#1
TestCaseStartingPartial: generator#2
TestCasePartialEnded: generator#2
TestCaseStartingPartial: generator#3
TestCasePartialEnded: generator#3
TestCaseEnded: generator
'''


from typing import List

def get_test_output(test_exe: str, sections: bool) -> List[str]:
    cmd = [test_exe, '--reporter', 'partial']
    if sections:
        cmd.append('section')
    else:
        cmd.append('generator')

    ret = subprocess.run(cmd,
                         stdout = subprocess.PIPE,
                         stderr = subprocess.PIPE,
                         timeout = 10,
                         check = True,
                         universal_newlines = True)

    return ret.stdout

def main():
    test_exe, = sys.argv[1:]
    actual_section_output = get_test_output(test_exe, sections = True)

    assert actual_section_output == expected_section_output, (
    'Sections\nActual:\n{}\nExpected:\n{}\n'.format(actual_section_output, expected_section_output))

    actual_generator_output = get_test_output(test_exe, sections = False)
    assert actual_generator_output == expected_generator_output, (
    'Generators\nActual:\n{}\nExpected:\n{}\n'.format(actual_generator_output, expected_generator_output))



if __name__ == '__main__':
    sys.exit(main())
