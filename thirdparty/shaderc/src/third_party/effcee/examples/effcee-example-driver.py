#!/usr/bin/env python

# Copyright 2017 The Effcee Authors.
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

"""Execute the effcee-example program, where the arguments are
the location of the example program, the input file, and a
list of check rules to match against the input.

Args:
    effcee-example:   Path to the effcee-example executable
    input_file:       Data file containing the input to match
    check1 .. checkN: Check rules to match
"""

import subprocess
import sys

def main():
    cmd = sys.argv[1]
    input_file = sys.argv[2]
    checks = sys.argv[3:]
    args = [cmd]
    args.extend(checks)
    print(args)
    with open(input_file) as input_stream:
        sys.exit(subprocess.call(args, stdin=input_stream))
    sys.exit(1)

if __name__ == '__main__':
    main()
