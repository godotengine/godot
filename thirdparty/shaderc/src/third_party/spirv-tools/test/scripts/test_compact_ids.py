#!/usr/bin/env python
# Copyright (c) 2017 Google Inc.

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
"""Tests correctness of opt pass tools/opt --compact-ids."""

from __future__ import print_function

import os.path
import sys
import tempfile

def test_spirv_file(path, temp_dir):
  optimized_spv_path = os.path.join(temp_dir, 'optimized.spv')
  optimized_dis_path = os.path.join(temp_dir, 'optimized.dis')
  converted_spv_path = os.path.join(temp_dir, 'converted.spv')
  converted_dis_path = os.path.join(temp_dir, 'converted.dis')

  os.system('tools/spirv-opt ' + path + ' -o ' + optimized_spv_path +
            ' --compact-ids')
  os.system('tools/spirv-dis ' + optimized_spv_path + ' -o ' +
            optimized_dis_path)

  os.system('tools/spirv-dis ' + path + ' -o ' + converted_dis_path)
  os.system('tools/spirv-as ' + converted_dis_path + ' -o ' +
            converted_spv_path)
  os.system('tools/spirv-dis ' + converted_spv_path + ' -o ' +
            converted_dis_path)

  with open(converted_dis_path, 'r') as f:
    converted_dis = f.readlines()[3:]

  with open(optimized_dis_path, 'r') as f:
    optimized_dis = f.readlines()[3:]

  return converted_dis == optimized_dis

def print_usage():
  template= \
"""{script} tests correctness of opt pass tools/opt --compact-ids

USAGE: python {script} [<spirv_files>]

Requires tools/spirv-dis, tools/spirv-as and tools/spirv-opt to be in path
(call the script from the SPIRV-Tools build output directory).

TIP: In order to test all .spv files under current dir use
find <path> -name "*.spv" -print0 | xargs -0 -s 2000000 python {script}
"""
  print(template.format(script=sys.argv[0]));

def main():
  if not os.path.isfile('tools/spirv-dis'):
      print('error: tools/spirv-dis not found')
      print_usage()
      exit(1)

  if not os.path.isfile('tools/spirv-as'):
      print('error: tools/spirv-as not found')
      print_usage()
      exit(1)

  if not os.path.isfile('tools/spirv-opt'):
      print('error: tools/spirv-opt not found')
      print_usage()
      exit(1)

  paths = sys.argv[1:]
  if not paths:
      print_usage()

  num_failed = 0

  temp_dir = tempfile.mkdtemp()

  for path in paths:
    success = test_spirv_file(path, temp_dir)
    if not success:
      print('Test failed for ' + path)
      num_failed += 1

  print('Tested ' + str(len(paths)) + ' files')

  if num_failed:
    print(str(num_failed) + ' tests failed')
    exit(1)
  else:
    print('All tests successful')
    exit(0)

if __name__ == '__main__':
  main()
