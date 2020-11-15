#!/usr/bin/env python
# coding: utf-8

# Copyright 2017 The Crashpad Authors. All rights reserved.
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

# Generating Doxygen documentation requires Doxygen, http://www.doxygen.org/.

import os
import shutil
import subprocess
import sys


def main(args):
  script_dir = os.path.dirname(__file__)
  crashpad_dir = os.path.join(script_dir, os.pardir, os.pardir)

  # Run from the Crashpad project root directory.
  os.chdir(crashpad_dir)

  output_dir = os.path.join('out', 'doc', 'doxygen')

  if os.path.isdir(output_dir) and not os.path.islink(output_dir):
    shutil.rmtree(output_dir)
  elif os.path.exists(output_dir):
    os.unlink(output_dir)

  os.makedirs(output_dir, 0o755)

  doxy_file = os.path.join('doc', 'support', 'crashpad.doxy')
  subprocess.check_call(['doxygen', doxy_file])

  return 0


if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))
