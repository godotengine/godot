# Copyright (c) 2018 Google LLC
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
"""Tests for the expect module."""

import expect
from spirv_test_framework import TestStatus
from nose.tools import assert_equal, assert_true, assert_false
import re


def nosetest_get_object_name():
  """Tests get_object_filename()."""
  source_and_object_names = [('a.vert', 'a.vert.spv'), ('b.frag', 'b.frag.spv'),
                             ('c.tesc', 'c.tesc.spv'), ('d.tese', 'd.tese.spv'),
                             ('e.geom', 'e.geom.spv'), ('f.comp', 'f.comp.spv'),
                             ('file', 'file.spv'), ('file.', 'file.spv'),
                             ('file.uk',
                              'file.spv'), ('file.vert.',
                                            'file.vert.spv'), ('file.vert.bla',
                                                               'file.vert.spv')]
  actual_object_names = [
      expect.get_object_filename(f[0]) for f in source_and_object_names
  ]
  expected_object_names = [f[1] for f in source_and_object_names]

  assert_equal(actual_object_names, expected_object_names)


class TestStdoutMatchADotC(expect.StdoutMatch):
  expected_stdout = re.compile('a.c')


def nosetest_stdout_match_regex_has_match():
  test = TestStdoutMatchADotC()
  status = TestStatus(
      test_manager=None,
      returncode=0,
      stdout='0abc1',
      stderr=None,
      directory=None,
      inputs=None,
      input_filenames=None)
  assert_true(test.check_stdout_match(status)[0])


def nosetest_stdout_match_regex_no_match():
  test = TestStdoutMatchADotC()
  status = TestStatus(
      test_manager=None,
      returncode=0,
      stdout='ab',
      stderr=None,
      directory=None,
      inputs=None,
      input_filenames=None)
  assert_false(test.check_stdout_match(status)[0])


def nosetest_stdout_match_regex_empty_stdout():
  test = TestStdoutMatchADotC()
  status = TestStatus(
      test_manager=None,
      returncode=0,
      stdout='',
      stderr=None,
      directory=None,
      inputs=None,
      input_filenames=None)
  assert_false(test.check_stdout_match(status)[0])
