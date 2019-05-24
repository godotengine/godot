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
"""Manages and runs tests from the current working directory.

This will traverse the current working directory and look for python files that
contain subclasses of SpirvTest.

If a class has an @inside_spirv_testsuite decorator, an instance of that
class will be created and serve as a test case in that testsuite.  The test
case is then run by the following steps:

  1. A temporary directory will be created.
  2. The spirv_args member variable will be inspected and all placeholders in it
     will be expanded by calling instantiate_for_spirv_args() on placeholders.
     The transformed list elements are then supplied as arguments to the spirv-*
     tool under test.
  3. If the environment member variable exists, its write() method will be
     invoked.
  4. All expected_* member variables will be inspected and all placeholders in
     them will be expanded by calling instantiate_for_expectation() on those
     placeholders. After placeholder expansion, if the expected_* variable is
     a list, its element will be joined together with '' to form a single
     string. These expected_* variables are to be used by the check_*() methods.
  5. The spirv-* tool will be run with the arguments supplied in spirv_args.
  6. All check_*() member methods will be called by supplying a TestStatus as
     argument. Each check_*() method is expected to return a (Success, Message)
     pair where Success is a boolean indicating success and Message is an error
     message.
  7. If any check_*() method fails, the error message is output and the
     current test case fails.

If --leave-output was not specified, all temporary files and directories will
be deleted.
"""

from __future__ import print_function

import argparse
import fnmatch
import inspect
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from placeholder import PlaceHolder

EXPECTED_BEHAVIOR_PREFIX = 'expected_'
VALIDATE_METHOD_PREFIX = 'check_'


def get_all_variables(instance):
  """Returns the names of all the variables in instance."""
  return [v for v in dir(instance) if not callable(getattr(instance, v))]


def get_all_methods(instance):
  """Returns the names of all methods in instance."""
  return [m for m in dir(instance) if callable(getattr(instance, m))]


def get_all_superclasses(cls):
  """Returns all superclasses of a given class.

    Returns:
      A list of superclasses of the given class. The order guarantees that
      * A Base class precedes its derived classes, e.g., for "class B(A)", it
        will be [..., A, B, ...].
      * When there are multiple base classes, base classes declared first
        precede those declared later, e.g., for "class C(A, B), it will be
        [..., A, B, C, ...]
    """
  classes = []
  for superclass in cls.__bases__:
    for c in get_all_superclasses(superclass):
      if c not in classes:
        classes.append(c)
  for superclass in cls.__bases__:
    if superclass not in classes:
      classes.append(superclass)
  return classes


def get_all_test_methods(test_class):
  """Gets all validation methods.

    Returns:
      A list of validation methods. The order guarantees that
      * A method defined in superclass precedes one defined in subclass,
        e.g., for "class A(B)", methods defined in B precedes those defined
        in A.
      * If a subclass has more than one superclass, e.g., "class C(A, B)",
        then methods defined in A precedes those defined in B.
    """
  classes = get_all_superclasses(test_class)
  classes.append(test_class)
  all_tests = [
      m for c in classes for m in get_all_methods(c)
      if m.startswith(VALIDATE_METHOD_PREFIX)
  ]
  unique_tests = []
  for t in all_tests:
    if t not in unique_tests:
      unique_tests.append(t)
  return unique_tests


class SpirvTest:
  """Base class for spirv test cases.

    Subclasses define test cases' facts (shader source code, spirv command,
    result validation), which will be used by the TestCase class for running
    tests. Subclasses should define spirv_args (specifying spirv_tool command
    arguments), and at least one check_*() method (for result validation) for
    a full-fledged test case. All check_*() methods should take a TestStatus
    parameter and return a (Success, Message) pair, in which Success is a
    boolean indicating success and Message is an error message. The test passes
    iff all check_*() methods returns true.

    Often, a test case class will delegate the check_* behaviors by inheriting
    from other classes.
    """

  def name(self):
    return self.__class__.__name__


class TestStatus:
  """A struct for holding run status of a test case."""

  def __init__(self, test_manager, returncode, stdout, stderr, directory,
               inputs, input_filenames):
    self.test_manager = test_manager
    self.returncode = returncode
    self.stdout = stdout
    self.stderr = stderr
    # temporary directory where the test runs
    self.directory = directory
    # List of inputs, as PlaceHolder objects.
    self.inputs = inputs
    # the names of input shader files (potentially including paths)
    self.input_filenames = input_filenames


class SpirvTestException(Exception):
  """SpirvTest exception class."""
  pass


def inside_spirv_testsuite(testsuite_name):
  """Decorator for subclasses of SpirvTest.

    This decorator checks that a class meets the requirements (see below)
    for a test case class, and then puts the class in a certain testsuite.
    * The class needs to be a subclass of SpirvTest.
    * The class needs to have spirv_args defined as a list.
    * The class needs to define at least one check_*() methods.
    * All expected_* variables required by check_*() methods can only be
      of bool, str, or list type.
    * Python runtime will throw an exception if the expected_* member
      attributes required by check_*() methods are missing.
    """

  def actual_decorator(cls):
    if not inspect.isclass(cls):
      raise SpirvTestException('Test case should be a class')
    if not issubclass(cls, SpirvTest):
      raise SpirvTestException(
          'All test cases should be subclasses of SpirvTest')
    if 'spirv_args' not in get_all_variables(cls):
      raise SpirvTestException('No spirv_args found in the test case')
    if not isinstance(cls.spirv_args, list):
      raise SpirvTestException('spirv_args needs to be a list')
    if not any(
        [m.startswith(VALIDATE_METHOD_PREFIX) for m in get_all_methods(cls)]):
      raise SpirvTestException('No check_*() methods found in the test case')
    if not all(
        [isinstance(v, (bool, str, list)) for v in get_all_variables(cls)]):
      raise SpirvTestException(
          'expected_* variables are only allowed to be bool, str, or '
          'list type.')
    cls.parent_testsuite = testsuite_name
    return cls

  return actual_decorator


class TestManager:
  """Manages and runs a set of tests."""

  def __init__(self, executable_path, assembler_path, disassembler_path):
    self.executable_path = executable_path
    self.assembler_path = assembler_path
    self.disassembler_path = disassembler_path
    self.num_successes = 0
    self.num_failures = 0
    self.num_tests = 0
    self.leave_output = False
    self.tests = defaultdict(list)

  def notify_result(self, test_case, success, message):
    """Call this to notify the manager of the results of a test run."""
    self.num_successes += 1 if success else 0
    self.num_failures += 0 if success else 1
    counter_string = str(self.num_successes + self.num_failures) + '/' + str(
        self.num_tests)
    print('%-10s %-40s ' % (counter_string, test_case.test.name()) +
          ('Passed' if success else '-Failed-'))
    if not success:
      print(' '.join(test_case.command))
      print(message)

  def add_test(self, testsuite, test):
    """Add this to the current list of test cases."""
    self.tests[testsuite].append(TestCase(test, self))
    self.num_tests += 1

  def run_tests(self):
    for suite in self.tests:
      print('SPIRV tool test suite: "{suite}"'.format(suite=suite))
      for x in self.tests[suite]:
        x.runTest()


class TestCase:
  """A single test case that runs in its own directory."""

  def __init__(self, test, test_manager):
    self.test = test
    self.test_manager = test_manager
    self.inputs = []  # inputs, as PlaceHolder objects.
    self.file_shaders = []  # filenames of shader files.
    self.stdin_shader = None  # text to be passed to spirv_tool as stdin

  def setUp(self):
    """Creates environment and instantiates placeholders for the test case."""

    self.directory = tempfile.mkdtemp(dir=os.getcwd())
    spirv_args = self.test.spirv_args
    # Instantiate placeholders in spirv_args
    self.test.spirv_args = [
        arg.instantiate_for_spirv_args(self)
        if isinstance(arg, PlaceHolder) else arg for arg in self.test.spirv_args
    ]
    # Get all shader files' names
    self.inputs = [arg for arg in spirv_args if isinstance(arg, PlaceHolder)]
    self.file_shaders = [arg.filename for arg in self.inputs]

    if 'environment' in get_all_variables(self.test):
      self.test.environment.write(self.directory)

    expectations = [
        v for v in get_all_variables(self.test)
        if v.startswith(EXPECTED_BEHAVIOR_PREFIX)
    ]
    # Instantiate placeholders in expectations
    for expectation_name in expectations:
      expectation = getattr(self.test, expectation_name)
      if isinstance(expectation, list):
        expanded_expections = [
            element.instantiate_for_expectation(self)
            if isinstance(element, PlaceHolder) else element
            for element in expectation
        ]
        setattr(self.test, expectation_name, expanded_expections)
      elif isinstance(expectation, PlaceHolder):
        setattr(self.test, expectation_name,
                expectation.instantiate_for_expectation(self))

  def tearDown(self):
    """Removes the directory if we were not instructed to do otherwise."""
    if not self.test_manager.leave_output:
      shutil.rmtree(self.directory)

  def runTest(self):
    """Sets up and runs a test, reports any failures and then cleans up."""
    self.setUp()
    success = False
    message = ''
    try:
      self.command = [self.test_manager.executable_path]
      self.command.extend(self.test.spirv_args)

      process = subprocess.Popen(
          args=self.command,
          stdin=subprocess.PIPE,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          cwd=self.directory)
      output = process.communicate(self.stdin_shader)
      test_status = TestStatus(self.test_manager, process.returncode, output[0],
                               output[1], self.directory, self.inputs,
                               self.file_shaders)
      run_results = [
          getattr(self.test, test_method)(test_status)
          for test_method in get_all_test_methods(self.test.__class__)
      ]
      success, message = zip(*run_results)
      success = all(success)
      message = '\n'.join(message)
    except Exception as e:
      success = False
      message = str(e)
    self.test_manager.notify_result(
        self, success,
        message + '\nSTDOUT:\n%s\nSTDERR:\n%s' % (output[0], output[1]))
    self.tearDown()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      'spirv_tool',
      metavar='path/to/spirv_tool',
      type=str,
      nargs=1,
      help='Path to the spirv-* tool under test')
  parser.add_argument(
      'spirv_as',
      metavar='path/to/spirv-as',
      type=str,
      nargs=1,
      help='Path to spirv-as')
  parser.add_argument(
      'spirv_dis',
      metavar='path/to/spirv-dis',
      type=str,
      nargs=1,
      help='Path to spirv-dis')
  parser.add_argument(
      '--leave-output',
      action='store_const',
      const=1,
      help='Do not clean up temporary directories')
  parser.add_argument(
      '--test-dir', nargs=1, help='Directory to gather the tests from')
  args = parser.parse_args()
  default_path = sys.path
  root_dir = os.getcwd()
  if args.test_dir:
    root_dir = args.test_dir[0]
  manager = TestManager(args.spirv_tool[0], args.spirv_as[0], args.spirv_dis[0])
  if args.leave_output:
    manager.leave_output = True
  for root, _, filenames in os.walk(root_dir):
    for filename in fnmatch.filter(filenames, '*.py'):
      if filename.endswith('nosetest.py'):
        # Skip nose tests, which are for testing functions of
        # the test framework.
        continue
      sys.path = default_path
      sys.path.append(root)
      mod = __import__(os.path.splitext(filename)[0])
      for _, obj, in inspect.getmembers(mod):
        if inspect.isclass(obj) and hasattr(obj, 'parent_testsuite'):
          manager.add_test(obj.parent_testsuite, obj())
  manager.run_tests()
  if manager.num_failures > 0:
    sys.exit(-1)


if __name__ == '__main__':
  main()
