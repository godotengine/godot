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
"""A number of common spirv result checks coded in mixin classes.

A test case can use these checks by declaring their enclosing mixin classes
as superclass and providing the expected_* variables required by the check_*()
methods in the mixin classes.
"""
import difflib
import functools
import os
import re
import subprocess
import traceback
from spirv_test_framework import SpirvTest
from builtins import bytes

def convert_to_unix_line_endings(source):
  """Converts all line endings in source to be unix line endings."""
  result = source.replace('\r\n', '\n').replace('\r', '\n')
  return result


def substitute_file_extension(filename, extension):
  """Substitutes file extension, respecting known shader extensions.

    foo.vert -> foo.vert.[extension] [similarly for .frag, .comp, etc.]
    foo.glsl -> foo.[extension]
    foo.unknown -> foo.[extension]
    foo -> foo.[extension]
    """
  if filename[-5:] not in [
      '.vert', '.frag', '.tesc', '.tese', '.geom', '.comp', '.spvasm'
  ]:
    return filename.rsplit('.', 1)[0] + '.' + extension
  else:
    return filename + '.' + extension


def get_object_filename(source_filename):
  """Gets the object filename for the given source file."""
  return substitute_file_extension(source_filename, 'spv')


def get_assembly_filename(source_filename):
  """Gets the assembly filename for the given source file."""
  return substitute_file_extension(source_filename, 'spvasm')


def verify_file_non_empty(filename):
  """Checks that a given file exists and is not empty."""
  if not os.path.isfile(filename):
    return False, 'Cannot find file: ' + filename
  if not os.path.getsize(filename):
    return False, 'Empty file: ' + filename
  return True, ''


class ReturnCodeIsZero(SpirvTest):
  """Mixin class for checking that the return code is zero."""

  def check_return_code_is_zero(self, status):
    if status.returncode:
      return False, 'Non-zero return code: {ret}\n'.format(
          ret=status.returncode)
    return True, ''


class NoOutputOnStdout(SpirvTest):
  """Mixin class for checking that there is no output on stdout."""

  def check_no_output_on_stdout(self, status):
    if status.stdout:
      return False, 'Non empty stdout: {out}\n'.format(out=status.stdout)
    return True, ''


class NoOutputOnStderr(SpirvTest):
  """Mixin class for checking that there is no output on stderr."""

  def check_no_output_on_stderr(self, status):
    if status.stderr:
      return False, 'Non empty stderr: {err}\n'.format(err=status.stderr)
    return True, ''


class SuccessfulReturn(ReturnCodeIsZero, NoOutputOnStdout, NoOutputOnStderr):
  """Mixin class for checking that return code is zero and no output on
    stdout and stderr."""
  pass


class NoGeneratedFiles(SpirvTest):
  """Mixin class for checking that there is no file generated."""

  def check_no_generated_files(self, status):
    all_files = os.listdir(status.directory)
    input_files = status.input_filenames
    if all([f.startswith(status.directory) for f in input_files]):
      all_files = [os.path.join(status.directory, f) for f in all_files]
    generated_files = set(all_files) - set(input_files)
    if len(generated_files) == 0:
      return True, ''
    else:
      return False, 'Extra files generated: {}'.format(generated_files)


class CorrectBinaryLengthAndPreamble(SpirvTest):
  """Provides methods for verifying preamble for a SPIR-V binary."""

  def verify_binary_length_and_header(self, binary, spv_version=0x10000):
    """Checks that the given SPIR-V binary has valid length and header.

        Returns:
            False, error string if anything is invalid
            True, '' otherwise
        Args:
            binary: a bytes object containing the SPIR-V binary
            spv_version: target SPIR-V version number, with same encoding
                 as the version word in a SPIR-V header.
        """

    def read_word(binary, index, little_endian):
      """Reads the index-th word from the given binary file."""
      word = binary[index * 4:(index + 1) * 4]
      if little_endian:
        word = reversed(word)
      return functools.reduce(lambda w, b: (w << 8) | b, word, 0)

    def check_endianness(binary):
      """Checks the endianness of the given SPIR-V binary.

            Returns:
              True if it's little endian, False if it's big endian.
              None if magic number is wrong.
            """
      first_word = read_word(binary, 0, True)
      if first_word == 0x07230203:
        return True
      first_word = read_word(binary, 0, False)
      if first_word == 0x07230203:
        return False
      return None

    num_bytes = len(binary)
    if num_bytes % 4 != 0:
      return False, ('Incorrect SPV binary: size should be a multiple'
                     ' of words')
    if num_bytes < 20:
      return False, 'Incorrect SPV binary: size less than 5 words'

    preamble = binary[0:19]
    little_endian = check_endianness(preamble)
    # SPIR-V module magic number
    if little_endian is None:
      return False, 'Incorrect SPV binary: wrong magic number'

    # SPIR-V version number
    version = read_word(preamble, 1, little_endian)
    # TODO(dneto): Recent Glslang uses version word 0 for opengl_compat
    # profile

    if version != spv_version and version != 0:
      return False, 'Incorrect SPV binary: wrong version number'
    # Shaderc-over-Glslang (0x000d....) or
    # SPIRV-Tools (0x0007....) generator number
    if read_word(preamble, 2, little_endian) != 0x000d0007 and \
            read_word(preamble, 2, little_endian) != 0x00070000:
      return False, ('Incorrect SPV binary: wrong generator magic ' 'number')
    # reserved for instruction schema
    if read_word(preamble, 4, little_endian) != 0:
      return False, 'Incorrect SPV binary: the 5th byte should be 0'

    return True, ''


class CorrectObjectFilePreamble(CorrectBinaryLengthAndPreamble):
  """Provides methods for verifying preamble for a SPV object file."""

  def verify_object_file_preamble(self, filename, spv_version=0x10000):
    """Checks that the given SPIR-V binary file has correct preamble."""

    success, message = verify_file_non_empty(filename)
    if not success:
      return False, message

    with open(filename, 'rb') as object_file:
      object_file.seek(0, os.SEEK_END)
      num_bytes = object_file.tell()

      object_file.seek(0)

      binary = bytes(object_file.read())
      return self.verify_binary_length_and_header(binary, spv_version)

    return True, ''


class CorrectAssemblyFilePreamble(SpirvTest):
  """Provides methods for verifying preamble for a SPV assembly file."""

  def verify_assembly_file_preamble(self, filename):
    success, message = verify_file_non_empty(filename)
    if not success:
      return False, message

    with open(filename) as assembly_file:
      line1 = assembly_file.readline()
      line2 = assembly_file.readline()
      line3 = assembly_file.readline()

    if (line1 != '; SPIR-V\n' or line2 != '; Version: 1.0\n' or
        (not line3.startswith('; Generator: Google Shaderc over Glslang;'))):
      return False, 'Incorrect SPV assembly'

    return True, ''


class ValidObjectFile(SuccessfulReturn, CorrectObjectFilePreamble):
  """Mixin class for checking that every input file generates a valid SPIR-V 1.0
    object file following the object file naming rule, and there is no output on
    stdout/stderr."""

  def check_object_file_preamble(self, status):
    for input_filename in status.input_filenames:
      object_filename = get_object_filename(input_filename)
      success, message = self.verify_object_file_preamble(
          os.path.join(status.directory, object_filename))
      if not success:
        return False, message
    return True, ''


class ValidObjectFile1_3(ReturnCodeIsZero, CorrectObjectFilePreamble):
  """Mixin class for checking that every input file generates a valid SPIR-V 1.3
    object file following the object file naming rule, and there is no output on
    stdout/stderr."""

  def check_object_file_preamble(self, status):
    for input_filename in status.input_filenames:
      object_filename = get_object_filename(input_filename)
      success, message = self.verify_object_file_preamble(
          os.path.join(status.directory, object_filename), 0x10300)
      if not success:
        return False, message
    return True, ''


class ValidObjectFileWithAssemblySubstr(SuccessfulReturn,
                                        CorrectObjectFilePreamble):
  """Mixin class for checking that every input file generates a valid object

    file following the object file naming rule, there is no output on
    stdout/stderr, and the disassmbly contains a specified substring per
    input.
  """

  def check_object_file_disassembly(self, status):
    for an_input in status.inputs:
      object_filename = get_object_filename(an_input.filename)
      obj_file = str(os.path.join(status.directory, object_filename))
      success, message = self.verify_object_file_preamble(obj_file)
      if not success:
        return False, message
      cmd = [status.test_manager.disassembler_path, '--no-color', obj_file]
      process = subprocess.Popen(
          args=cmd,
          stdin=subprocess.PIPE,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          cwd=status.directory)
      output = process.communicate(None)
      disassembly = output[0]
      if not isinstance(an_input.assembly_substr, str):
        return False, 'Missing assembly_substr member'
      if an_input.assembly_substr not in disassembly:
        return False, ('Incorrect disassembly output:\n{asm}\n'
                       'Expected substring not found:\n{exp}'.format(
                           asm=disassembly, exp=an_input.assembly_substr))
    return True, ''


class ValidNamedObjectFile(SuccessfulReturn, CorrectObjectFilePreamble):
  """Mixin class for checking that a list of object files with the given
    names are correctly generated, and there is no output on stdout/stderr.

    To mix in this class, subclasses need to provide expected_object_filenames
    as the expected object filenames.
    """

  def check_object_file_preamble(self, status):
    for object_filename in self.expected_object_filenames:
      success, message = self.verify_object_file_preamble(
          os.path.join(status.directory, object_filename))
      if not success:
        return False, message
    return True, ''


class ValidFileContents(SpirvTest):
  """Mixin class to test that a specific file contains specific text
    To mix in this class, subclasses need to provide expected_file_contents as
    the contents of the file and target_filename to determine the location."""

  def check_file(self, status):
    target_filename = os.path.join(status.directory, self.target_filename)
    if not os.path.isfile(target_filename):
      return False, 'Cannot find file: ' + target_filename
    with open(target_filename, 'r') as target_file:
      file_contents = target_file.read()
      if isinstance(self.expected_file_contents, str):
        if file_contents == self.expected_file_contents:
          return True, ''
        return False, ('Incorrect file output: \n{act}\n'
                       'Expected:\n{exp}'
                       'With diff:\n{diff}'.format(
                           act=file_contents,
                           exp=self.expected_file_contents,
                           diff='\n'.join(
                               list(
                                   difflib.unified_diff(
                                       self.expected_file_contents.split('\n'),
                                       file_contents.split('\n'),
                                       fromfile='expected_output',
                                       tofile='actual_output')))))
      elif isinstance(self.expected_file_contents, type(re.compile(''))):
        if self.expected_file_contents.search(file_contents):
          return True, ''
        return False, ('Incorrect file output: \n{act}\n'
                       'Expected matching regex pattern:\n{exp}'.format(
                           act=file_contents,
                           exp=self.expected_file_contents.pattern))
    return False, (
        'Could not open target file ' + target_filename + ' for reading')


class ValidAssemblyFile(SuccessfulReturn, CorrectAssemblyFilePreamble):
  """Mixin class for checking that every input file generates a valid assembly
    file following the assembly file naming rule, and there is no output on
    stdout/stderr."""

  def check_assembly_file_preamble(self, status):
    for input_filename in status.input_filenames:
      assembly_filename = get_assembly_filename(input_filename)
      success, message = self.verify_assembly_file_preamble(
          os.path.join(status.directory, assembly_filename))
      if not success:
        return False, message
    return True, ''


class ValidAssemblyFileWithSubstr(ValidAssemblyFile):
  """Mixin class for checking that every input file generates a valid assembly
    file following the assembly file naming rule, there is no output on
    stdout/stderr, and all assembly files have the given substring specified
    by expected_assembly_substr.

    To mix in this class, subclasses need to provde expected_assembly_substr
    as the expected substring.
    """

  def check_assembly_with_substr(self, status):
    for input_filename in status.input_filenames:
      assembly_filename = get_assembly_filename(input_filename)
      success, message = self.verify_assembly_file_preamble(
          os.path.join(status.directory, assembly_filename))
      if not success:
        return False, message
      with open(assembly_filename, 'r') as f:
        content = f.read()
        if self.expected_assembly_substr not in convert_to_unix_line_endings(
            content):
          return False, ('Incorrect assembly output:\n{asm}\n'
                         'Expected substring not found:\n{exp}'.format(
                             asm=content, exp=self.expected_assembly_substr))
    return True, ''


class ValidAssemblyFileWithoutSubstr(ValidAssemblyFile):
  """Mixin class for checking that every input file generates a valid assembly
    file following the assembly file naming rule, there is no output on
    stdout/stderr, and no assembly files have the given substring specified
    by unexpected_assembly_substr.

    To mix in this class, subclasses need to provde unexpected_assembly_substr
    as the substring we expect not to see.
    """

  def check_assembly_for_substr(self, status):
    for input_filename in status.input_filenames:
      assembly_filename = get_assembly_filename(input_filename)
      success, message = self.verify_assembly_file_preamble(
          os.path.join(status.directory, assembly_filename))
      if not success:
        return False, message
      with open(assembly_filename, 'r') as f:
        content = f.read()
        if self.unexpected_assembly_substr in convert_to_unix_line_endings(
            content):
          return False, ('Incorrect assembly output:\n{asm}\n'
                         'Unexpected substring found:\n{unexp}'.format(
                             asm=content, exp=self.unexpected_assembly_substr))
    return True, ''


class ValidNamedAssemblyFile(SuccessfulReturn, CorrectAssemblyFilePreamble):
  """Mixin class for checking that a list of assembly files with the given
    names are correctly generated, and there is no output on stdout/stderr.

    To mix in this class, subclasses need to provide expected_assembly_filenames
    as the expected assembly filenames.
    """

  def check_object_file_preamble(self, status):
    for assembly_filename in self.expected_assembly_filenames:
      success, message = self.verify_assembly_file_preamble(
          os.path.join(status.directory, assembly_filename))
      if not success:
        return False, message
    return True, ''


class ErrorMessage(SpirvTest):
  """Mixin class for tests that fail with a specific error message.

    To mix in this class, subclasses need to provide expected_error as the
    expected error message.

    The test should fail if the subprocess was terminated by a signal.
    """

  def check_has_error_message(self, status):
    if not status.returncode:
      return False, ('Expected error message, but returned success from '
                     'command execution')
    if status.returncode < 0:
      # On Unix, a negative value -N for Popen.returncode indicates
      # termination by signal N.
      # https://docs.python.org/2/library/subprocess.html
      return False, ('Expected error message, but command was terminated by '
                     'signal ' + str(status.returncode))
    if not status.stderr:
      return False, 'Expected error message, but no output on stderr'
    if self.expected_error != convert_to_unix_line_endings(status.stderr):
      return False, ('Incorrect stderr output:\n{act}\n'
                     'Expected:\n{exp}'.format(
                         act=status.stderr, exp=self.expected_error))
    return True, ''


class ErrorMessageSubstr(SpirvTest):
  """Mixin class for tests that fail with a specific substring in the error
    message.

    To mix in this class, subclasses need to provide expected_error_substr as
    the expected error message substring.

    The test should fail if the subprocess was terminated by a signal.
    """

  def check_has_error_message_as_substring(self, status):
    if not status.returncode:
      return False, ('Expected error message, but returned success from '
                     'command execution')
    if status.returncode < 0:
      # On Unix, a negative value -N for Popen.returncode indicates
      # termination by signal N.
      # https://docs.python.org/2/library/subprocess.html
      return False, ('Expected error message, but command was terminated by '
                     'signal ' + str(status.returncode))
    if not status.stderr:
      return False, 'Expected error message, but no output on stderr'
    if self.expected_error_substr not in convert_to_unix_line_endings(
        status.stderr.decode('utf8')):
      return False, ('Incorrect stderr output:\n{act}\n'
                     'Expected substring not found in stderr:\n{exp}'.format(
                         act=status.stderr, exp=self.expected_error_substr))
    return True, ''


class WarningMessage(SpirvTest):
  """Mixin class for tests that succeed but have a specific warning message.

    To mix in this class, subclasses need to provide expected_warning as the
    expected warning message.
    """

  def check_has_warning_message(self, status):
    if status.returncode:
      return False, ('Expected warning message, but returned failure from'
                     ' command execution')
    if not status.stderr:
      return False, 'Expected warning message, but no output on stderr'
    if self.expected_warning != convert_to_unix_line_endings(status.stderr.decode('utf8')):
      return False, ('Incorrect stderr output:\n{act}\n'
                     'Expected:\n{exp}'.format(
                         act=status.stderr, exp=self.expected_warning))
    return True, ''


class ValidObjectFileWithWarning(NoOutputOnStdout, CorrectObjectFilePreamble,
                                 WarningMessage):
  """Mixin class for checking that every input file generates a valid object
    file following the object file naming rule, with a specific warning message.
    """

  def check_object_file_preamble(self, status):
    for input_filename in status.input_filenames:
      object_filename = get_object_filename(input_filename)
      success, message = self.verify_object_file_preamble(
          os.path.join(status.directory, object_filename))
      if not success:
        return False, message
    return True, ''


class ValidAssemblyFileWithWarning(NoOutputOnStdout,
                                   CorrectAssemblyFilePreamble, WarningMessage):
  """Mixin class for checking that every input file generates a valid assembly
    file following the assembly file naming rule, with a specific warning
    message."""

  def check_assembly_file_preamble(self, status):
    for input_filename in status.input_filenames:
      assembly_filename = get_assembly_filename(input_filename)
      success, message = self.verify_assembly_file_preamble(
          os.path.join(status.directory, assembly_filename))
      if not success:
        return False, message
    return True, ''


class StdoutMatch(SpirvTest):
  """Mixin class for tests that can expect output on stdout.

    To mix in this class, subclasses need to provide expected_stdout as the
    expected stdout output.

    For expected_stdout, if it's True, then they expect something on stdout but
    will not check what it is. If it's a string, expect an exact match.  If it's
    anything else, it is assumed to be a compiled regular expression which will
    be matched against re.search(). It will expect
    expected_stdout.search(status.stdout) to be true.
    """

  def check_stdout_match(self, status):
    # "True" in this case means we expect something on stdout, but we do not
    # care what it is, we want to distinguish this from "blah" which means we
    # expect exactly the string "blah".
    if self.expected_stdout is True:
      if not status.stdout:
        return False, 'Expected something on stdout'
    elif type(self.expected_stdout) == str:
      if self.expected_stdout != convert_to_unix_line_endings(status.stdout.decode('utf8')):
        return False, ('Incorrect stdout output:\n{ac}\n'
                       'Expected:\n{ex}'.format(
                           ac=status.stdout, ex=self.expected_stdout))
    else:
      converted = convert_to_unix_line_endings(status.stdout.decode('utf8'))
      if not self.expected_stdout.search(converted):
        return False, ('Incorrect stdout output:\n{ac}\n'
                       'Expected to match regex:\n{ex}'.format(
                           ac=status.stdout.decode('utf8'), ex=self.expected_stdout.pattern))
    return True, ''


class StderrMatch(SpirvTest):
  """Mixin class for tests that can expect output on stderr.

    To mix in this class, subclasses need to provide expected_stderr as the
    expected stderr output.

    For expected_stderr, if it's True, then they expect something on stderr,
    but will not check what it is. If it's a string, expect an exact match.
    If it's anything else, it is assumed to be a compiled regular expression
    which will be matched against re.search(). It will expect
    expected_stderr.search(status.stderr) to be true.
    """

  def check_stderr_match(self, status):
    # "True" in this case means we expect something on stderr, but we do not
    # care what it is, we want to distinguish this from "blah" which means we
    # expect exactly the string "blah".
    if self.expected_stderr is True:
      if not status.stderr:
        return False, 'Expected something on stderr'
    elif type(self.expected_stderr) == str:
      if self.expected_stderr != convert_to_unix_line_endings(status.stderr.decode('utf8')):
        return False, ('Incorrect stderr output:\n{ac}\n'
                       'Expected:\n{ex}'.format(
                           ac=status.stderr, ex=self.expected_stderr))
    else:
      if not self.expected_stderr.search(
          convert_to_unix_line_endings(status.stderr.decode('utf8'))):
        return False, ('Incorrect stderr output:\n{ac}\n'
                       'Expected to match regex:\n{ex}'.format(
                           ac=status.stderr, ex=self.expected_stderr.pattern))
    return True, ''


class StdoutNoWiderThan80Columns(SpirvTest):
  """Mixin class for tests that require stdout to 80 characters or narrower.

    To mix in this class, subclasses need to provide expected_stdout as the
    expected stdout output.
    """

  def check_stdout_not_too_wide(self, status):
    if not status.stdout:
      return True, ''
    else:
      for line in status.stdout.splitlines():
        if len(line) > 80:
          return False, ('Stdout line longer than 80 columns: %s' % line)
    return True, ''


class NoObjectFile(SpirvTest):
  """Mixin class for checking that no input file has a corresponding object
    file."""

  def check_no_object_file(self, status):
    for input_filename in status.input_filenames:
      object_filename = get_object_filename(input_filename)
      full_object_file = os.path.join(status.directory, object_filename)
      print('checking %s' % full_object_file)
      if os.path.isfile(full_object_file):
        return False, (
            'Expected no object file, but found: %s' % full_object_file)
    return True, ''


class NoNamedOutputFiles(SpirvTest):
  """Mixin class for checking that no specified output files exist.

    The expected_output_filenames member should be full pathnames."""

  def check_no_named_output_files(self, status):
    for object_filename in self.expected_output_filenames:
      if os.path.isfile(object_filename):
        return False, (
            'Expected no output file, but found: %s' % object_filename)
    return True, ''


class ExecutedListOfPasses(SpirvTest):
  """Mixin class for checking that a list of passes where executed.

  It works by analyzing the output of the --print-all flag to spirv-opt.

  For this mixin to work, the class member expected_passes should be a sequence
  of pass names as returned by Pass::name().
  """

  def check_list_of_executed_passes(self, status):
    # Collect all the output lines containing a pass name.
    pass_names = []
    pass_name_re = re.compile(r'.*IR before pass (?P<pass_name>[\S]+)')
    for line in status.stderr.decode('utf8').splitlines():
      match = pass_name_re.match(line)
      if match:
        pass_names.append(match.group('pass_name'))

    for (expected, actual) in zip(self.expected_passes, pass_names):
      if expected != actual:
        return False, (
            'Expected pass "%s" but found pass "%s"\n' % (expected, actual))

    return True, ''
