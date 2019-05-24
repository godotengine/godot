# Copyright 2015 The Shaderc Authors. All rights reserved.
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

import expect
import os.path
from glslc_test_framework import inside_glslc_testsuite
from placeholder import FileShader, TempFileName
from builtins import bytes


@inside_glslc_testsuite('OptionDashO')
class TestOptionDashOConcatenatedArg(expect.SuccessfulReturn,
                                     expect.CorrectObjectFilePreamble):
    """Tests that we can concatenate -o and the output filename."""

    shader = FileShader('#version 140\nvoid main() {}', '.vert')
    glslc_args = ['-ofoo', shader]

    def check_output_foo(self, status):
        output_name = os.path.join(status.directory, 'foo')
        return self.verify_object_file_preamble(output_name)


@inside_glslc_testsuite('OptionDashO')
class ManyOutputFilesWithDashO(expect.ErrorMessage):
    """Tests -o and -c with several files generates an error."""

    shader1 = FileShader('', '.vert')
    shader2 = FileShader('', '.frag')
    glslc_args = ['-o', 'foo', '-c', shader1, shader2]
    expected_error = [
        'glslc: error: cannot specify -o when '
        'generating multiple output files\n']


@inside_glslc_testsuite('OptionDashO')
class OutputFileLocation(expect.SuccessfulReturn,
                         expect.CorrectObjectFilePreamble):
    """Tests that the -o flag puts a file in a new location."""

    shader = FileShader('#version 310 es\nvoid main() {}', '.frag')
    glslc_args = [shader, '-o', TempFileName('a.out')]

    def check_output_a_out(self, status):
        output_name = os.path.join(status.directory, 'a.out')
        return self.verify_object_file_preamble(output_name)


@inside_glslc_testsuite('OptionDashO')
class DashOMissingArgumentIsAnError(expect.ErrorMessage):
    """Tests that -o without an argument is an error."""

    glslc_args = ['-o']
    expected_error = ['glslc: error: argument to \'-o\' is missing ' +
                      '(expected 1 value)\n']


@inside_glslc_testsuite('OptionDashO')
class OutputFileBinaryAvoidsCRLFTranslation(expect.ReturnCodeIsZero,
                                            expect.NoOutputOnStderr,
                                            expect.NoGeneratedFiles,
                                            expect.CorrectBinaryLengthAndPreamble):
    """Tests that the -o flag emits a binary file without CR/LF translation.
    """

    # A shader whose compiled output has three bytes that are newlines.
    # If the output stream converts the newlines to CR/LF, then we end up
    # with a file that is 4k + 3 bytes long.  That will be caught by the
    # object file checks.
    SHADER_WITH_THREE_NEWLINES_IN_BINARY = """#version 450
       layout(location = 0) out uint ovar;
       void main() { ovar = 10; }
    """

    shader = FileShader(SHADER_WITH_THREE_NEWLINES_IN_BINARY, '.vert')
    glslc_args = [shader, '-o', '-']

    def check_stdout_binary(self, status):
        binary = bytes(status.stdout)
        newlines = [x for x in binary if x == ord('\n')]
        num_newlines = len(newlines)
        if num_newlines % 4 == 0:
            return False, "Bad test. Need nontrivial number of newlines"
        if num_newlines != 3:
            return False, ("Update this test. Expected 3 newlines in the "
                           "binary, but found {}").format(num_newlines)
        return self.verify_binary_length_and_header(bytes(status.stdout))
