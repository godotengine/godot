# Copyright 2016 The Shaderc Authors. All rights reserved.
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
import re
from glslc_test_framework import inside_glslc_testsuite
from placeholder import FileShader

MINIMAL_SHADER = '#version 140\nvoid main() {}'
# Regular expression patterns for the minimal shader. The magic number should
# match exactly, and there should not be a trailing comma at the end of the
# list. When -mfmt=c is specified, curly brackets should be presented.
MINIMAL_SHADER_NUM_FORMAT_PATTERN = "^0x07230203.*[0-9a-f]$"
MINIMAL_SHADER_C_FORMAT_PATTERN = "^\{0x07230203.*[0-9a-f]\}"
ERROR_SHADER = '#version 140\n#error\nvoid main() {}'


@inside_glslc_testsuite('OptionMfmt')
class TestFmtCWorksWithDashC(expect.ValidFileContents):
    """Tests that -mfmt=c works with -c for single input file. SPIR-V binary
    code output should be emitted as a C-style initializer list in the output
    file.
    """
    shader = FileShader(MINIMAL_SHADER, '.vert')
    glslc_args = [shader, '-c', '-mfmt=c', '-o', 'output_file']
    target_filename = 'output_file'
    expected_file_contents = re.compile(MINIMAL_SHADER_C_FORMAT_PATTERN, re.S)


@inside_glslc_testsuite('OptionMfmt')
class TestFmtNumWorksWithDashC(expect.ValidFileContents):
    """Tests that -mfmt=num works with -c for single input file. SPIR-V binary
    code output should be emitted as a list of hex numbers in the output file.
    """
    shader = FileShader(MINIMAL_SHADER, '.vert')
    glslc_args = [shader, '-c', '-mfmt=num', '-o', 'output_file']
    target_filename = 'output_file'
    expected_file_contents = re.compile(MINIMAL_SHADER_NUM_FORMAT_PATTERN, re.S)


@inside_glslc_testsuite('OptionMfmt')
class TestFmtBinWorksWithDashC(expect.ValidObjectFile):
    """Tests that -mfmt=bin works with -c for single input file. This test
    should simply have the SPIR-V binary generated.
    """
    shader = FileShader(MINIMAL_SHADER, '.vert')
    glslc_args = [shader, '-c', '-mfmt=bin']


@inside_glslc_testsuite('OptionMfmt')
class TestFmtCWithLinking(expect.ValidFileContents):
    """Tests that -mfmt=c works when linkding is enabled (no -c specified).
    SPIR-V binary code should be emitted as a C-style initializer list in the
    output file.
    """
    shader = FileShader(MINIMAL_SHADER, '.vert')
    glslc_args = [shader, '-mfmt=c']
    target_filename = 'a.spv'
    expected_file_contents = re.compile(MINIMAL_SHADER_C_FORMAT_PATTERN, re.S)


@inside_glslc_testsuite('OptionMfmt')
class TestFmtNumWithLinking(expect.ValidFileContents):
    """Tests that -mfmt=num works when linkding is enabled (no -c specified).
    SPIR-V binary code should be emitted as a C-style initializer list in the
    output file.
    """
    shader = FileShader(MINIMAL_SHADER, '.vert')
    glslc_args = [shader, '-mfmt=num']
    target_filename = 'a.spv'
    expected_file_contents = re.compile(MINIMAL_SHADER_NUM_FORMAT_PATTERN, re.S)


@inside_glslc_testsuite('OptionMfmt')
class TestFmtCErrorWhenOutputDisasembly(expect.ErrorMessage):
    """Tests that specifying '-mfmt=c' when the compiler is set to
    disassembly mode should trigger an error.
    """
    shader = FileShader(MINIMAL_SHADER, '.vert')
    glslc_args = [shader, '-mfmt=c', '-S', '-o', 'output_file']
    expected_error = ("glslc: error: cannot emit output as a C-style "
                      "initializer list when the output is not SPIR-V "
                      "binary code\n")


@inside_glslc_testsuite('OptionMfmt')
class TestFmtNumErrorWhenOutputDisasembly(expect.ErrorMessage):
    """Tests that specifying '-mfmt=num' when the compiler is set to
    disassembly mode should trigger an error.
    """
    shader = FileShader(MINIMAL_SHADER, '.vert')
    glslc_args = [shader, '-mfmt=num', '-S', '-o', 'output_file']
    expected_error = (
        "glslc: error: cannot emit output as a list of hex numbers "
        "when the output is not SPIR-V binary code\n")


@inside_glslc_testsuite('OptionMfmt')
class TestFmtBinErrorWhenOutputDisasembly(expect.ErrorMessage):
    """Tests that specifying '-mfmt=bin' when the compiler is set to
    disassembly mode should trigger an error.
    """
    shader = FileShader(MINIMAL_SHADER, '.vert')
    glslc_args = [shader, '-mfmt=bin', '-S', '-o', 'output_file']
    expected_error = ("glslc: error: cannot emit output as a binary "
                      "when the output is not SPIR-V binary code\n")


@inside_glslc_testsuite('OptionMfmt')
class TestFmtNumErrorWhenOutputPreprocess(expect.ErrorMessage):
    """Tests that specifying '-mfmt=num' when the compiler is set to
    preprocessing only mode should trigger an error.
    """
    shader = FileShader(MINIMAL_SHADER, '.vert')
    glslc_args = [shader, '-mfmt=num', '-E', '-o', 'output_file']
    expected_error = (
        "glslc: error: cannot emit output as a list of hex numbers "
        "when the output is not SPIR-V binary code\n")


@inside_glslc_testsuite('OptionMfmt')
class TestFmtCErrorWithDashCapM(expect.ErrorMessage):
    """Tests that specifying '-mfmt=c' should trigger an error when the
    compiler is set to dump dependency info as the output (-M or -MM is
    specified).
    """
    shader = FileShader(MINIMAL_SHADER, '.vert')
    glslc_args = [shader, '-mfmt=c', '-M', '-o', 'output_file']
    expected_error = ("glslc: error: cannot emit output as a C-style "
                      "initializer list when the output is not SPIR-V "
                      "binary code\n")


@inside_glslc_testsuite('OptionMfmt')
class TestFmtCWorksWithDashCapMD(expect.ValidFileContents):
    """Tests that -mfmt=c works with '-c -MD'. SPIR-V binary code
    should be emitted as a C-style initializer list in the output file.
    """
    shader = FileShader(MINIMAL_SHADER, '.vert')
    glslc_args = [shader, '-mfmt=c', '-c', '-MD', '-o', 'output_file']
    target_filename = 'output_file'
    expected_file_contents = re.compile(MINIMAL_SHADER_C_FORMAT_PATTERN, re.S)


@inside_glslc_testsuite('OptionMfmt')
class TestFmtNumWorksWithDashCapMD(expect.ValidFileContents):
    """Tests that -mfmt=num works with '-c -MD'. SPIR-V binary code
    should be emitted as a C-style initializer list in the output file.
    """
    shader = FileShader(MINIMAL_SHADER, '.vert')
    glslc_args = [shader, '-mfmt=num', '-c', '-MD', '-o', 'output_file']
    target_filename = 'output_file'
    expected_file_contents = re.compile(MINIMAL_SHADER_NUM_FORMAT_PATTERN, re.S)


@inside_glslc_testsuite('OptionMfmt')
class TestFmtCExitsElegantlyWithErrorInShader(expect.ErrorMessage):
    """Tests that the compiler fails elegantly with -mfmt=c when there are
    errors in the input shader.
    """
    shader = FileShader(ERROR_SHADER, '.vert')
    glslc_args = [shader, '-mfmt=c']
    expected_error = [shader, ':3: error: \'#error\' :\n',
                      '1 error generated.\n']


@inside_glslc_testsuite('OptionMfmt')
class TestFmtNumExitsElegantlyWithErrorInShader(expect.ErrorMessage):
    """Tests that the compiler fails elegantly with -mfmt=num when there are
    errors in the input shader.
    """
    shader = FileShader(ERROR_SHADER, '.vert')
    glslc_args = [shader, '-mfmt=num']
    expected_error = [shader, ':3: error: \'#error\' :\n',
                      '1 error generated.\n']


@inside_glslc_testsuite('OptionMfmt')
class TestFmtBinExitsElegantlyWithErrorInShader(expect.ErrorMessage):
    """Tests that the compiler fails elegantly with -mfmt=binary when there are
    errors in the input shader.
    """
    shader = FileShader(ERROR_SHADER, '.vert')
    glslc_args = [shader, '-mfmt=bin']
    expected_error = [shader, ':3: error: \'#error\' :\n',
                      '1 error generated.\n']
