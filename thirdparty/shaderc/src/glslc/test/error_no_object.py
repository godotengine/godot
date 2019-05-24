# Copyright 2017 The Shaderc Authors. All rights reserved.
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
from placeholder import FileShader


@inside_glslc_testsuite('ErrorNoObject')
class ErrorGeneratesNoObjectFile(expect.NoObjectFile,
                                 expect.NoOutputOnStdout,
                                 expect.ErrorMessageSubstr):
    """Tests that on error, no object file is generated."""

    shader = FileShader('#version 150\nBad', '.frag')
    glslc_args = ['-c', shader]
    expected_error_substr = ['syntax error']


@inside_glslc_testsuite('ErrorNoObject')
class FailureToMakeOutputFileIsErrorWithNoOutputFile(
         expect.NoNamedOutputFiles,
         expect.NoOutputOnStdout,
         expect.ErrorMessageSubstr):
    """Tests that if we fail to make an output file, no file is generated,
    and we have certain error messages."""

    shader = FileShader('#version 150\nvoid main() {}', '.frag')
    bad_file = '/file/should/not/exist/today'
    glslc_args = ['-c', shader, '-o', bad_file]
    expected_output_filenames = [bad_file]
    expected_error_substr = ['cannot open output file']


@inside_glslc_testsuite('ErrorNoObject')
class FailureToMakeOutputFileAsCurrentDirIsErrorWithNoOutputFile(
         expect.NoNamedOutputFiles,
         expect.NoOutputOnStdout,
         expect.ErrorMessageSubstr):
    """Tests that if we fail to make an output file because it is the current
    directory, then no file is generated, and we have certain error messages."""

    shader = FileShader('#version 150\nvoid main() {}', '.frag')
    bad_file = '.'  # Current directory
    glslc_args = ['-c', shader, '-o', bad_file]
    expected_output_filenames = [bad_file]
    expected_error_substr = ['cannot open output file']
