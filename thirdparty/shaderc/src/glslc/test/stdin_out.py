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
from glslc_test_framework import inside_glslc_testsuite
from placeholder import FileShader, StdinShader


@inside_glslc_testsuite('StdInOut')
class VerifyStdinWorks(expect.ValidObjectFile):
    """Tests glslc accepts vertex shader extension (.vert)."""

    shader = StdinShader('#version 140\nvoid main() { }')
    glslc_args = ['-c', '-fshader-stage=vertex', shader]


@inside_glslc_testsuite('StdInOut')
class VerifyStdoutWorks(
    expect.ReturnCodeIsZero, expect.StdoutMatch, expect.StderrMatch):

    shader = FileShader('#version 140\nvoid main() {}', '.vert')
    glslc_args = [shader, '-o', '-']

    # We expect SOME stdout, we just do not care what.
    expected_stdout = True
    expected_stderr = ''
