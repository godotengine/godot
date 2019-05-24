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
from glslc_test_framework import inside_glslc_testsuite
from placeholder import FileShader


# A GLSL shader with uniforms without explicit bindings.
GLSL_SHADER = """#version 450
  buffer B { float x; vec3 y; } my_ssbo;
  void main() {
    my_ssbo.x = 1.0;
  }"""


@inside_glslc_testsuite('OptionFHlslOffsets')
class StandardOffsetsByDefault(expect.ValidAssemblyFileWithSubstr):
    """Tests that standard GLSL packign is used by default."""

    shader = FileShader(GLSL_SHADER, '.vert')
    glslc_args = ['-S', shader]
    expected_assembly_substr = "OpMemberDecorate %B 1 Offset 16"


@inside_glslc_testsuite('OptionFHlslOffsets')
class HlslOffsetsWhenRequested(expect.ValidAssemblyFileWithSubstr):
    """Tests that standard GLSL packign is used by default."""

    shader = FileShader(GLSL_SHADER, '.vert')
    glslc_args = ['-S', '-fhlsl-offsets', shader]
    expected_assembly_substr = "OpMemberDecorate %B 1 Offset 4"
