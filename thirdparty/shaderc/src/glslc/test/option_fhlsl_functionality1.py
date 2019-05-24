# Copyright 2018 The Shaderc Authors. All rights reserved.
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

# An HLSL shader with a counter buffer with a counter increment.
# Glslang doesn't automatically assign a binding to the counter, and
# it doesn't understand [[vk::counter_binding(n)]], so compile this
# with --auto-bind-uniforms.
# See https://github.com/KhronosGroup/glslang/issues/1616
HLSL_VERTEX_SHADER_WITH_COUNTER_BUFFER = """
RWStructuredBuffer<int> Ainc;
float4 main() : SV_Target0 {
  return float4(Ainc.IncrementCounter(), 0, 1, 2);
}
"""


@inside_glslc_testsuite('OptionFHlslFunctionality1')
class TestHlslFunctionality1MentionsExtension(expect.ValidAssemblyFileWithSubstr):
    """Tests that -fhlsl_functionality1 enabled SPV_GOOGLE_hlsl_functionality1."""

    shader = FileShader(HLSL_VERTEX_SHADER_WITH_COUNTER_BUFFER, '.frag')
    glslc_args = ['-S', '-x', 'hlsl', '-fhlsl_functionality1',
                  '-fauto-bind-uniforms', shader]
    expected_assembly_substr = 'OpExtension "SPV_GOOGLE_hlsl_functionality1"'


@inside_glslc_testsuite('OptionFHlslFunctionality1')
class TestHlslFunctionality1DecoratesCounter(expect.ValidAssemblyFileWithSubstr):
    """Tests that -fhlsl_functionality1 decorates the output target"""

    shader = FileShader(HLSL_VERTEX_SHADER_WITH_COUNTER_BUFFER, '.frag')
    glslc_args = ['-S', '-x', 'hlsl', '-fhlsl_functionality1',
                  '-fauto-bind-uniforms', shader]
    expected_assembly_substr = 'OpDecorateStringGOOGLE'


## Next tests use the option with the hypen instead of underscore.

@inside_glslc_testsuite('OptionFHlslFunctionality1')
class TestHlslHyphenFunctionality1MentionsExtension(expect.ValidAssemblyFileWithSubstr):
    """Tests that -fhlsl-functionality1 enabled SPV_GOOGLE_hlsl_functionality1."""

    shader = FileShader(HLSL_VERTEX_SHADER_WITH_COUNTER_BUFFER, '.frag')
    glslc_args = ['-S', '-x', 'hlsl', '-fhlsl_functionality1',
                  '-fauto-bind-uniforms', shader]
    expected_assembly_substr = 'OpExtension "SPV_GOOGLE_hlsl_functionality1"'


@inside_glslc_testsuite('OptionFHlslFunctionality1')
class TestHlslHyphenFunctionality1DecoratesCounter(expect.ValidAssemblyFileWithSubstr):
    """Tests that -fhlsl-functionality1 decorates the output target"""

    shader = FileShader(HLSL_VERTEX_SHADER_WITH_COUNTER_BUFFER, '.frag')
    glslc_args = ['-S', '-x', 'hlsl', '-fhlsl_functionality1',
                  '-fauto-bind-uniforms', shader]
    expected_assembly_substr = 'OpDecorateStringGOOGLE'
