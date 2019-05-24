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

# An HLSL shader with uniforms without explicit bindings.
HLSL_SHADER = """
Buffer<float4> t4 : register(t4);
Buffer<float4> t5 : register(t5);

float4 main() : SV_Target0 {
   return float4(t4.Load(0) + t5.Load(1));
}
"""


NEED_THREE_ARGS_ERR = "error: Option -fresource-set-binding requires at least 3 arguments"

@inside_glslc_testsuite('OptionFRegisterSetBinding')
class FRegisterSetBindingForFragRespected(expect.ValidAssemblyFileWithSubstr):
    """Tests -fresource-set-binding on specific shader two textures"""

    shader = FileShader(HLSL_SHADER, '.frag')
    glslc_args = ['-S', '-x', 'hlsl', shader,
                  '-fresource-set-binding', 'frag',
                  't4', '9', '16',
                  't5', '17', '18']
    expected_assembly_substr = """OpDecorate %t4 DescriptorSet 9
               OpDecorate %t4 Binding 16
               OpDecorate %t5 DescriptorSet 17
               OpDecorate %t5 Binding 18"""


@inside_glslc_testsuite('OptionFRegisterSetBinding')
class FRegisterSetBindingForFragRespectedJustOneTriple(expect.ValidAssemblyFileWithSubstr):
    """Tests -fresource-set-binding on specific shader just one texture specified."""

    shader = FileShader(HLSL_SHADER, '.frag')
    glslc_args = ['-S', '-x', 'hlsl', shader,
                  '-fresource-set-binding', 'frag',
                  't4', '9', '16']
    expected_assembly_substr = """OpDecorate %t4 DescriptorSet 9
               OpDecorate %t4 Binding 16
               OpDecorate %t5 DescriptorSet 0
               OpDecorate %t5 Binding 5"""


@inside_glslc_testsuite('OptionFRegisterSetBinding')
class FRegisterSetBindingForWrongStageIgnored(expect.ValidAssemblyFileWithSubstr):
    """Tests -fresource-set-binding on wrong shader ignored"""

    shader = FileShader(HLSL_SHADER, '.frag')
    glslc_args = ['-S', '-x', 'hlsl', shader,
                  '-fresource-set-binding', 'vert',
                  't4', '9', '16',
                  't5', '17', '18']
    expected_assembly_substr = """OpDecorate %t4 DescriptorSet 0
               OpDecorate %t4 Binding 4
               OpDecorate %t5 DescriptorSet 0
               OpDecorate %t5 Binding 5"""


@inside_glslc_testsuite('OptionFRegisterSetBinding')
class FRegisterSetBindingForAllRespected(expect.ValidAssemblyFileWithSubstr):
    """Tests -fresource-set-binding on all stages respected"""

    shader = FileShader(HLSL_SHADER, '.frag')
    glslc_args = ['-S', '-x', 'hlsl', shader,
                  '-fresource-set-binding',
                  't4', '9', '16',
                  't5', '17', '18']
    expected_assembly_substr = """OpDecorate %t4 DescriptorSet 9
               OpDecorate %t4 Binding 16
               OpDecorate %t5 DescriptorSet 17
               OpDecorate %t5 Binding 18"""


@inside_glslc_testsuite('OptionFRegisterSetBinding')
class FRegisterSetBindingTooFewArgs(expect.ErrorMessageSubstr):
    """Tests -fresource-set-binding with too few arguments"""

    shader = FileShader(HLSL_SHADER, '.frag')
    glslc_args = ['-S', '-x', 'hlsl', shader,
                  '-fresource-set-binding', 'frag',
                  't4', '9']
    expected_error_substr = NEED_THREE_ARGS_ERR


@inside_glslc_testsuite('OptionFRegisterSetBinding')
class FRegisterSetBindingInvalidSetNumber(expect.ErrorMessageSubstr):
    """Tests -fresource-set-binding with inavlid set number"""

    shader = FileShader(HLSL_SHADER, '.frag')
    glslc_args = ['-S', '-x', 'hlsl', shader,
                  '-fresource-set-binding', 'frag',
                  't4', '-9', '16']
    expected_error_substr = NEED_THREE_ARGS_ERR


@inside_glslc_testsuite('OptionFRegisterSetBinding')
class FRegisterSetBindingInvalidBindingNumber(expect.ErrorMessageSubstr):
    """Tests -fresource-set-binding with inavlid binding number"""

    shader = FileShader(HLSL_SHADER, '.frag')
    glslc_args = ['-S', '-x', 'hlsl', shader,
                  '-fresource-set-binding', 'frag',
                  't4', '9', '-16']
    expected_error_substr = NEED_THREE_ARGS_ERR
