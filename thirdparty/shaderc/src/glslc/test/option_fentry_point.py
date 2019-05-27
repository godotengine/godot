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
from glslc_test_framework import inside_glslc_testsuite
from placeholder import FileShader

MINIMAL_SHADER = "#version 140\nvoid main(){}"
# This one is valid GLSL but not valid HLSL.
GLSL_VERTEX_SHADER = "#version 140\nvoid main(){ gl_Position = vec4(1.0);}"
# This one is valid HLSL but not valid GLSL.
HLSL_VERTEX_SHADER = "float4 EntryPoint() : SV_POSITION { return float4(1.0); }"
HLSL_VERTEX_SHADER_WITH_MAIN = "float4 main() : SV_POSITION { return float4(1.0); }"
HLSL_VERTEX_SHADER_WITH_FOOBAR = "float4 Foobar() : SV_POSITION { return float4(1.0); }"

# Expected assembly code within certain shaders.
ASSEMBLY_ENTRY_POINT = "OpEntryPoint Vertex %EntryPoint \"EntryPoint\""
ASSEMBLY_MAIN = "OpEntryPoint Vertex %main \"main\""
ASSEMBLY_FOOBAR = "OpEntryPoint Vertex %Foobar \"Foobar\""


@inside_glslc_testsuite('OptionFEntryPoint')
class TestEntryPointDefaultsToMainForGlsl(expect.ValidAssemblyFileWithSubstr):
    """Tests that entry point name defaults to "main" in a GLSL shader."""

    shader = FileShader(GLSL_VERTEX_SHADER, '.vert')
    glslc_args = ['-S', shader]
    expected_assembly_substr = ASSEMBLY_MAIN


@inside_glslc_testsuite('OptionFEntryPoint')
class TestEntryPointDefaultsToMainForHlsl(expect.ValidAssemblyFileWithSubstr):
    """Tests that entry point name defaults to "main" in an HLSL shader."""

    shader = FileShader(HLSL_VERTEX_SHADER_WITH_MAIN, '.vert')
    glslc_args = ['-x', 'hlsl', '-S', shader]
    expected_assembly_substr = ASSEMBLY_MAIN


@inside_glslc_testsuite('OptionFEntryPoint')
class TestFEntryPointMainOnGlslShader(expect.ValidAssemblyFileWithSubstr):
    """Tests -fentry-point=main with a GLSL shader."""

    shader = FileShader(GLSL_VERTEX_SHADER, '.vert')
    glslc_args = ['-fentry-point=main', '-S', shader]
    expected_assembly_substr = ASSEMBLY_MAIN


@inside_glslc_testsuite('OptionFEntryPoint')
class TestFEntryPointMainOnHlslShaderNotMatchingSource(expect.ValidObjectFileWithWarning):
    """Tests -x hlsl on an HLSL shader with -fentry-point=main
    not matching the source."""

    shader = FileShader(HLSL_VERTEX_SHADER, '.vert')
    glslc_args = ['-x', 'hlsl', '-fentry-point=main', '-c', shader]
    expected_warning = [shader,
                        ': warning: Linking vertex stage: Entry point not found\n'
                        '1 warning generated.\n']


@inside_glslc_testsuite('OptionFEntryPoint')
class TestFEntryPointSpecifiedOnHlslShaderInDisassembly(expect.ValidObjectFileWithAssemblySubstr):
    """Tests -x hlsl on an HLSL shader with -fentry-point=EntryPoint
    matching source."""

    shader = FileShader(HLSL_VERTEX_SHADER, '.vert', assembly_substr=ASSEMBLY_ENTRY_POINT)
    glslc_args = ['-x', 'hlsl', '-fentry-point=EntryPoint', '-c', shader]


@inside_glslc_testsuite('OptionFEntryPoint')
class TestFEntryPointAffectsSubsequentShaderFiles(expect.ValidObjectFileWithAssemblySubstr):
    """Tests -x hlsl affects several subsequent shader source files."""

    shader1 = FileShader(HLSL_VERTEX_SHADER, '.vert', assembly_substr=ASSEMBLY_ENTRY_POINT)
    shader2 = FileShader(HLSL_VERTEX_SHADER, '.vert', assembly_substr=ASSEMBLY_ENTRY_POINT)
    glslc_args = ['-x', 'hlsl', '-fentry-point=EntryPoint', '-c', shader1, shader2]


@inside_glslc_testsuite('OptionFEntryPoint')
class TestFEntryPointOverridesItself(expect.ValidObjectFileWithAssemblySubstr):
    """Tests that a later -fentry-point option overrides an earlier use."""

    shader = FileShader(HLSL_VERTEX_SHADER, '.vert', assembly_substr=ASSEMBLY_ENTRY_POINT)
    glslc_args = ['-x', 'hlsl', '-fentry-point=foobar', '-fentry-point=EntryPoint',
                  '-c', shader]


@inside_glslc_testsuite('OptionFEntryPoint')
class TestFEntryPointDefaultAndTwoOthers(expect.ValidObjectFileWithAssemblySubstr):
    """Tests three shaders with different entry point names. The first uses "main"
    with default entry point processing, and the remaining shaders get their
    own -fentry-point argument."""

    shaderMain = FileShader(HLSL_VERTEX_SHADER_WITH_MAIN, '.vert',
                            assembly_substr=ASSEMBLY_MAIN)
    shaderEntryPoint = FileShader(HLSL_VERTEX_SHADER, '.vert',
                                  assembly_substr=ASSEMBLY_ENTRY_POINT)
    shaderFoobar = FileShader(HLSL_VERTEX_SHADER_WITH_FOOBAR, '.vert',
                              assembly_substr=ASSEMBLY_FOOBAR)
    glslc_args = ['-x', 'hlsl', '-c', shaderMain,
                  '-fentry-point=EntryPoint', shaderEntryPoint,
                  '-fentry-point=Foobar', shaderFoobar]
