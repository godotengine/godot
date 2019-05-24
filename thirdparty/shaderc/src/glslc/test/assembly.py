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
from placeholder import FileShader


def assembly_comments():
    return """
    ; SPIR-V
    ; Version: 1.0
    ; Generator: Google Shaderc over Glslang; 7
    ; Bound: 6
    ; Schema: 0"""


def empty_main_assembly():
    return assembly_comments() + """
         OpCapability Shader
    %1 = OpExtInstImport "GLSL.std.450"
         OpMemoryModel Logical GLSL450
         OpEntryPoint Vertex %4 "main"
         OpSource ESSL 310
         OpName %4 "main"
    %2 = OpTypeVoid
    %3 = OpTypeFunction %2
    %4 = OpFunction %2 None %3
    %5 = OpLabel
         OpReturn
         OpFunctionEnd"""


def empty_main():
    return '#version 310 es\nvoid main() {}'


@inside_glslc_testsuite('SpirvAssembly')
class TestAssemblyFileAsOnlyParameter(expect.ValidNamedObjectFile):
    """Tests that glslc accepts a SPIR-V assembly file as the only parameter."""

    shader = FileShader(empty_main_assembly(), '.spvasm')
    glslc_args = [shader]
    expected_object_filenames = ('a.spv',)


@inside_glslc_testsuite('SpirvAssembly')
class TestDashCAssemblyFile(expect.ValidObjectFile):
    """Tests that -c works with SPIR-V assembly file."""

    shader = FileShader(empty_main_assembly(), '.spvasm')
    glslc_args = ['-c', shader]


@inside_glslc_testsuite('SpirvAssembly')
class TestAssemblyFileWithOnlyComments(expect.ValidObjectFile):
    """Tests that glslc accepts an assembly file with only comments inside."""

    shader = FileShader(assembly_comments(), '.spvasm')
    glslc_args = ['-c', shader]


@inside_glslc_testsuite('SpirvAssembly')
class TestEmptyAssemblyFile(expect.ValidObjectFile):
    """Tests that glslc accepts an empty assembly file."""

    shader = FileShader('', '.spvasm')
    glslc_args = ['-c', shader]


@inside_glslc_testsuite('SpirvAssembly')
class TestDashEAssemblyFile(expect.SuccessfulReturn, expect.NoGeneratedFiles):
    """Tests that -E works with SPIR-V assembly file."""

    shader = FileShader(empty_main_assembly(), '.spvasm')
    glslc_args = ['-E', shader]


@inside_glslc_testsuite('SpirvAssembly')
class TestDashSAssemblyFile(expect.SuccessfulReturn, expect.NoGeneratedFiles):
    """Tests that -S works with SPIR-V assembly file."""

    shader = FileShader(empty_main_assembly(), '.spvasm')
    glslc_args = ['-S', shader]


@inside_glslc_testsuite('SpirvAssembly')
class TestMultipleAssemblyFiles(expect.ValidObjectFile):
    """Tests that glslc accepts multiple SPIR-V assembly files."""

    shader1 = FileShader(empty_main_assembly(), '.spvasm')
    shader2 = FileShader(empty_main_assembly(), '.spvasm')
    shader3 = FileShader(empty_main_assembly(), '.spvasm')
    glslc_args = ['-c', shader1, shader2, shader3]


@inside_glslc_testsuite('SpirvAssembly')
class TestHybridInputFiles(expect.ValidObjectFile):
    """Tests that glslc accepts a mix of SPIR-V assembly files and
    GLSL source files."""

    shader1 = FileShader(empty_main_assembly(), '.spvasm')
    shader2 = FileShader(empty_main(), '.vert')
    shader3 = FileShader(empty_main(), '.frag')
    glslc_args = ['-c', shader1, shader2, shader3]


@inside_glslc_testsuite('SpirvAssembly')
class TestShaderStageWithAssemblyFile(expect.ErrorMessage):
    """Tests that assembly files don't work with -fshader-stage"""

    shader = FileShader(empty_main_assembly(), '.spvasm')
    glslc_args = ['-c', '-fshader-stage=vertex', shader]

    expected_error = [
        shader, ": error: #version: Desktop shaders for Vulkan SPIR-V require "
        "version 140 or higher\n",
        shader, ":2: error: 'extraneous semicolon' :",
        " not supported for this version or the enabled extensions\n",
        shader, ":2: error: '' :  syntax error, unexpected IDENTIFIER\n",
        '3 errors generated.\n']


@inside_glslc_testsuite('SpirvAssembly')
class TestStdWithAssemblyFile(expect.ValidObjectFile):
    """Tests that --std= doesn't affect the processing of assembly files."""

    shader = FileShader(empty_main_assembly(), '.spvasm')
    glslc_args = ['-c', '-std=310es', shader]
