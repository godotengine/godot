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

# A GLSL shader with inputs and outputs explicit locations.
GLSL_SHADER_IO_WITHOUT_LOCATIONS = """#version 310 es
  in vec4 m_in;
  in vec4 m_in1;
  out vec4 m_out;
  out vec4 m_out1;
  void main() {
    m_out = m_in;
    m_out1 = m_in1;
  }"""


# An HLSL fragment shader with inputs and outputs explicit locations.
HLSL_SHADER_IO_WITHOUT_LOCATIONS = """
  float4 Foo(float4 a, float4 b) : COLOR0 {
    return a + b;
  }"""


@inside_glslc_testsuite('OptionFAutoMapLocations')
class MissingLocationsResultsInError(expect.ErrorMessageSubstr):
    """Tests that compilation fails when inputs or outputs have no location."""

    shader = FileShader(GLSL_SHADER_IO_WITHOUT_LOCATIONS, '.vert')
    glslc_args = ['-S', shader]
    expected_error_substr = "SPIR-V requires location for user input/output"


@inside_glslc_testsuite('OptionFAutoMapLocations')
class FAutoMapLocationsGeneratesLocationsCheckInput(expect.ValidAssemblyFileWithSubstr):
    """Tests that the compiler generates locations upon request:  Input 0"""

    shader = FileShader(GLSL_SHADER_IO_WITHOUT_LOCATIONS, '.vert')
    glslc_args = ['-S', shader, '-fauto-map-locations']
    expected_assembly_substr = "OpDecorate %m_in Location 0"


@inside_glslc_testsuite('OptionFAutoMapLocations')
class FAutoMapLocationsGeneratesLocationsCheckOutput0(expect.ValidAssemblyFileWithSubstr):
    """Tests that the compiler generates locations upon request:  Output 0"""

    shader = FileShader(GLSL_SHADER_IO_WITHOUT_LOCATIONS, '.vert')
    glslc_args = ['-S', shader, '-fauto-map-locations']
    expected_assembly_substr = "OpDecorate %m_out Location 0"


# Currently Glslang only generates Location 0.
# See https://github.com/KhronosGroup/glslang/issues/1261
# TODO(dneto): Write tests that check Location 1 is generated for inputs and
# outputs.


# Glslang's HLSL compiler automatically assigns locations inptus and outputs.
@inside_glslc_testsuite('OptionFAutoMapLocations')
class HLSLCompilerGeneratesLocationsCheckInput0(expect.ValidAssemblyFileWithSubstr):
    """Tests that the HLSL compiler generates locations automatically: Input 0."""

    shader = FileShader(HLSL_SHADER_IO_WITHOUT_LOCATIONS, '.hlsl')
    glslc_args = ['-S', '-fshader-stage=frag', '-fentry-point=Foo', shader]
    expected_assembly_substr = "OpDecorate %a Location 0"


@inside_glslc_testsuite('OptionFAutoMapLocations')
class HLSLCompilerGeneratesLocationsCheckOutput(expect.ValidAssemblyFileWithSubstr):
    """Tests that the HLSL compiler generates locations automatically: Output."""

    shader = FileShader(HLSL_SHADER_IO_WITHOUT_LOCATIONS, '.hlsl')
    glslc_args = ['-S', '-fshader-stage=frag', '-fentry-point=Foo', shader]
    expected_assembly_substr = "OpDecorate %_entryPointOutput Location 0"
