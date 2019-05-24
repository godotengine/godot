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

MINIMAL_SHADER = "#version 140\nvoid main(){}"
# This one is valid GLSL but not valid HLSL.
GLSL_VERTEX_SHADER = "#version 140\nvoid main(){ gl_Position = vec4(1.0);}"
# This one is GLSL but without leading #version.  Should result in
# a parser error when compiled as HLSL.
GLSL_VERTEX_SHADER_WITHOUT_VERSION = "void main(){ gl_Position = vec4(1.0);}"
# This one is valid HLSL but not valid GLSL.
# Use entry point "main" so we don't have to specify -fentry-point
HLSL_VERTEX_SHADER = "float4 main() : SV_POSITION { return float4(1.0); }"

@inside_glslc_testsuite('OptionDashX')
class TestDashXNoArg(expect.ErrorMessage):
    """Tests -x with nothing."""

    glslc_args = ['-x']
    expected_error = [
        "glslc: error: argument to '-x' is missing (expected 1 value)\n",
        'glslc: error: no input files\n']


@inside_glslc_testsuite('OptionDashX')
class TestDashXGlslOnGlslShader(expect.ValidObjectFile):
    """Tests -x glsl on a GLSL shader."""

    shader = FileShader(GLSL_VERTEX_SHADER, '.vert')
    glslc_args = ['-x', 'glsl', '-c', shader]


@inside_glslc_testsuite('OptionDashX')
class TestDashXGlslOnHlslShader(expect.ErrorMessageSubstr):
    """Tests -x glsl on an HLSL shader."""

    shader = FileShader(HLSL_VERTEX_SHADER, '.vert')
    glslc_args = ['-x', 'glsl', '-c', shader]
    expected_error_substr = ["error: #version: Desktop shaders for Vulkan SPIR-V"
                             " require version 140 or higher\n"]


@inside_glslc_testsuite('OptionDashX')
class TestDashXHlslOnGlslShader(expect.ErrorMessageSubstr):
    """Tests -x hlsl on a GLSL shader."""

    shader = FileShader(GLSL_VERTEX_SHADER, '.vert')
    glslc_args = ['-x', 'hlsl', '-c', shader]
    expected_error_substr = ["error: '#version' : invalid preprocessor command\n"]


@inside_glslc_testsuite('OptionDashX')
class TestDashXHlslOnGlslShaderWithoutVertex(expect.ErrorMessageSubstr):
    """Tests -x hlsl on a GLSL shader without leading #version."""

    shader = FileShader(GLSL_VERTEX_SHADER_WITHOUT_VERSION, '.vert')
    glslc_args = ['-x', 'hlsl', '-c', shader]
    expected_error_substr = ["error: 'vec4' : no matching overloaded function found\n"]


@inside_glslc_testsuite('OptionDashX')
class TestDashXWrongParam(expect.ErrorMessage):
    """Tests -x with wrong parameter."""

    shader = FileShader(MINIMAL_SHADER, '.vert')
    glslc_args = ['-x', 'gl', shader]
    expected_error = ["glslc: error: language not recognized: 'gl'\n"]


@inside_glslc_testsuite('OptionDashX')
class TestMultipleDashX(expect.ValidObjectFile):
    """Tests that multiple -x works with a single language."""

    shader = FileShader(GLSL_VERTEX_SHADER, '.vert')
    glslc_args = ['-c', '-x', 'glsl', '-x', 'glsl', shader, '-x', 'glsl']


@inside_glslc_testsuite('OptionDashX')
class TestMultipleDashXMixedLanguages(expect.ValidObjectFile):
    """Tests that multiple -x works with different languages."""

    glsl_shader = FileShader(GLSL_VERTEX_SHADER, '.vert')
    hlsl_shader = FileShader(HLSL_VERTEX_SHADER, '.vert')
    glslc_args = ['-c', '-x', 'hlsl', hlsl_shader,
                  '-x', 'glsl', glsl_shader,
                  '-x', 'hlsl', hlsl_shader,
                  '-x', 'glsl', glsl_shader]


@inside_glslc_testsuite('OptionDashX')
class TestMultipleDashXCorrectWrong(expect.ErrorMessage):
    """Tests -x glsl -x [wrong-language]."""

    shader = FileShader(MINIMAL_SHADER, '.vert')
    glslc_args = ['-x', 'glsl', '-x', 'foo', shader]
    expected_error = ["glslc: error: language not recognized: 'foo'\n"]


@inside_glslc_testsuite('OptionDashX')
class TestMultipleDashXWrongCorrect(expect.ErrorMessage):
    """Tests -x [wrong-language] -x glsl."""

    shader = FileShader(MINIMAL_SHADER, '.vert')
    glslc_args = ['-xbar', '-x', 'glsl', shader]
    expected_error = ["glslc: error: language not recognized: 'bar'\n"]


@inside_glslc_testsuite('OptionDashX')
class TestDashXGlslConcatenated(expect.ValidObjectFile):
    """Tests -xglsl."""

    shader = FileShader(MINIMAL_SHADER, '.vert')
    glslc_args = ['-xglsl', shader, '-c']


@inside_glslc_testsuite('OptionDashX')
class TestDashXWrongParamConcatenated(expect.ErrorMessage):
    """Tests -x concatenated with a wrong language."""

    shader = FileShader(MINIMAL_SHADER, '.vert')
    glslc_args = ['-xsl', shader]
    expected_error = ["glslc: error: language not recognized: 'sl'\n"]


@inside_glslc_testsuite('OptionDashX')
class TestDashXEmpty(expect.ErrorMessage):
    """Tests -x ''."""

    shader = FileShader(MINIMAL_SHADER, '.vert')
    glslc_args = ['-x', '', shader]
    expected_error = ["glslc: error: language not recognized: ''\n"]
