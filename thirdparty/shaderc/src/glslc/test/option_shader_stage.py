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


def simple_vertex_shader():
    return """#version 310 es
    void main() {
        gl_Position = vec4(1., 2., 3., 4.);
    }"""


def simple_hlsl_vertex_shader():
    # Use "main" so we don't have to specify -fentry-point
    return """float4 main() : SV_POSITION { return float4(1.0); } """


def simple_fragment_shader():
    return """#version 310 es
    void main() {
        gl_FragDepth = 10.;
    }"""


def simple_tessellation_control_shader():
    return """#version 440 core
    layout(vertices = 3) out;
    void main() { }"""


def simple_tessellation_evaluation_shader():
    return """#version 440 core
    layout(triangles) in;
    void main() { }"""


def simple_geometry_shader():
    return """#version 150 core
    layout (triangles) in;
    layout (line_strip, max_vertices = 4) out;
    void main() { }"""


def simple_compute_shader():
    return """#version 310 es
    void main() {
        uvec3 temp = gl_WorkGroupID;
    }"""


@inside_glslc_testsuite('OptionShaderStage')
class TestShaderStageWithGlslExtension(expect.ValidObjectFile):
    """Tests -fshader-stage with .glsl extension."""

    shader = FileShader(simple_vertex_shader(), '.glsl')
    glslc_args = ['-c', '-fshader-stage=vertex', shader]


@inside_glslc_testsuite('OptionShaderStage')
class TestShaderStageWithHlslExtension(expect.ValidObjectFile):
    """Tests -fshader-stage with .hlsl extension."""

    shader = FileShader(simple_hlsl_vertex_shader(), '.hlsl')
    glslc_args = ['-c', '-fshader-stage=vertex', shader]


@inside_glslc_testsuite('OptionShaderStage')
class TestShaderStageWithKnownExtension(expect.ValidObjectFile):
    """Tests -fshader-stage with known extension."""

    shader = FileShader(simple_fragment_shader(), '.frag')
    glslc_args = ['-c', '-fshader-stage=fragment', shader]


@inside_glslc_testsuite('OptionShaderStage')
class TestShaderStageWithUnknownExtension(expect.ValidObjectFile):
    """Tests -fshader-stage with unknown extension."""

    shader = FileShader(simple_vertex_shader(), '.unknown')
    glslc_args = ['-c', '-fshader-stage=vertex', shader]


@inside_glslc_testsuite('OptionShaderStage')
class TestShaderStageWithNoExtension(expect.ValidObjectFile):
    """Tests -fshader-stage with no extension."""

    shader = FileShader(simple_vertex_shader(), '')
    glslc_args = ['-c', '-fshader-stage=vertex', shader]


@inside_glslc_testsuite('OptionShaderStage')
class TestAllShaderStages(expect.ValidObjectFile):
    """Tests all possible -fshader-stage values."""

    shader1 = FileShader(simple_vertex_shader(), '.glsl')
    shader2 = FileShader(simple_fragment_shader(), '.glsl')
    shader3 = FileShader(simple_tessellation_control_shader(), '.glsl')
    shader4 = FileShader(simple_tessellation_evaluation_shader(), '.glsl')
    shader5 = FileShader(simple_geometry_shader(), '.glsl')
    shader6 = FileShader(simple_compute_shader(), '.glsl')
    glslc_args = [
        '-c',
        '-fshader-stage=vertex', shader1,
        '-fshader-stage=fragment', shader2,
        '-fshader-stage=tesscontrol', shader3,
        '-fshader-stage=tesseval', shader4,
        '-fshader-stage=geometry', shader5,
        '-fshader-stage=compute', shader6]


@inside_glslc_testsuite('OptionShaderStage')
class TestShaderStageOverwriteFileExtension(expect.ValidObjectFile):
    """Tests -fshader-stage has precedence over file extension."""

    # a vertex shader camouflaged with .frag extension
    shader = FileShader(simple_vertex_shader(), '.frag')
    # Command line says it's vertex shader. Should compile successfully
    # as a vertex shader.
    glslc_args = ['-c', '-fshader-stage=vertex', shader]


@inside_glslc_testsuite('OptionShaderStage')
class TestShaderStageLatterOverwriteFormer(expect.ValidObjectFile):
    """Tests a latter -fshader-stage overwrite a former one."""

    shader = FileShader(simple_vertex_shader(), '.glsl')
    glslc_args = [
        '-c', '-fshader-stage=fragment', '-fshader-stage=vertex', shader]


@inside_glslc_testsuite('OptionShaderStage')
class TestShaderStageWithMultipleFiles(expect.ValidObjectFile):
    """Tests -fshader-stage covers all subsequent files."""

    shader1 = FileShader(simple_vertex_shader(), '.glsl')
    # a vertex shader with .frag extension
    shader2 = FileShader(simple_vertex_shader(), '.frag')
    shader3 = FileShader(simple_vertex_shader(), '.a_vert_shader')
    glslc_args = ['-c', '-fshader-stage=vertex', shader1, shader2, shader3]


@inside_glslc_testsuite('OptionShaderStage')
class TestShaderStageMultipleShaderStage(expect.ValidObjectFile):
    """Tests multiple -fshader-stage."""

    shader1 = FileShader(simple_vertex_shader(), '.glsl')
    shader2 = FileShader(simple_fragment_shader(), '.frag')
    shader3 = FileShader(simple_vertex_shader(), '.a_vert_shader')
    glslc_args = [
        '-c',
        '-fshader-stage=vertex', shader1,
        '-fshader-stage=fragment', shader2,
        '-fshader-stage=vertex', shader3]


@inside_glslc_testsuite('OptionShaderStage')
class TestFileExtensionBeforeShaderStage(expect.ValidObjectFile):
    """Tests that file extensions before -fshader-stage are not affected."""

    # before -fshader-stage
    shader1 = FileShader(simple_vertex_shader(), '.vert')
    # after -fshader-stage
    shader2 = FileShader(simple_fragment_shader(), '.frag')
    shader3 = FileShader(simple_fragment_shader(), '.vert')
    glslc_args = ['-c', shader1, '-fshader-stage=fragment', shader2, shader3]


@inside_glslc_testsuite('OptionShaderStage')
class TestShaderStageWrongShaderStageValue(expect.ErrorMessage):
    """Tests that wrong shader stage value results in an error."""

    shader = FileShader(simple_vertex_shader(), '.glsl')
    glslc_args = ['-c', '-fshader-stage=unknown', shader]
    expected_error = ["glslc: error: stage not recognized: 'unknown'\n"]


@inside_glslc_testsuite('OptionShaderStage')
class TestShaderStageGlslExtensionMissingShaderStage(expect.ErrorMessage):
    """Tests that missing -fshader-stage for .glsl extension results in
    an error."""

    shader = FileShader(simple_vertex_shader(), '.glsl')
    glslc_args = ['-c', shader]
    expected_error = [
        "glslc: error: '", shader,
        "': .glsl file encountered but no -fshader-stage specified ahead\n"]


@inside_glslc_testsuite('OptionShaderStage')
class TestShaderStageHlslExtensionMissingShaderStage(expect.ErrorMessage):
    """Tests that missing -fshader-stage for .hlsl extension results in
    an error."""

    shader = FileShader(simple_hlsl_vertex_shader(), '.hlsl')
    glslc_args = ['-c', '-x', 'hlsl', shader]
    expected_error = [
        "glslc: error: '", shader,
        "': .hlsl file encountered but no -fshader-stage specified ahead\n"]


@inside_glslc_testsuite('OptionShaderStage')
class TestShaderStageUnknownExtensionMissingShaderStage(expect.ErrorMessage):
    """Tests that missing -fshader-stage for unknown extension results in
    an error."""

    shader = FileShader(simple_vertex_shader(), '.a_vert_shader')
    glslc_args = ['-c', shader]
    expected_error = [
        "glslc: error: '", shader,
        "': file not recognized: File format not recognized\n"]
