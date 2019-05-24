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


def opengl_compat_fragment_shader():
    return """#version 330
uniform highp sampler2D tex;
void main() {
  gl_FragColor = texture2D(tex, vec2(0.0, 0.0));
}"""


def opengl_vertex_shader():
    return """#version 330
void main() { int t = gl_VertexID; }"""


def vulkan_vertex_shader():
    return """#version 310 es
void main() { int t = gl_VertexIndex; }"""


def vulkan_compute_subgroup_shader():
    """Returns a compute shader that requires Vulkan 1.1"""
    return """#version 450
              #extension GL_KHR_shader_subgroup_basic : enable
              void main() { subgroupBarrier(); }"""


@inside_glslc_testsuite('OptionTargetEnv')
class TestTargetEnvEqOpenglCompatWithOpenGlCompatShader(expect.ValidObjectFile):
    """Tests that compiling OpenGL Compatibility Fragment shader with
    --target-env=opengl_compat works correctly"""
    shader = FileShader(opengl_compat_fragment_shader(), '.frag')
    glslc_args = ['--target-env=opengl_compat', '-c', shader]


@inside_glslc_testsuite('OptionTargetEnv')
class TestTargetEnvEqOpenglWithOpenGlCompatShader(expect.ErrorMessageSubstr):
    """Tests the error message of compiling OpenGL Compatibility Fragment shader
    with --target-env=opengl"""
    shader = FileShader(opengl_compat_fragment_shader(), '.frag')
    glslc_args = ['--target-env=opengl', shader]
    expected_error_substr = [shader, ":4: error: 'assign' :  ",
                             "cannot convert from ' const float' to ",
                             "'layout( location=0) out 4-component ",
                             "vector of float"]


@inside_glslc_testsuite('OptionTargetEnv')
class TestTargetEnvEqOpenglCompatWithOpenGlVertexShader(expect.ValidObjectFile):
    """Tests that compiling OpenGL vertex shader with --target-env=opengl_compat
    generates valid SPIR-V code"""
    shader = FileShader(opengl_vertex_shader(), '.vert')
    glslc_args = ['--target-env=opengl_compat', '-c', shader]


@inside_glslc_testsuite('OptionTargetEnv')
class TestTargetEnvEqOpenglWithOpenGlVertexShader(expect.ValidObjectFile):
    """Tests that compiling OpenGL vertex shader with --target-env=opengl
    generates valid SPIR-V code"""
    shader = FileShader(opengl_vertex_shader(), '.vert')
    glslc_args = ['--target-env=opengl', '-c', shader]


@inside_glslc_testsuite('OptionTargetEnv')
class TestDefaultTargetEnvWithVulkanShader(expect.ValidObjectFile):
    """Tests that compiling a Vulkan-specific shader with a default
    target environment succeeds"""
    shader = FileShader(vulkan_vertex_shader(), '.vert')
    glslc_args = ['-c', shader]


@inside_glslc_testsuite('OptionTargetEnv')
class TestTargetEnvEqVulkanWithVulkan1_0ShaderSucceeds(expect.ValidObjectFile):
    """Tests that compiling a Vulkan-specific Vulkan 1.0 shader succeeds with
    --target-env=vulkan"""
    shader = FileShader(vulkan_vertex_shader(), '.vert')
    glslc_args = ['--target-env=vulkan', '-c', shader]


@inside_glslc_testsuite('OptionTargetEnv')
class TestTargetEnvEqVulkan1_0WithVulkan1_0ShaderSucceeds(expect.ValidObjectFile):
    """Tests that compiling a Vulkan-specific Vulkan 1.0 shader succeeds with
    --target-env=vulkan1.0"""
    shader = FileShader(vulkan_vertex_shader(), '.vert')
    glslc_args = ['--target-env=vulkan1.0', '-c', shader]


@inside_glslc_testsuite('OptionTargetEnv')
class TestTargetEnvEqVulkan1_0WithVulkan1_1ShaderFails(expect.ErrorMessageSubstr):
    shader = FileShader(vulkan_compute_subgroup_shader(), '.comp')
    glslc_args = ['--target-env=vulkan1.0', '-c', shader]
    expected_error_substr = "error: 'subgroup op' : requires SPIR-V 1.3"


@inside_glslc_testsuite('OptionTargetEnv')
class TestTargetEnvEqVulkan1_1WithVulkan1_0ShaderSucceeds(expect.ValidObjectFile1_3):
    shader = FileShader(vulkan_vertex_shader(), '.vert')
    glslc_args = ['--target-env=vulkan1.1', '-c', shader]


@inside_glslc_testsuite('OptionTargetEnv')
class TestTargetEnvEqVulkan1_1WithVulkan1_1ShaderSucceeds(expect.ValidObjectFile1_3):
    shader = FileShader(vulkan_compute_subgroup_shader(), '.comp')
    glslc_args = ['--target-env=vulkan1.1', '-c', shader]


@inside_glslc_testsuite('OptionTargetEnv')
class TestTargetEnvEqOpenGL4_5WithOpenGLShaderSucceeds(expect.ValidObjectFile):
    shader = FileShader(opengl_vertex_shader(), '.vert')
    glslc_args = ['--target-env=opengl4.5', '-c', shader]


@inside_glslc_testsuite('OptionTargetEnv')
class TestTargetEnvEqOpenGL4_6WithOpenGLShaderFailsUnsupported(expect.ErrorMessageSubstr):
    shader = FileShader(opengl_vertex_shader(), '.vert')
    glslc_args = ['--target-env=opengl4.6', '-c', shader]
    expected_error_substr = "invalid value 'opengl4.6' in '--target-env=opengl4.6'"


# Note: Negative tests are covered in the libshaderc_util unit tests.
# For example, that an OpenGL-specific shader should fail to compile
# for Vulkan, or a Vulkan-specific shader should fail to compile for
# OpenGL.


@inside_glslc_testsuite('OptionTargetEnv')
class TestTargetEnvEqNoArg(expect.ErrorMessage):
    """Tests the error message of assigning empty string to --target-env"""
    shader = FileShader(opengl_vertex_shader(), '.vert')
    glslc_args = ['--target-env=', shader]
    expected_error = ["glslc: error: invalid value ",
                      "'' in '--target-env='\n"]


@inside_glslc_testsuite('OptionTargetEnv')
class TestTargetEnvNoEqNoArg(expect.ErrorMessage):
    """Tests the error message of using --target-env without equal sign and
    arguments"""
    shader = FileShader(opengl_vertex_shader(), '.vert')
    glslc_args = ['--target-env', shader]
    expected_error = ["glslc: error: unsupported option: ",
                      "'--target-env'\n"]


@inside_glslc_testsuite('OptionTargetEnv')
class TestTargetEnvNoEqWithArg(expect.ErrorMessage):
    """Tests the error message of using --target-env without equal sign but
    arguments"""
    shader = FileShader(opengl_vertex_shader(), '.vert')
    glslc_args = ['--target-env', 'opengl', shader]
    expected_error = ["glslc: error: unsupported option: ",
                      "'--target-env'\n"]


@inside_glslc_testsuite('OptionTargetEnv')
class TestTargetEnvEqWrongArg(expect.ErrorMessage):
    """Tests the error message of using --target-env with wrong argument"""
    shader = FileShader(opengl_vertex_shader(), '.vert')
    glslc_args = ['--target-env=wrong_arg', shader]
    expected_error = ["glslc: error: invalid value ",
                      "'wrong_arg' in '--target-env=wrong_arg'\n"]
