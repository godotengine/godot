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
import os.path
from glslc_test_framework import inside_glslc_testsuite
from placeholder import FileShader, StdinShader


def simple_vertex_shader():
    return """#version 310 es
    void main() {
        gl_Position = vec4(1., 2., 3., 4.);
    }"""


def simple_fragment_shader():
    return """#version 310 es
    void main() {
        gl_FragDepth = 10.;
    }"""


def simple_compute_shader():
    return """#version 310 es
    void main() {
        uvec3 temp = gl_WorkGroupID;
    }"""


@inside_glslc_testsuite('OptionDashCapS')
class TestSingleDashCapSSingleFile(expect.ValidAssemblyFile):
    """Tests that -S works with a single file."""

    shader = FileShader(simple_vertex_shader(), '.vert')
    glslc_args = ['-S', shader]


@inside_glslc_testsuite('OptionDashCapS')
class TestSingleFileSingleDashCapS(expect.ValidAssemblyFile):
    """Tests that the position of -S doesn't matter."""

    shader = FileShader(simple_vertex_shader(), '.vert')
    glslc_args = [shader, '-S']


@inside_glslc_testsuite('OptionDashCapS')
class TestSingleDashCapSMultipleFiles(expect.ValidAssemblyFile):
    """Tests that -S works with multiple files."""

    shader1 = FileShader(simple_vertex_shader(), '.vert')
    shader2 = FileShader(simple_vertex_shader(), '.vert')
    shader3 = FileShader(simple_fragment_shader(), '.frag')
    glslc_args = ['-S', shader1, shader2, shader3]


@inside_glslc_testsuite('OptionDashCapS')
class TestMultipleDashCapSSingleFile(expect.ValidAssemblyFile):
    """Tests that multiple -Ss works as one."""

    shader = FileShader(simple_vertex_shader(), '.vert')
    glslc_args = ['-S', '-S', shader, '-S']


@inside_glslc_testsuite('OptionDashCapS')
class TestMultipleDashCapSMultipleFiles(expect.ValidAssemblyFile):
    """Tests a mix of -Ss and files."""

    shader1 = FileShader(simple_fragment_shader(), '.frag')
    shader2 = FileShader(simple_vertex_shader(), '.vert')
    shader3 = FileShader(simple_compute_shader(), '.comp')
    glslc_args = ['-S', shader1, '-S', '-S', shader2, '-S', shader3, '-S']


@inside_glslc_testsuite('OptionDashCapS')
class TestDashCapSWithDashC(expect.ValidAssemblyFile):
    """Tests that -S overwrites -c."""

    shader1 = FileShader(simple_fragment_shader(), '.frag')
    shader2 = FileShader(simple_vertex_shader(), '.vert')
    glslc_args = ['-c', '-S', shader1, '-c', '-c', shader2]


@inside_glslc_testsuite('OptionDashCapS')
class TestDashCapSWithDashFShaderStage(expect.ValidAssemblyFile):
    """Tests that -S works with -fshader-stage=."""

    shader1 = FileShader(simple_fragment_shader(), '.glsl')
    shader2 = FileShader(simple_vertex_shader(), '.glsl')
    shader3 = FileShader(simple_compute_shader(), '.glsl')
    glslc_args = ['-S',
                  '-fshader-stage=fragment', shader1,
                  '-fshader-stage=vertex', shader2,
                  '-fshader-stage=compute', shader3]


@inside_glslc_testsuite('OptionDashCapS')
class TestDashCapSWithDashStd(expect.ValidAssemblyFileWithWarning):
    """Tests that -S works with -std=."""

    shader1 = FileShader(simple_fragment_shader(), '.frag')
    shader2 = FileShader(simple_vertex_shader(), '.vert')
    shader3 = FileShader(simple_compute_shader(), '.comp')
    glslc_args = ['-S', '-std=450', shader1, shader2, shader3]

    w = (': warning: (version, profile) forced to be (450, none), '
         'while in source code it is (310, es)\n')
    expected_warning = [
        shader1, w, shader2, w, shader3, w, '3 warnings generated.\n']


@inside_glslc_testsuite('OptionDashCapS')
class TestDashCapSWithDashOSingleFile(expect.SuccessfulReturn,
                                      expect.CorrectAssemblyFilePreamble):
    """Tests that -S works with -o on a single file."""

    shader = FileShader(simple_fragment_shader(), '.frag')
    glslc_args = ['-S', '-o', 'blabla', shader]

    def check_output_blabla(self, status):
        output_name = os.path.join(status.directory, 'blabla')
        return self.verify_assembly_file_preamble(output_name)


@inside_glslc_testsuite('OptionDashCapS')
class TestDashCapSWithDashOMultipleFiles(expect.ErrorMessage):
    """Tests that -S works with -o on a single file."""

    shader1 = FileShader(simple_fragment_shader(), '.frag')
    shader2 = FileShader(simple_vertex_shader(), '.vert')
    glslc_args = ['-S', '-o', 'blabla', shader1, shader2]

    expected_error = ['glslc: error: cannot specify -o when '
                      'generating multiple output files\n']


@inside_glslc_testsuite('OptionDashCapS')
class TestDashCapSWithStdIn(expect.ValidAssemblyFile):
    """Tests that -S works with stdin."""

    shader = StdinShader(simple_fragment_shader())
    glslc_args = ['-S', '-fshader-stage=fragment', shader]


@inside_glslc_testsuite('OptionDashCapS')
class TestDashCapSWithStdOut(
    expect.ReturnCodeIsZero, expect.StdoutMatch, expect.StderrMatch):
    """Tests that -S works with stdout."""

    shader = FileShader(simple_fragment_shader(), '.frag')
    glslc_args = ['-S', '-o', '-', shader]

    expected_stdout = True
    expected_stderr = ''
