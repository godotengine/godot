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
from glslc_test_framework import GlslCTest, inside_glslc_testsuite
from placeholder import FileShader


def empty_es_310_shader():
    return '#version 310 es\n void main() {}\n'


@inside_glslc_testsuite('FileExtension')
class VerifyVertExtension(expect.ValidObjectFile):
    """Tests glslc accepts vertex shader extension (.vert)."""

    shader = FileShader(empty_es_310_shader(), '.vert')
    glslc_args = ['-c', shader]


@inside_glslc_testsuite('FileExtension')
class VerifyFragExtension(expect.ValidObjectFile):
    """Tests glslc accepts fragment shader extension (.frag)."""

    shader = FileShader(empty_es_310_shader(), '.frag')
    glslc_args = ['-c', shader]


@inside_glslc_testsuite('FileExtension')
class VerifyTescExtension(expect.ValidObjectFile):
    """Tests glslc accepts tessellation control shader extension (.tesc)."""

    shader = FileShader(
        '#version 440 core\n layout(vertices = 3) out;\n void main() {}',
        '.tesc')
    glslc_args = ['-c', shader]


@inside_glslc_testsuite('FileExtension')
class VerifyTeseExtension(expect.ValidObjectFile):
    """Tests glslc accepts tessellation evaluation shader extension (.tese)."""

    shader = FileShader(
        '#version 440 core\n layout(triangles) in;\n void main() {}', '.tese')
    glslc_args = ['-c', shader]


@inside_glslc_testsuite('FileExtension')
class VerifyGeomExtension(expect.ValidObjectFile):
    """Tests glslc accepts geomtry shader extension (.geom)."""

    shader = FileShader(
        '#version 150 core\n layout (triangles) in;\n'
        'layout (line_strip, max_vertices = 4) out;\n void main() {}',
        '.geom')
    glslc_args = ['-c', shader]


@inside_glslc_testsuite('FileExtension')
class VerifyCompExtension(expect.ValidObjectFile):
    """Tests glslc accepts compute shader extension (.comp)."""

    shader = FileShader(empty_es_310_shader(), '.comp')
    glslc_args = ['-c', shader]


@inside_glslc_testsuite('FileExtension')
class InvalidExtension(expect.ErrorMessage):
    """Tests the error message if a file extension cannot be determined."""

    shader = FileShader('#version 150\n', '.fraga')
    glslc_args = ['-c', shader]
    expected_error = [
        "glslc: error: '", shader,
        "': file not recognized: File format not recognized\n"]
