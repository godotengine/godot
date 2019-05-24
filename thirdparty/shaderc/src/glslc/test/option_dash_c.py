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


def empty_es_310_shader():
    return '#version 310 es\n void main() {}\n'


@inside_glslc_testsuite('OptionC')
class TestSingleDashCSingleFile(expect.ValidObjectFile):
    """Tests that glslc accepts -c [filename]."""

    shader = FileShader(empty_es_310_shader(), '.vert')
    glslc_args = ['-c', shader]


@inside_glslc_testsuite('OptionC')
class TestSingleFileSingleDashC(expect.ValidObjectFile):
    """Tests that glslc accepts [filename] -c."""

    shader = FileShader(empty_es_310_shader(), '.vert')
    glslc_args = [shader, '-c']


@inside_glslc_testsuite('OptionC')
class TestMultipleFiles(expect.ValidObjectFile):
    """Tests that glslc accepts -c and multiple source files."""

    shader1 = FileShader(empty_es_310_shader(), '.vert')
    shader2 = FileShader(empty_es_310_shader(), '.frag')
    glslc_args = ['-c', shader1, shader2]


@inside_glslc_testsuite('OptionC')
class TestMultipleDashC(expect.ValidObjectFile):
    """Tests that glslc accepts multiple -c and treated them as one."""

    shader1 = FileShader(empty_es_310_shader(), '.vert')
    shader2 = FileShader(empty_es_310_shader(), '.vert')
    glslc_args = ['-c', shader1, '-c', '-c', shader2]
