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
from environment import Directory, File
from glslc_test_framework import inside_glslc_testsuite
from placeholder import FileShader, StdinShader

VERTEX_ONLY_SHADER_WITH_PRAGMA = \
    """#version 310 es
    #pragma shader_stage(vertex)
    void main() {
        gl_Position = vec4(1.);
    }"""

FRAGMENT_ONLY_SHADER_WITH_PRAGMA = \
    """#version 310 es
    #pragma shader_stage(fragment)
    void main() {
        gl_FragDepth = 10.;
    }"""


TESS_CONTROL_ONLY_SHADER_WITH_PRAGMA = \
    """#version 440 core
    #pragma shader_stage(tesscontrol)
    layout(vertices = 3) out;
    void main() { }"""


TESS_EVAL_ONLY_SHADER_WITH_PRAGMA = \
    """#version 440 core
    #pragma shader_stage(tesseval)
    layout(triangles) in;
    void main() { }"""


GEOMETRY_ONLY_SHDER_WITH_PRAGMA = \
    """#version 150 core
    #pragma shader_stage(geometry)
    layout (triangles) in;
    layout (line_strip, max_vertices = 4) out;
    void main() { }"""


COMPUTE_ONLY_SHADER_WITH_PRAGMA = \
    """#version 310 es
    #pragma shader_stage(compute)
    void main() {
        uvec3 temp = gl_WorkGroupID;
    }"""

# In the following tests,
# PSS stands for PragmaShaderStage, and OSS stands for OptionShaderStage.


@inside_glslc_testsuite('PragmaShaderStage')
class TestPSSWithGlslExtension(expect.ValidObjectFile):
    """Tests #pragma shader_stage() with .glsl extension."""

    shader = FileShader(VERTEX_ONLY_SHADER_WITH_PRAGMA, '.glsl')
    glslc_args = ['-c', shader]


@inside_glslc_testsuite('PragmaShaderStage')
class TestPSSWithUnkownExtension(expect.ValidObjectFile):
    """Tests #pragma shader_stage() with unknown extension."""

    shader = FileShader(VERTEX_ONLY_SHADER_WITH_PRAGMA, '.unkown')
    glslc_args = ['-c', shader]


@inside_glslc_testsuite('PragmaShaderStage')
class TestPSSWithStdin(expect.ValidObjectFile):
    """Tests #pragma shader_stage() with stdin."""

    shader = StdinShader(VERTEX_ONLY_SHADER_WITH_PRAGMA)
    glslc_args = ['-c', shader]


@inside_glslc_testsuite('PragmaShaderStage')
class TestPSSWithSameShaderExtension(expect.ValidObjectFile):
    """Tests that #pragma shader_stage() specifies the same stage as file
    extesion."""

    shader = FileShader(VERTEX_ONLY_SHADER_WITH_PRAGMA, '.vert')
    glslc_args = ['-c', shader]


@inside_glslc_testsuite('PragmaShaderStage')
class TestPSSOverrideShaderExtension(expect.ValidObjectFile):
    """Tests that #pragma shader_stage() overrides file extension."""

    shader = FileShader(VERTEX_ONLY_SHADER_WITH_PRAGMA, '.frag')
    glslc_args = ['-c', shader]


@inside_glslc_testsuite('PragmaShaderStage')
class TestOSSOverridePSS(expect.ValidObjectFile):
    """Tests that -fshader-stage overrides #pragma shader_stage()."""

    # wrong pragma and wrong file extension
    shader = FileShader(
        """#version 310 es
        #pragma shader_stage(fragment)
        void main() {
            gl_Position = vec4(1.);
        }""", '.frag')
    # -fshader-stage to the rescue! ^.^
    glslc_args = ['-c', '-fshader-stage=vertex', shader]


@inside_glslc_testsuite('PragmaShaderStage')
class TestMultipleSamePSS(expect.ValidObjectFile):
    """Tests that multiple identical #pragma shader_stage() works."""

    shader = FileShader(
        """#version 310 es
        #pragma shader_stage(vertex)
        #pragma shader_stage(vertex)
        void main() {
        #pragma shader_stage(vertex)
            gl_Position = vec4(1.);
        #pragma shader_stage(vertex)
        }
        #pragma shader_stage(vertex)
        #pragma shader_stage(vertex)
        """, '.glsl')
    glslc_args = ['-c', shader]


@inside_glslc_testsuite('PragmaShaderStage')
class TestConflictingPSS(expect.ErrorMessage):
    """Conflicting #pragma shader_stage() directives result in an error."""

    shader = FileShader(
        """#version 310 es
        #pragma shader_stage(vertex)
        void main() {
            gl_Position = vec4(1.);
        }
        #pragma shader_stage(fragment)
        """, '.glsl')
    glslc_args = ['-c', shader]
    expected_error = [
        shader, ":6: error: '#pragma': conflicting stages for 'shader_stage' "
        "#pragma: 'fragment' (was 'vertex' at ", shader, ':2)\n']


@inside_glslc_testsuite('PragmaShaderStage')
class TestAllPSSValues(expect.ValidObjectFile):
    """Tests all possible #pragma shader_stage() values."""

    shader1 = FileShader(VERTEX_ONLY_SHADER_WITH_PRAGMA, '.glsl')
    shader2 = FileShader(FRAGMENT_ONLY_SHADER_WITH_PRAGMA, '.glsl')
    shader3 = FileShader(TESS_CONTROL_ONLY_SHADER_WITH_PRAGMA, '.glsl')
    shader4 = FileShader(TESS_EVAL_ONLY_SHADER_WITH_PRAGMA, '.glsl')
    shader5 = FileShader(GEOMETRY_ONLY_SHDER_WITH_PRAGMA, '.glsl')
    shader6 = FileShader(COMPUTE_ONLY_SHADER_WITH_PRAGMA, '.glsl')
    glslc_args = ['-c', shader1, shader2, shader3, shader4, shader5, shader6]


@inside_glslc_testsuite('PragmaShaderStage')
class TestWrongPSSValue(expect.ErrorMessage):
    """Tests that #pragma shader_stage([wrong-stage]) results in an error."""

    shader = FileShader(
        """#version 310 es
        #pragma shader_stage(superstage)
        void main() {
            gl_Position = vec4(1.);
        }
        """, '.glsl')
    glslc_args = ['-c', shader]
    expected_error = [
        shader, ":2: error: '#pragma': invalid stage for 'shader_stage' "
        "#pragma: 'superstage'\n"]


@inside_glslc_testsuite('PragmaShaderStage')
class TestEmptyPSSValue(expect.ErrorMessage):
    """Tests that #pragma shader_stage([empty]) results in an error."""

    shader = FileShader(
        """#version 310 es
        #pragma shader_stage()
        void main() {
            gl_Position = vec4(1.);
        }
        """, '.glsl')
    glslc_args = ['-c', shader]
    expected_error = [
        shader, ":2: error: '#pragma': invalid stage for 'shader_stage' "
        "#pragma: ''\n"]


@inside_glslc_testsuite('PragmaShaderStage')
class TestFirstPSSBeforeNonPPCode(expect.ErrorMessage):
    """Tests that the first #pragma shader_stage() should appear before
    any non-preprocessing code."""

    shader = FileShader(
        """#version 310 es
        #ifndef REMOVE_UNUSED_FUNCTION
          int inc(int i) { return i + 1; }
        #endif
        #pragma shader_stage(vertex)
        void main() {
            gl_Position = vec4(1.);
        }
        """, '.glsl')
    glslc_args = ['-c', shader]
    expected_error = [
        shader, ":5: error: '#pragma': the first 'shader_stage' #pragma "
        'must appear before any non-preprocessing code\n']


@inside_glslc_testsuite('PragmaShaderStage')
class TestPSSMultipleErrors(expect.ErrorMessage):
    """Tests that if there are multiple errors, they are all reported."""

    shader = FileShader(
        """#version 310 es
        #pragma shader_stage(idontknow)
        #pragma shader_stage(vertex)
        void main() {
            gl_Position = vec4(1.);
        }
        #pragma shader_stage(fragment)
        """, '.glsl')
    glslc_args = ['-c', shader]
    expected_error = [
        shader, ":2: error: '#pragma': invalid stage for 'shader_stage' "
        "#pragma: 'idontknow'\n",
        shader, ":3: error: '#pragma': conflicting stages for 'shader_stage' "
        "#pragma: 'vertex' (was 'idontknow' at ", shader, ':2)\n',
        shader, ":7: error: '#pragma': conflicting stages for 'shader_stage' "
        "#pragma: 'fragment' (was 'idontknow' at ", shader, ':2)\n']


@inside_glslc_testsuite('PragmaShaderStage')
class TestSpacesAroundPSS(expect.ValidObjectFile):
    """Tests that spaces around #pragma shader_stage() works."""

    shader = FileShader(
        """#version 310 es
               #     pragma      shader_stage     (    vertex     )
        void main() {
            gl_Position = vec4(1.);
        }
        """, '.glsl')
    glslc_args = ['-c', shader]


@inside_glslc_testsuite('PragmaShaderStage')
class TestTabsAroundPSS(expect.ValidObjectFile):
    """Tests that tabs around #pragma shader_stage() works."""

    shader = FileShader(
        """#version 310 es
        \t\t#\tpragma\t\t\tshader_stage\t\t(\t\t\t\tvertex\t\t)\t\t\t\t
        void main() {
            gl_Position = vec4(1.);
        }
        """, '.glsl')
    glslc_args = ['-c', shader]


@inside_glslc_testsuite('PragmaShaderStage')
class TestPSSWithMacro(expect.ValidObjectFile):
    """Tests that #pragma shader_stage() works with macros."""

    shader = FileShader(
        """#version 310 es
        #if 0
          some random stuff here which can cause a problem
        #else
        # pragma shader_stage(vertex)
        #endif
        void main() {
            gl_Position = vec4(1.);
        }
        """, '.glsl')
    glslc_args = ['-c', shader]


@inside_glslc_testsuite('PragmaShaderStage')
class TestPSSWithCmdLineMacroDef(expect.ValidObjectFile):
    """Tests that macro definitions passed in from command line work."""

    shader = FileShader(
        """#version 310 es
        #ifdef IS_A_VERTEX_SHADER
        # pragma shader_stage(vertex)
        #else
        # pragma shader_stage(fragment)
        #endif
        void main() {
            gl_Position = vec4(1.);
        }
        """, '.glsl')
    glslc_args = ['-c', '-DIS_A_VERTEX_SHADER', shader]


@inside_glslc_testsuite('PragmaShaderStage')
class TestNoMacroExpansionInsidePSS(expect.ErrorMessage):
    """Tests that there is no macro expansion inside #pragma shader_stage()."""

    shader = FileShader(
        """#version 310 es
        #pragma shader_stage(STAGE_FROM_CMDLINE)
        void main() {
            gl_Position = vec4(1.);
        }
        """, '.glsl')
    glslc_args = ['-c', '-DSTAGE_FROM_CMDLINE=vertex', shader]

    expected_error = [
        shader, ":2: error: '#pragma': invalid stage for 'shader_stage' "
        "#pragma: 'STAGE_FROM_CMDLINE'\n"]


@inside_glslc_testsuite('PragmaShaderStage')
class TestPSSWithPoundLine310(expect.ErrorMessage):
    """Tests that #pragma shader_stage() works with #line."""

    shader = FileShader(
        """#version 310 es
        #pragma shader_stage(unknown)
        #line 42
        #pragma shader_stage(google)
        #line 100
        #pragma shader_stage(elgoog)
        void main() {
            gl_Position = vec4(1.);
        }
        """, '.glsl')
    glslc_args = ['-c', shader]

    expected_error = [
        shader, ":2: error: '#pragma': invalid stage for 'shader_stage' "
        "#pragma: 'unknown'\n",
        shader, ":42: error: '#pragma': conflicting stages for 'shader_stage' "
        "#pragma: 'google' (was 'unknown' at ", shader, ':2)\n',
        shader, ":100: error: '#pragma': conflicting stages for 'shader_stage' "
        "#pragma: 'elgoog' (was 'unknown' at ", shader, ':2)\n']


@inside_glslc_testsuite('PragmaShaderStage')
class TestPSSWithPoundLine150(expect.ErrorMessage):
    """Tests that #pragma shader_stage() works with #line.

    For older desktop versions, a #line directive specify the line number of
    the #line directive, not the next line.
    """

    shader = FileShader(
        """#version 150
        #pragma shader_stage(unknown)
        #line 42
        #pragma shader_stage(google)
        #line 100
        #pragma shader_stage(elgoog)
        void main() {
            gl_Position = vec4(1.);
        }
        """, '.glsl')
    glslc_args = ['-c', shader]

    expected_error = [
        shader, ":2: error: '#pragma': invalid stage for 'shader_stage' "
        "#pragma: 'unknown'\n",
        shader, ":43: error: '#pragma': conflicting stages for 'shader_stage' "
        "#pragma: 'google' (was 'unknown' at ", shader, ':2)\n',
        shader, ":101: error: '#pragma': conflicting stages for 'shader_stage' "
        "#pragma: 'elgoog' (was 'unknown' at ", shader, ':2)\n']


@inside_glslc_testsuite('PragmaShaderStage')
class ErrorBeforePragma(expect.ErrorMessage):
    """Tests that errors before pragmas are emitted."""

    shader = FileShader(
        """#version 310 es
        #something
        #pragma shader_stage(vertex)
        void main() {
            gl_Position = vec4(1.);
        }
        """, '.glsl')
    glslc_args = ['-c', shader]
    expected_error = [shader, ':2: error: \'#\' : invalid directive:',
                      ' something\n'
                      '1 error generated.\n']


@inside_glslc_testsuite('PragmaShaderStage')
class TestPSSfromIncludedFile(expect.ValidObjectFile):
    """Tests that #pragma shader_stage() from included files works."""

    environment = Directory('.', [
        File('a.glsl', '#version 140\n#include "b.glsl"\n'
             'void main() { gl_Position = vec4(1.); }\n'),
        File('b.glsl', '#pragma shader_stage(vertex)')])
    glslc_args = ['-c', 'a.glsl']


@inside_glslc_testsuite('PragmaShaderStage')
class TestConflictingPSSfromIncludingAndIncludedFile(expect.ErrorMessage):
    """Tests that conflicting #pragma shader_stage() from including and
    included files results in an error with the correct location spec."""

    environment = Directory('.', [
        File('a.vert',
             '#version 140\n'
             '#pragma shader_stage(fragment)\n'
             'void main() { gl_Position = vec4(1.); }\n'
             '#include "b.glsl"\n'),
        File('b.glsl', '#pragma shader_stage(vertex)')])
    glslc_args = ['-c', 'a.vert']

    expected_error = [
        "b.glsl:1: error: '#pragma': conflicting stages for 'shader_stage' "
        "#pragma: 'vertex' (was 'fragment' at a.vert:2)\n"]


@inside_glslc_testsuite('PragmaShaderStage')
class TestPSSWithFileNameBasedPoundLine(expect.ErrorMessage):
    """Tests that #pragma shader_stage() works with filename-based #line."""

    shader = FileShader(
        """#version 310 es
        #pragma shader_stage(unknown)
        #line 42 "abc"
        #pragma shader_stage(google)
        #line 100 "def"
        #pragma shader_stage(elgoog)
        void main() {
            gl_Position = vec4(1.);
        }
        """, '.glsl')
    glslc_args = ['-c', shader]

    expected_error = [
        shader, ":2: error: '#pragma': invalid stage for 'shader_stage' "
        "#pragma: 'unknown'\n",
        "abc:42: error: '#pragma': conflicting stages for 'shader_stage' "
        "#pragma: 'google' (was 'unknown' at ", shader, ':2)\n',
        "def:100: error: '#pragma': conflicting stages for 'shader_stage' "
        "#pragma: 'elgoog' (was 'unknown' at ", shader, ':2)\n']
