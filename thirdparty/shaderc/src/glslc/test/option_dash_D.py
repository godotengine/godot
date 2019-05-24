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


@inside_glslc_testsuite('OptionCapD')
class TestDashCapDNoArg(expect.ErrorMessage):
    """Tests -D without macroname."""

    glslc_args = ['-D']
    expected_error = [
        "glslc: error: argument to '-D' is missing\n",
        'glslc: error: no input files\n']


@inside_glslc_testsuite('OptionCapD')
class TestDashCapDXeqY(expect.ValidObjectFile):
    """Tests -DX=Y."""

    shader = FileShader('#version 150\nvoid main(){X=vec4(1.);}', '.vert')
    glslc_args = ['-c', '-DX=gl_Position', shader]


@inside_glslc_testsuite('OptionCapD')
class TestDashCapDXeq(expect.ValidObjectFile):
    """Tests -DX=."""

    shader = FileShader('#version 150\nvoid main(){X}', '.vert')
    glslc_args = ['-c', '-DX=', shader]


@inside_glslc_testsuite('OptionCapD')
class TestDashCapDX(expect.ValidObjectFile):
    """Tests -DX."""

    shader = FileShader('#version 150\nvoid main(){X}', '.vert')
    glslc_args = ['-c', '-DX', shader]


@inside_glslc_testsuite('OptionCapD')
class TestDashCapDeq(expect.ErrorMessage):
    """Tests -D=.

    This is actually allowed by clang, though the resulting #define
    causes a preprocessing error.
    """

    shader = FileShader('#version 150\nvoid main(){}', '.vert')
    glslc_args = ['-c', '-D=', shader]
    # TODO(antiagainst): figure out what should we report as the line number
    # for errors in predefined macros and fix here.
    expected_error = [
        "<command line>:2: error: '#define' : must be followed by macro name\n",
        '1 error generated.\n']


@inside_glslc_testsuite('OptionCapD')
class TestMultipleDashCapD(expect.ValidObjectFile):
    """Tests multiple -D occurrences."""

    shader = FileShader('#version 150\nvoid main(){X Y a=Z;}', '.vert')
    glslc_args = ['-c', '-DX', '-DY=int', '-DZ=(1+2)', shader]


@inside_glslc_testsuite('OptionCapD')
class TestMultipleDashCapDOfSameName(expect.ValidObjectFile):
    """Tests multiple -D occurrences with same macro name."""

    shader = FileShader('#version 150\nvoid main(){X Y a=Z;}', '.vert')
    glslc_args = ['-c', '-DX=main', '-DY=int', '-DZ=(1+2)', '-DX', shader]


@inside_glslc_testsuite('OptionCapD')
class TestDashCapDGL_(expect.ErrorMessage):
    """Tests that we cannot -D macros beginning with GL_."""

    shader = FileShader('#version 150\nvoid main(){}', '.vert')
    glslc_args = ['-DGL_ES=1', shader]
    expected_error = [
        "glslc: error: names beginning with 'GL_' cannot be "
        'defined: -DGL_ES=1\n']


@inside_glslc_testsuite('OptionCapD')
class TestDashCapDReservedMacro(expect.WarningMessage):
    """Tests that we cannot -D GLSL's predefined macros."""

    shader = FileShader('#version 150\nvoid main(){}', '.vert')
    # Consecutive underscores are banned anywhere in the name.
    glslc_args = [
        '-D__LINE__=1', '-Dmid__dle', '-Dend__', '-D_single_is_valid_', shader]

    w = 'glslc: warning: names containing consecutive underscores are reserved: '
    expected_warning = [w, '-D__LINE__=1\n', w, '-Dmid__dle\n', w, '-Dend__\n']


@inside_glslc_testsuite('OptionCapD')
class TestDashCapDWithVersion(expect.ErrorMessage):
    """Tests -D works well when #version is present."""

    shader = FileShader(
        """#version 310 es
        void main(){X}
        void foo(){Y}""", '.vert')
    glslc_args = ['-DX=', '-DY=return 3;', shader]

    expected_error = [
        shader, ":3: error: 'return' : void function cannot return a value\n",
        '1 error generated.\n']


@inside_glslc_testsuite('OptionCapD')
class TestDashCapDWithCommentBeforeVersion(expect.ErrorMessage):
    """Tests -D works well with #version preceded by comments."""

    shader = FileShader(
        """// comment 1
        /*
         * comment 2
         */
        #version 450 core
        void main(){X}
        void foo(){Y}""", '.vert')
    glslc_args = ['-DX=', '-DY=return 3;', shader]

    expected_error = [
        shader, ":7: error: 'return' : void function cannot return a value\n",
        '1 error generated.\n']


@inside_glslc_testsuite('OptionCapD')
class TestDashCapDWithCommentAfterVersion(expect.ErrorMessage):
    """Tests -D works well with #version followed by comments."""

    shader = FileShader(
        """

        #version 150 core /*
        comment
        */
        void main(){X}
        void foo(){Y}""", '.vert')
    glslc_args = ['-DX=', '-DY=return 3;', shader]

    expected_error = [
        shader, ":7: error: 'return' : void function cannot return a value\n",
        '1 error generated.\n']


@inside_glslc_testsuite('OptionCapD')
class TestDashCapDWithDashStd(expect.ErrorMessageSubstr):
    """Tests -D works well with -std."""

    shader = FileShader('void main(){X}\nvoid foo(){Y}', '.vert')
    glslc_args = ['-DX=', '-DY=return 3;', '-std=310es', shader]

    expected_error_substr = [
        shader, ":2: error: 'return' : void function cannot return a value\n",
        '1 error generated.\n']


@inside_glslc_testsuite('OptionCapD')
class TestDashCapDWithDashStdAndVersion(expect.ErrorMessage):
    """Tests -D works well with both -std and #version."""

    shader = FileShader(
        """#version 310 es
        void main(){X}
        void foo(){Y}""", '.vert')
    glslc_args = ['-DX=', '-DY=return 3;', '-std=450core', shader]

    expected_error = [
        shader, ': warning: (version, profile) forced to be (450, ',
        'core), while in source code it is (310, es)\n',
        shader, ":3: error: 'return' : void function cannot return a value\n",
        '1 warning and 1 error generated.\n']


@inside_glslc_testsuite('OptionCapD')
class TestDashCapDWithDashStdAndVersionAndComments(expect.ErrorMessage):
    """Tests -D works well with -std, #version, and comments around it."""

    shader = FileShader(
        """// comment before

        #version 310 es /* comment after
        */

        void main(){X}
        void foo(){Y}""", '.vert')
    glslc_args = ['-DX=', '-DY=return 3;', '-std=450core', shader]

    expected_error = [
        shader, ': warning: (version, profile) forced to be (450, core), while '
        'in source code it is (310, es)\n',
        shader, ":7: error: 'return' : void function cannot return a value\n",
        '1 warning and 1 error generated.\n']


@inside_glslc_testsuite('OptionCapD')
class TestDashCapDWithDashE(expect.ReturnCodeIsZero,
                            expect.StdoutMatch):
    """Tests -E outputs expanded -D macros."""

    shader = FileShader(
        """
        void main(){Y}
""", '.vert')
    glslc_args = ['-DY=return 3;', '-E', '-std=450core', shader]

    expected_stdout =  [
        """
        void main(){ return 3;}
"""]

@inside_glslc_testsuite('OptionCapD')
class TestDashCapDWithDashEIfDef(expect.ReturnCodeIsZero,
                                 expect.StdoutMatch):
    """Tests -E processes -DX #ifdefs correctly."""

    shader = FileShader(
        """
        #ifdef X
          void f() { }
        #else
          void f() { int x; }
        #endif
        void main(){ return 3;}
""", '.vert')
    glslc_args = ['-DX', '-E', '-std=450core', shader]

    expected_stdout =  [
        """

          void f(){ }



        void main(){ return 3;}
"""]

@inside_glslc_testsuite('OptionCapD')
class TestDashCapDWithDashEIfNDef(expect.ReturnCodeIsZero,
                                  expect.StdoutMatch):
    """Tests -E processes -DX #ifndefs correctly."""

    shader = FileShader(
        """
        #ifndef X
          void f() { }
        #else
          void f() { int x; }
        #endif
        void main(){ return 3;}
""", '.vert')
    glslc_args = ['-DX', '-E', '-std=450core', shader]

    expected_stdout =  [
        """



          void f(){ int x;}

        void main(){ return 3;}
"""]


@inside_glslc_testsuite('OptionCapD')
class TestDashCapDWithDashEEqIfDef(expect.ReturnCodeIsZero,
                                   expect.StdoutMatch):
    """Tests -E processes -DX= #ifdefs correctly."""

    shader = FileShader(
        """
        #ifdef X
          void f() { }
        #else
          void f() { int x; }
        #endif
        void main(){ return 3;}
""", '.vert')
    glslc_args = ['-DX=', '-E', '-std=450core', shader]

    expected_stdout =  [
        """

          void f(){ }



        void main(){ return 3;}
"""]

@inside_glslc_testsuite('OptionCapD')
class TestDashCapDWithDashEEqIfNDef(expect.ReturnCodeIsZero,
                                    expect.StdoutMatch):
    """Tests -E processes -DX= #ifndefs correctly."""

    shader = FileShader(
        """
        #ifndef X
          void f() { }
        #else
          void f() { int x; }
        #endif
        void main(){ return 3;}
""", '.vert')
    glslc_args = ['-DX=', '-E', '-std=450core', shader]

    expected_stdout =  [
        """



          void f(){ int x;}

        void main(){ return 3;}
"""]

@inside_glslc_testsuite('OptionCapD')
class TestDashCapDWithDashEFunctionMacro(expect.ReturnCodeIsZero,
                                         expect.StdoutMatch):
    """Tests -E processes -D function macros correctly."""

    shader = FileShader(
        """
        void main(){ return FOO(3);}
""", '.vert')
    glslc_args = ['-DFOO(x)=(2*x+1)*x*x', '-E', '-std=450core', shader]

    expected_stdout =  [
        """
        void main(){ return(2 * 3 + 1)* 3 * 3;}
"""]

@inside_glslc_testsuite('OptionCapD')
class TestDashCapDWithDashENestedMacro(expect.ReturnCodeIsZero,
                                       expect.StdoutMatch):
    """Tests -E processes referencing -D options correctly."""

    shader = FileShader(
        """
        void main(){ return X;}
""", '.vert')
    glslc_args = ['-DY=4', '-DX=Y', '-E', '-std=450core', shader]

    expected_stdout =  [
        """
        void main(){ return 4;}
"""]
