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


@inside_glslc_testsuite('#line')
class TestPoundVersion310InIncludingFile(
    expect.ReturnCodeIsZero, expect.StdoutMatch, expect.StderrMatch):
    """Tests that #line directives follows the behavior of version 310
    (specifying the line number for the next line) when we find a
    #version 310 directive in the including file."""

    environment = Directory('.', [
        File('a.vert', '#version 310 es\n#include "b.glsl"\n'),
        File('b.glsl', 'void main() {}\n')])
    glslc_args = ['-E', 'a.vert']

    expected_stderr = ''
    expected_stdout = \
"""#version 310 es
#extension GL_GOOGLE_include_directive : enable
#line 1 "a.vert"

#line 1 "b.glsl"
void main(){ }
#line 3 "a.vert"

"""


@inside_glslc_testsuite('#line')
class TestPoundVersion150InIncludingFile(
    expect.ReturnCodeIsZero, expect.StdoutMatch, expect.StderrMatch):
    """Tests that #line directives follows the behavior of version 150
    (specifying the line number for itself) when we find a #version 150
    directive in the including file."""

    environment = Directory('.', [
        File('a.vert', '#version 150\n#include "b.glsl"\n'),
        File('b.glsl', 'void main() {}\n')])
    glslc_args = ['-E', 'a.vert']

    expected_stderr = ''
    expected_stdout = \
"""#version 150
#extension GL_GOOGLE_include_directive : enable
#line 0 "a.vert"

#line 0 "b.glsl"
void main(){ }
#line 2 "a.vert"

"""


@inside_glslc_testsuite('#line')
class TestPoundVersionSyntaxErrorInIncludingFile(expect.ErrorMessageSubstr):
    """Tests that error message for #version directive has the correct
    filename and line number."""

    environment = Directory('.', [
        File('a.vert', '#version abc def\n#include "b.glsl"\n'),
        File('b.glsl', 'void main() {}\n')])
    glslc_args = ['-E', 'a.vert']

    expected_error_substr = [
        "a.vert:1: error: '#version' : must occur first in shader\n",
        "a.vert:1: error: '#version' : must be followed by version number\n",
        "a.vert:1: error: '#version' : bad profile name; use es, core, or "
        "compatibility\n",
    ]


# TODO(antiagainst): now #version in included files results in an error.
# Fix this after #version in included files are supported.
@inside_glslc_testsuite('#line')
class TestPoundVersion310InIncludedFile(expect.ErrorMessageSubstr):
    """Tests that #line directives follows the behavior of version 310
    (specifying the line number for the next line) when we find a
    #version 310 directive in the included file."""

    environment = Directory('.', [
        File('a.vert', '#include "b.glsl"\nvoid main() {}'),
        File('b.glsl', '#version 310 es\n')])
    glslc_args = ['-E', 'a.vert']

    expected_error_substr = [
        "b.glsl:1: error: '#version' : must occur first in shader\n"
    ]


# TODO(antiagainst): now #version in included files results in an error.
# Fix this after #version in included files are supported.
@inside_glslc_testsuite('#line')
class TestPoundVersion150InIncludedFile(expect.ErrorMessageSubstr):
    """Tests that #line directives follows the behavior of version 150
    (specifying the line number for itself) when we find a #version 150
    directive in the included file."""

    environment = Directory('.', [
        File('a.vert', '#include "b.glsl"\nvoid main() {}'),
        File('b.glsl', '#version 150\n')])
    glslc_args = ['-E', 'a.vert']

    expected_error_substr = [
        "b.glsl:1: error: '#version' : must occur first in shader\n"
    ]


@inside_glslc_testsuite('#line')
class TestSpaceAroundPoundVersion310InIncludingFile(
    expect.ReturnCodeIsZero, expect.StdoutMatch, expect.StderrMatch):
    """Tests that spaces around #version & #include directive doesn't matter."""

    environment = Directory('.', [
        File('a.vert', '\t #  \t version   310 \t  es\n#\tinclude  "b.glsl"\n'),
        File('b.glsl', 'void main() {}\n')])
    glslc_args = ['-E', 'a.vert']

    expected_stderr = ''
    expected_stdout = \
"""#version 310 es
#extension GL_GOOGLE_include_directive : enable
#line 1 "a.vert"

#line 1 "b.glsl"
void main(){ }
#line 3 "a.vert"

"""


@inside_glslc_testsuite('#line')
class TestSpaceAroundPoundVersion150InIncludingFile(
    expect.ReturnCodeIsZero, expect.StdoutMatch, expect.StderrMatch):
    """Tests that spaces around #version & #include directive doesn't matter."""

    environment = Directory('.', [
        File('a.vert', '  \t #\t\tversion\t   150\t \n# include \t "b.glsl"\n'),
        File('b.glsl', 'void main() {}\n')])
    glslc_args = ['-E', 'a.vert']

    expected_stderr = ''
    expected_stdout = \
"""#version 150
#extension GL_GOOGLE_include_directive : enable
#line 0 "a.vert"

#line 0 "b.glsl"
void main(){ }
#line 2 "a.vert"

"""


@inside_glslc_testsuite('#line')
class TestPoundLineWithForcedVersion310(
    expect.ReturnCodeIsZero, expect.StdoutMatch, expect.StderrMatch):
    """Tests that #line directives follows the behavior for the version
    specified via command-line."""

    environment = Directory('.', [
        File('a.vert', '#include "b.glsl"\n'),
        File('b.glsl', 'void main() {}\n')])
    glslc_args = ['-E', '-std=310es', 'a.vert']

    expected_stderr = ''
    expected_stdout = \
"""#extension GL_GOOGLE_include_directive : enable
#line 1 "a.vert"
#line 1 "b.glsl"
void main(){ }
#line 2 "a.vert"

"""


@inside_glslc_testsuite('#line')
class TestPoundLineWithForcedVersion150(
    expect.ReturnCodeIsZero, expect.StdoutMatch, expect.StderrMatch):
    """Tests that #line directives follows the behavior for the version
    specified via command-line."""

    environment = Directory('.', [
        File('a.vert', '#include "b.glsl"\n'),
        File('b.glsl', 'void main() {}\n')])
    glslc_args = ['-E', '-std=150', 'a.vert']

    expected_stderr = ''
    expected_stdout = \
"""#extension GL_GOOGLE_include_directive : enable
#line 0 "a.vert"
#line 0 "b.glsl"
void main(){ }
#line 1 "a.vert"

"""


@inside_glslc_testsuite('#line')
class TestPoundLineWithForcedDifferentVersion(
    expect.ReturnCodeIsZero, expect.StdoutMatch, expect.StderrMatch):
    """Tests that #line directives follows the behavior for the version
    specified via command-line, even if there is a version specification
    in the source code."""

    environment = Directory('.', [
        File('a.vert', '#version 150\n#include "b.glsl"\n'),
        File('b.glsl', 'void main() {}\n')])
    glslc_args = ['-E', '-std=310es', 'a.vert']

    expected_stderr = ''
    expected_stdout = \
"""#version 150
#extension GL_GOOGLE_include_directive : enable
#line 1 "a.vert"

#line 1 "b.glsl"
void main(){ }
#line 3 "a.vert"

"""


@inside_glslc_testsuite('#line')
class TestErrorsFromMultipleFiles(expect.ErrorMessage):
    """Tests that errors from different files have the correct error message
    filename specification."""
    including_file = '''#version 310 es
#include "error.glsl"
int no_return() {}
#include "main.glsl"
'''

    environment = Directory('.', [
        File('a.vert', including_file),
        File('error.glsl', 'int unknown_identifier(int) { return a; }'),
        File('main.glsl', 'void main() {\n  int b = 1.5;\n}')])
    glslc_args = ['-c', 'a.vert']

    expected_error = [
        "error.glsl:1: error: 'a' : undeclared identifier\n",
        "error.glsl:1: error: 'return' : type does not match, or is not "
        "convertible to, the function's return type\n",
        "a.vert:3: error: '' : function does not return a value: no_return\n",
        "main.glsl:2: error: '=' :  cannot convert from ' const float' to "
        "' temp highp int'\n",
        "4 errors generated.\n"]


@inside_glslc_testsuite('#line')
class TestExplicitPoundLineWithPoundInclude(
    expect.ReturnCodeIsZero, expect.StdoutMatch, expect.StderrMatch):
    """Tests that #line works correctly in the presence of #include (which
    itself will generate some #line directives."""
    including_file = '''#version 310 es
#line 10000 "injected.glsl"
int plus1(int a) { return a + 1; }
#include "inc.glsl"
int plus2(int a) { return a + 2; }
#line 55555
#include "main.glsl"
'''

    environment = Directory('.', [
        File('a.vert', including_file),
        File('inc.glsl', 'int inc(int a) { return a + 1; }'),
        File('main.glsl', 'void main() {\n  gl_Position = vec4(1.);\n}')])
    glslc_args = ['-E', 'a.vert']

    expected_stderr = ''
    expected_stdout = '''#version 310 es
#extension GL_GOOGLE_include_directive : enable
#line 1 "a.vert"

#line 10000 "injected.glsl"
int plus1(int a){ return a + 1;}
#line 1 "inc.glsl"
 int inc(int a){ return a + 1;}
#line 10002 "injected.glsl"
 int plus2(int a){ return a + 2;}
#line 55555
#line 1 "main.glsl"
 void main(){
  gl_Position = vec4(1.);
}
#line 55556 "injected.glsl"

'''
