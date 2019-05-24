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
from placeholder import FileShader, StdinShader


@inside_glslc_testsuite('ErrorMessages')
class MultipleErrors(expect.ErrorMessage):
    """Test Multiple error messages generated."""
    shader = FileShader('#version 140\nint main() {}', '.vert')
    glslc_args = ['-c', shader]
    expected_error = [
        shader, ":2: error: 'int' :  entry point cannot return a value\n",
        shader, ":2: error: '' : function does not return a value: main\n",
        '2 errors generated.\n']


@inside_glslc_testsuite('ErrorMessages')
class OneError(expect.ErrorMessage):
    """Tests that only one error message is generated correctly."""

    shader = FileShader(
        """#version 140
    int a() {
    }
    void main() {
      int x = a();
    }
    """, '.vert')
    glslc_args = ['-c', shader]

    expected_error = [
        shader, ":2: error: '' : function does not return a value: a\n",
        '1 error generated.\n']


@inside_glslc_testsuite('ErrorMessages')
class ManyLineError(expect.ErrorMessage):
    """Tests that only one error message is generated correctly."""

    shader = FileShader(
        """#version 140










    int a() {
    }
    void main() {
      int x = a();
    }
    """, '.vert')
    glslc_args = ['-c', shader]

    expected_error = [
        shader, ":12: error: '' : function does not return a value: a\n",
        '1 error generated.\n']

@inside_glslc_testsuite('ErrorMessages')
class GlobalWarning(expect.WarningMessage):
    """Tests that a warning message without file/line number is emitted."""

    shader = FileShader(
        """#version 550
    void main() {
    }
    """, '.vert')
    glslc_args = ['-c', '-std=400', shader]

    expected_warning = [
            shader, ': warning: (version, profile) forced to be (400, none),'
            ' while in source code it is (550, none)\n1 warning generated.\n']

@inside_glslc_testsuite('ErrorMessages')
class SuppressedGlobalWarning(expect.SuccessfulReturn):
    """Tests that warning messages without file/line numbers are suppressed
    with -w."""

    shader = FileShader(
        """#version 550
    void main() {
    }
    """, '.vert')
    glslc_args = ['-c', '-std=400', shader, '-w']


@inside_glslc_testsuite('ErrorMessages')
class GlobalWarningAsError(expect.ErrorMessage):
    """Tests that with -Werror an error warning message without file/line
    number is emitted instead of a warning."""

    shader = FileShader(
        """#version 550
    void main() {
    }
    """, '.vert')
    glslc_args = ['-c', '-std=400', shader, '-Werror']

    expected_error= [
            shader, ': error: (version, profile) forced to be (400, none),'
            ' while in source code it is (550, none)\n1 error generated.\n']

@inside_glslc_testsuite('ErrorMessages')
class WarningOnLine(expect.WarningMessage):
    """Tests that a warning message with a file/line number is emitted."""

    shader = FileShader(
        """#version 400
    layout(location = 0) attribute float x;
    void main() {
    }
    """, '.vert')
    glslc_args = ['-c', shader]

    expected_warning = [
        shader, ':2: warning: attribute deprecated in version 130; ',
        'may be removed in future release\n1 warning generated.\n']

@inside_glslc_testsuite('ErrorMessages')
class SuppressedWarningOnLine(expect.SuccessfulReturn):
    """Tests that a warning message with a file/line number is suppressed in the
    presence of -w."""

    shader = FileShader(
        """#version 400
    layout(location = 0) attribute float x;
    void main() {
    }
    """, '.vert')
    glslc_args = ['-c', shader, '-w']

@inside_glslc_testsuite('ErrorMessages')
class WarningOnLineAsError(expect.ErrorMessage):
    """Tests that with -Werror an error message with a file/line
    number is emitted instead of a warning."""

    shader = FileShader(
        """#version 400
    layout(location = 0) attribute float x;
    void main() {
    }
    """, '.vert')
    glslc_args = ['-c', shader, '-Werror']

    expected_error = [
        shader, ':2: error: attribute deprecated in version 130; ',
        'may be removed in future release\n1 error generated.\n']

@inside_glslc_testsuite('ErrorMessages')
class WarningAndError(expect.ErrorMessage):
    """Tests that both warnings and errors are emitted together."""

    shader = FileShader(
        """#version 400
    layout(location = 0) attribute float x;
    int main() {
    }
    """, '.vert')
    glslc_args = ['-c', shader]

    expected_error = [
        shader, ':2: warning: attribute deprecated in version 130; ',
        'may be removed in future release\n',
        shader, ":3: error: 'int' :  entry point cannot return a value\n",
        shader, ":3: error: '' : function does not return a value: main\n",
        '1 warning and 2 errors generated.\n']

@inside_glslc_testsuite('ErrorMessages')
class SuppressedWarningAndError(expect.ErrorMessage):
    """Tests that only warnings are suppressed in the presence of -w."""

    shader = FileShader(
        """#version 400
    layout(location = 0) attribute float x;
    int main() {
    }
    """, '.vert')
    glslc_args = ['-c', shader, '-w']

    expected_error = [
        shader, ":3: error: 'int' :  entry point cannot return a value\n",
        shader, ":3: error: '' : function does not return a value: main\n",
        '2 errors generated.\n']

@inside_glslc_testsuite('ErrorMessages')
class WarningAsErrorAndError(expect.ErrorMessage):
    """Tests that with -Werror an warnings and errors are emitted as errors."""

    shader = FileShader(
        """#version 400
    layout(location = 0) attribute float x;
    int main() {
    }
    """, '.vert')
    glslc_args = ['-c', shader, '-Werror']

    expected_error = [
        shader, ':2: error: attribute deprecated in version 130; ',
        'may be removed in future release\n',
        shader, ":3: error: 'int' :  entry point cannot return a value\n",
        shader, ":3: error: '' : function does not return a value: main\n",
        '3 errors generated.\n']

@inside_glslc_testsuite('ErrorMessages')
class StdinErrorMessages(expect.StdoutMatch, expect.StderrMatch):
    """Tests that error messages using input from stdin are correct."""

    shader = StdinShader(
        """#version 140
    int a() {
    }
    void main() {
      int x = a();
    }
    """)
    glslc_args = ['-c', '-fshader-stage=vertex', shader]

    expected_stdout = ''
    expected_stderr = [
        "<stdin>:2: error: '' : function does not return a value: a\n",
        '1 error generated.\n']

@inside_glslc_testsuite('ErrorMessages')
class WarningAsErrorMultipleFiles(expect.ErrorMessage):
    """Tests that with -Werror multiple files emit errors instead of warnings.
    """

    shader = FileShader(
        """#version 400
    layout(location = 0) attribute float x;
    void main() {
    }
    """, '.vert')

    shader2 = FileShader(
        """#version 550
    void main() {
    }
    """, '.vert')

    glslc_args = ['-c', '-std=400', shader, '-Werror', shader2]

    expected_error = [
        shader, ':2: error: attribute deprecated in version 130; ',
        'may be removed in future release\n',
        shader2, ': error: (version, profile) forced to be (400, none),'
            ' while in source code it is (550, none)\n',
        '2 errors generated.\n']


@inside_glslc_testsuite('ErrorMessages')
class SuppressedWarningAsError(expect.SuccessfulReturn):
    """Tests that nothing is returned in the presence of -w -Werror."""

    shader = FileShader(
        """#version 400
    layout(location = 0) attribute float x;
    void main() {
    }
    """, '.vert')
    glslc_args = ['-c', shader, '-w', '-Werror']

@inside_glslc_testsuite('ErrorMessages')
class MultipleSuppressed(expect.SuccessfulReturn):
    """Tests that multiple -w arguments are supported."""

    shader = FileShader(
        """#version 400
    layout(location = 0) attribute float x;
    void main() {
    }
    """, '.vert')
    glslc_args = ['-w', '-c', shader, '-w', '-w', '-w']

@inside_glslc_testsuite('ErrorMessages')
class MultipleSuppressedFiles(expect.SuccessfulReturn):
    """Tests that -w suppresses warnings from all files."""

    shader = FileShader(
        """#version 400
    layout(location = 0) attribute float x;
    void main() {
    }
    """, '.vert')

    shader2 = FileShader(
        """#version 400
    layout(location = 0) attribute float x;
    void main() {
    }
    """, '.vert')
    glslc_args = ['-w', '-c', shader, shader2]
