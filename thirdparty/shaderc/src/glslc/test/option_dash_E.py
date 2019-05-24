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


@inside_glslc_testsuite('OptionCapE')
class TestDashCapENoDefs(expect.StdoutMatch):
    """Tests -E without any defines."""

    shader = FileShader('#version 140\nvoid main(){}', '.vert')
    expected_stdout = '#version 140\nvoid main(){ }\n'
    glslc_args = ['-E', shader]


@inside_glslc_testsuite('OptionCapE')
class TestDashCapEGlslFileAccepted(expect.StdoutMatch):
    """Tests -E if we provide a .glsl file without an explicit stage."""

    shader = FileShader('#version 140\nvoid main(){}', '.glsl')
    expected_stdout = '#version 140\nvoid main(){ }\n'
    glslc_args = ['-E', shader]


@inside_glslc_testsuite('OptionCapE')
class TestDashCapESingleDefine(expect.StdoutMatch):
    """Tests -E with command-line define."""

    shader = FileShader('#version 140\nvoid main(){ int a = X; }', '.vert')
    expected_stdout = '#version 140\nvoid main(){ int a = 4;}\n'
    glslc_args = ['-DX=4', '-E', shader]


@inside_glslc_testsuite('OptionCapE')
class TestDashCapEExpansion(expect.StdoutMatch):
    """Tests -E with macro expansion."""

    shader = FileShader('''#version 140
#define X 4
void main() {
  int a = X;
}
''', '.vert')

    expected_stdout = '''#version 140

void main(){
  int a = 4;
}
'''
    glslc_args = ['-E', shader]


@inside_glslc_testsuite('OptionCapE')
class TestDashCapEFunctionMacro(expect.StdoutMatch):
    """Tests -E with function-style macro expansion."""

    shader = FileShader('''#version 140
#define X(A) 4+A
void main() {
  int a = X(1);
}
''', '.vert')

    expected_stdout = '''#version 140

void main(){
  int a = 4 + 1;
}
'''
    glslc_args = ['-E', shader]


@inside_glslc_testsuite('OptionCapE')
class TestDashCapEPragma(expect.StdoutMatch):
    """Tests -E to make sure pragmas get retained."""

    shader = FileShader('''#version 140
#pragma optimize(off)
void main() {
}
''', '.vert')

    expected_stdout = '''#version 140
#pragma optimize(off)
void main(){
}
'''
    glslc_args = ['-E', shader]


@inside_glslc_testsuite('OptionCapE')
class TestDashCapEExtension(expect.StdoutMatch):
    """Tests -E to make sure extensions get retained."""

    shader = FileShader('''#version 140
#extension foo: require
void main() {
}
''', '.vert')

    expected_stdout = '''#version 140
#extension foo : require
void main(){
}
'''
    glslc_args = ['-E', shader]


@inside_glslc_testsuite('OptionCapE')
class TestDashCapELine(expect.StdoutMatch):
    """Tests -E to make sure line numbers get retained."""

    shader = FileShader('''#version 140
#define X 4
#line X

#line 2 3

void main() {
}
''', '.vert')

    expected_stdout = '''#version 140

#line 4

#line 2 3

void main(){
}
'''
    glslc_args = ['-E', shader]


@inside_glslc_testsuite('OptionCapE')
class TestDashCapEError(expect.ErrorMessage):
    """Tests -E to make sure #errors get retained."""

    shader = FileShader('''#version 140
#if 1
  #error This is an error
#endif

void main() {
}
''', '.vert')

    expected_error = [
        shader, ':3: error: \'#error\' : This is an error\n',
        '1 error generated.\n']

    glslc_args = ['-E', shader]


@inside_glslc_testsuite('OptionCapE')
class TestDashCapEStdin(expect.StdoutMatch):
    """Tests to make sure -E works with stdin."""

    shader = StdinShader('''#version 140
void main() {
}
''')

    expected_stdout = '''#version 140
void main(){
}
'''
    glslc_args = ['-E', '-fshader-stage=vertex', shader]


@inside_glslc_testsuite('OptionCapE')
class TestDashCapEStdinDoesNotRequireShaderStage(expect.StdoutMatch):
    """Tests to make sure -E works with stdin even when no shader-stage
    is specified."""

    shader = StdinShader('''#version 140
void main() {
}
''')

    expected_stdout = '''#version 140
void main(){
}
'''
    glslc_args = ['-E', shader]


@inside_glslc_testsuite('OptionCapE')
class TestDashCapEMultipleFiles(expect.StdoutMatch):
    """Tests to make sure -E works with multiple files."""

    shader = StdinShader('''#version 140
void main() {
}
''')
    shader2 = FileShader('''#version 140
void function() {
}
''', '.vert')
    expected_stdout = '''#version 140
void main(){
}
#version 140
void function(){
}
'''
    glslc_args = ['-E', '-fshader-stage=vertex', shader, shader2]


@inside_glslc_testsuite('OptionCapE')
class TestDashCapEMultipleFilesWithoutStage(expect.StdoutMatch):
    """Tests to make sure -E works with multiple files even if we do not
    specify a stage."""

    shader = StdinShader('''#version 140
void main() {
}
''')
    shader2 = FileShader('''#version 140
void function() {
}
''', '.glsl')
    expected_stdout = '''#version 140
void main(){
}
#version 140
void function(){
}
'''
    glslc_args = ['-E', shader, shader2]


@inside_glslc_testsuite('OptionCapE')
class TestDashCapEOutputFile(expect.SuccessfulReturn, expect.ValidFileContents):
    """Tests to make sure -E works with output files."""

    shader = FileShader('''#version 140
void function() {
}
''', '.vert')
    expected_file_contents = '''#version 140
void function(){
}
'''
    target_filename = 'foo'
    glslc_args = ['-E', shader, '-ofoo']


@inside_glslc_testsuite('OptionCapE')
class TestDashCapEWithS(expect.StdoutMatch):
    """Tests -E in the presence of -S."""

    shader = FileShader('#version 140\nvoid main(){}', '.vert')
    expected_stdout = '#version 140\nvoid main(){ }\n'
    glslc_args = ['-E', '-S', shader]


@inside_glslc_testsuite('OptionCapE')
class TestMultipileDashCapE(expect.StdoutMatch):
    """Tests that using -E multiple times works."""

    shader = FileShader('#version 140\nvoid main(){}', '.vert')
    expected_stdout = '#version 140\nvoid main(){ }\n'
    glslc_args = ['-E', '-E', shader, '-E']


@inside_glslc_testsuite('OptionCapE')
class TestDashCapEAfterFile(expect.StdoutMatch):
    """Tests that using -E after the filename also works."""

    shader = FileShader('#version 140\nvoid main(){}', '.vert')
    expected_stdout = '#version 140\nvoid main(){ }\n'
    glslc_args = [shader, '-E']


@inside_glslc_testsuite('OptionCapE')
class TestDashCapEWithDashC(expect.StdoutMatch):
    """Tests to make sure -E works in the presence of -c."""

    shader = FileShader('''#version 140
void main() {
}
''', '.vert')
    shader2 = FileShader('''#version 140
void function() {
}
''', '.vert')
    expected_stdout = '''#version 140
void main(){
}
#version 140
void function(){
}
'''
    glslc_args = ['-E', '-c', shader, shader2]


@inside_glslc_testsuite('OptionCapE')
class TestDashCapEWithPPErrors(expect.ErrorMessage):
    """Tests to make sure -E outputs error messages for preprocessing errors."""

    shader = FileShader('''#version 310 es
#extension s enable // missing :
#defin A // Bad define
#if X // In glsl X must be defined for X to work.
      // Lack of endif.
void main() {
}
''', '.vert')
    expected_error = [
        shader, ':2: error: \'#extension\' : \':\' missing after extension',
        ' name\n',
        shader, ':3: error: \'#\' : invalid directive: defin\n',
        shader, ':4: error: \'preprocessor evaluation\' : undefined macro in',
        ' expression not allowed in es profile X\n',
        shader, ':8: error: \'\' : missing #endif\n',
        '4 errors generated.\n']
    glslc_args = ['-E', shader]


@inside_glslc_testsuite('OptionCapE')
class TestDashCapEStdinErrors(expect.ErrorMessage):
    """Tests that -E outputs error messages correctly for stdin input."""

    shader = StdinShader('''#version 310 es
#extension s enable // missing :
void main() {
}
''')
    expected_error = [
        '<stdin>:2: error: \'#extension\' : \':\' missing after extension',
        ' name\n',
        '1 error generated.\n']
    glslc_args = ['-E', shader]


# OpenGL compatibility fragment shader. Can be compiled to SPIR-V successfully
# when target environment is set to opengl_compat. Compilation will fail when
# target environment is set to other values. (gl_FragColor is predefined only
# in the compatibility profile.) But preprocessing should succeed with any
# target environment values.
def opengl_compat_frag_shader():
    return '''#version 330
uniform highp sampler2D tex;
void main(){
  gl_FragColor = texture2D(tex, vec2(0.0, 0.0));
}\n'''


@inside_glslc_testsuite('OptionCapE')
class TestDashCapEIgnoresTargetEnvOpengl(expect.StdoutMatch):
    """Tests that --target-env=opengl is ignored when -E is set."""

    shader = FileShader(opengl_compat_frag_shader(), '.frag')
    expected_stdout = opengl_compat_frag_shader()
    glslc_args = ['-E', '--target-env=opengl', shader]


@inside_glslc_testsuite('OptionCapE')
class TestDashCapEIgnoresTargetEnvOpenglCompat(expect.StdoutMatch):
    """Tests that --target-env=opengl_compat is ignored when -E is set."""

    shader = FileShader(opengl_compat_frag_shader(), '.frag')
    expected_stdout = opengl_compat_frag_shader()
    glslc_args = ['-E', '--target-env=opengl_compat', shader]
