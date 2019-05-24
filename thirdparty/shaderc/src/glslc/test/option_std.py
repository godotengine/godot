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


def core_vert_shader_without_version():
    # gl_ClipDistance doesn't exist in es profile (at least until 3.10).
    return 'void main() { gl_ClipDistance[0] = 5.; }'


def core_frag_shader_without_version():
    # gl_SampleID appears in core profile from 4.00.
    # gl_sampleID doesn't exsit in es profile (at least until 3.10).
    return 'void main() { int temp = gl_SampleID; }'


def hlsl_compute_shader_with_barriers():
    # Use "main" to avoid the need for -fentry-point
    return 'void main() { AllMemoryBarrierWithGroupSync(); }'


@inside_glslc_testsuite('OptionStd')
class TestStdNoArg(expect.ErrorMessage):
    """Tests -std alone."""

    glslc_args = ['-std']
    expected_error = ["glslc: error: unknown argument: '-std'\n"]


@inside_glslc_testsuite('OptionStd')
class TestStdEqNoArg(expect.ErrorMessage):
    """Tests -std= with no argument."""

    glslc_args = ['-std=']
    expected_error = ["glslc: error: invalid value '' in '-std='\n"]


@inside_glslc_testsuite('OptionStd')
class TestStdEqSpaceArg(expect.ErrorMessage):
    """Tests -std= <version-profile>."""

    shader = FileShader(core_frag_shader_without_version(), '.frag')
    glslc_args = ['-c', '-std=', '450core', shader]
    expected_error = ["glslc: error: invalid value '' in '-std='\n"]


# TODO(dneto): The error message changes with different versions of glslang.
@inside_glslc_testsuite('OptionStd')
class TestMissingVersionAndStd(expect.ErrorMessageSubstr):
    """Tests that missing both #version and -std results in errors."""

    shader = FileShader(core_frag_shader_without_version(), '.frag')
    glslc_args = ['-c', shader]
    expected_error_substr = ['error:']


@inside_glslc_testsuite('OptionStd')
class TestMissingVersionButHavingStd(expect.ValidObjectFile):
    """Tests that correct -std fixes missing #version."""

    shader = FileShader(core_frag_shader_without_version(), '.frag')
    glslc_args = ['-c', '-std=450core', shader]


@inside_glslc_testsuite('OptionStd')
class TestGLSL460(expect.ValidObjectFile):
    """Tests that GLSL version 4.6 is supported."""

    shader = FileShader(core_frag_shader_without_version(), '.frag')
    glslc_args = ['-c', '-std=460', shader]


@inside_glslc_testsuite('OptionStd')
class TestGLSL460Core(expect.ValidObjectFile):
    """Tests that GLSL version 4.6 core profile is supported."""

    shader = FileShader(core_frag_shader_without_version(), '.frag')
    glslc_args = ['-c', '-std=460core', shader]


@inside_glslc_testsuite('OptionStd')
class TestESSL320(expect.ValidObjectFile):
    """Tests that ESSL version 3.2 is supported."""

    shader = FileShader(core_frag_shader_without_version(), '.frag')
    glslc_args = ['-c', '-std=320es', shader]


@inside_glslc_testsuite('OptionStd')
class TestStdIgnoredInHlsl(expect.ValidObjectFile):
    """Tests HLSL compilation ignores -std."""

    # Compute shaders are not available in OpenGL 150
    shader = FileShader(hlsl_compute_shader_with_barriers(), '.comp')
    glslc_args = ['-c', '-x', 'hlsl', '-std=150', shader]


@inside_glslc_testsuite('OptionStd')
class TestMissingVersionAndWrongStd(expect.ErrorMessage):
    """Tests missing #version and wrong -std results in errors."""

    shader = FileShader(core_frag_shader_without_version(), '.frag')
    glslc_args = ['-c', '-std=310es', shader]
    expected_error = [
        shader, ":1: error: 'gl_SampleID' : required extension not requested: "
        'GL_OES_sample_variables\n1 error generated.\n']


@inside_glslc_testsuite('OptionStd')
class TestConflictingVersionAndStd(expect.ValidObjectFileWithWarning):
    """Tests that with both #version and -std, -std takes precedence."""

    # Wrong #version here on purpose.
    shader = FileShader(
        '#version 310 es\n' + core_frag_shader_without_version(), '.frag')
    # -std overwrites the wrong #version.
    glslc_args = ['-c', '-std=450core', shader]

    expected_warning = [
        shader, ': warning: (version, profile) forced to be (450, core), while '
        'in source code it is (310, es)\n1 warning generated.\n']


@inside_glslc_testsuite('OptionStd')
class TestMultipleStd(expect.ValidObjectFile):
    """Tests that for multiple -std, the last one takes effect."""

    shader = FileShader(core_frag_shader_without_version(), '.frag')
    glslc_args = ['-c', '-std=100', '-std=310es', shader, '-std=450core']


@inside_glslc_testsuite('OptionStd')
class TestMultipleFiles(expect.ValidObjectFileWithWarning):
    """Tests that -std covers all files."""

    shader1 = FileShader(core_frag_shader_without_version(), '.frag')
    shader2 = FileShader(core_vert_shader_without_version(), '.vert')
    shader3 = FileShader(
        '#version 310 es\n' + core_frag_shader_without_version(), '.frag')
    glslc_args = ['-c', '-std=450core', shader1, shader2, shader3]

    expected_warning = [
        shader3, ': warning: (version, profile) forced to be (450, '
        'core), while in source code it is (310, es)\n'
        '1 warning generated.\n']


@inside_glslc_testsuite('OptionStd')
class TestUnkownProfile(expect.ErrorMessage):
    """Tests that -std rejects unknown profile."""

    shader = FileShader(core_frag_shader_without_version(), '.frag')
    glslc_args = ['-c', '-std=450google', shader]
    expected_error = [
        "glslc: error: invalid value '450google' in '-std=450google'\n"]


@inside_glslc_testsuite('OptionStd')
class TestUnkownVersion(expect.ErrorMessage):
    """Tests that -std rejects unknown version."""

    shader = FileShader(core_frag_shader_without_version(), '.frag')
    glslc_args = ['-c', '-std=42core', shader]
    expected_error = [
        "glslc: error: invalid value '42core' in '-std=42core'\n"]


@inside_glslc_testsuite('OptionStd')
class TestTotallyWrongStdValue(expect.ErrorMessage):
    """Tests that -std rejects totally wrong -std value."""

    shader = FileShader(core_vert_shader_without_version(), '.vert')
    glslc_args = ['-c', '-std=wrong42', shader]

    expected_error = [
        "glslc: error: invalid value 'wrong42' in '-std=wrong42'\n"]


@inside_glslc_testsuite('OptionStd')
class TestVersionInsideSlashSlashComment(expect.ValidObjectFileWithWarning):
    """Tests that -std substitutes the correct #version string."""

    # The second #version string should be substituted and this shader
    # should compile successfully with -std=450core.
    shader = FileShader(
        '// #version 310 es\n#version 310 es\n' +
        core_vert_shader_without_version(), '.vert')
    glslc_args = ['-c', '-std=450core', shader]

    expected_warning = [
        shader, ': warning: (version, profile) forced to be (450, core), while '
        'in source code it is (310, es)\n1 warning generated.\n']


@inside_glslc_testsuite('OptionStd')
class TestVersionInsideSlashStarComment(expect.ValidObjectFileWithWarning):
    """Tests that -std substitutes the correct #version string."""

    # The second #version string should be substituted and this shader
    # should compile successfully with -std=450core.
    shader = FileShader(
        '/* #version 310 es */\n#version 310 es\n' +
        core_vert_shader_without_version(), '.vert')
    glslc_args = ['-c', '-std=450core', shader]

    expected_warning = [
        shader, ': warning: (version, profile) forced to be (450, core), while '
        'in source code it is (310, es)\n1 warning generated.\n']


@inside_glslc_testsuite('OptionStd')
class TestCommentBeforeVersion(expect.ValidObjectFileWithWarning):
    """Tests that comments before #version (same line) is correctly handled."""

    shader = FileShader(
        '/* some comment */ #version 150\n' +
        core_vert_shader_without_version(), '.vert')
    glslc_args = ['-c', '-std=450', shader]

    expected_warning = [
        shader, ': warning: (version, profile) forced to be (450, none), while '
        'in source code it is (150, none)\n1 warning generated.\n']


@inside_glslc_testsuite('OptionStd')
class TestCommentAfterVersion(expect.ValidObjectFileWithWarning):
    """Tests that multiple-line comments after #version is correctly handled."""

    shader = FileShader(
        '#version 150 compatibility ' +
        '/* start \n second line \n end */\n' +
        core_vert_shader_without_version(), '.vert')
    glslc_args = ['-c', '-std=450core', shader]

    expected_warning = [
        shader, ': warning: (version, profile) forced to be (450, core), while '
        'in source code it is (150, compatibility)\n1 warning generated.\n']


# The following test case is disabled because of a bug in glslang.
# When checking non-newline whitespaces, glslang only recognizes
# ' ' and '\t', leaving '\v' and '\f' unhandled. The following test
# case exposes this problem. It should be turned on once a fix for
# glslang is landed.
#@inside_glslc_testsuite('OptionStd')
class TestSpaceAroundVersion(expect.ValidObjectFileWithWarning):
    """Tests that space around #version is correctly handled."""

    shader = FileShader(
        '\t   \t  # \t \f\f version  \v \t\t  310 \v\v \t  es \n' +
        core_vert_shader_without_version(), '.vert')
    glslc_args = ['-c', '-std=450core', shader]

    expected_warning = [
        shader, ': warning: (version, profile) forced to be (450, core), while '
        'in source code it is (310, es)\n1 warning generated.\n']


@inside_glslc_testsuite('OptionStd')
class TestVersionInsideCrazyComment(expect.ValidObjectFileWithWarning):
    """Tests that -std substitutes the correct #version string."""

    # The fourth #version string should be substituted and this shader
    # should compile successfully with -std=450core.
    shader = FileShader(
        '/* */ /* // /* #version 310 es */\n' +  # /*-style comment
        '// /* */ /* /* // #version 310 es\n' +  # //-style comment
        '///*////*//*/*/ #version 310 es*/\n' +  # //-style comment
        '#version 310 es\n' + core_vert_shader_without_version(), '.vert')
    glslc_args = ['-c', '-std=450core', shader]

    expected_warning = [
        shader, ': warning: (version, profile) forced to be (450, core), while '
        'in source code it is (310, es)\n1 warning generated.\n']


@inside_glslc_testsuite('OptionStd')
class TestVersionMissingProfile(expect.ErrorMessage):
    """Tests that missing required profile in -std results in an error."""

    shader = FileShader('#version 140\nvoid main() {}', '.vert')
    glslc_args = ['-c', '-std=310', shader]

    expected_error = [
        shader, ': error: #version: versions 300, 310, and 320 require ',
        "specifying the 'es' profile\n1 error generated.\n"]


@inside_glslc_testsuite('OptionStd')
class TestVersionRedundantProfile(expect.ErrorMessageSubstr):
    """Tests that adding non-required profile in -std results in an error."""

    shader = FileShader('#version 140\nvoid main() {}', '.vert')
    glslc_args = ['-c', '-std=100core', shader]

    expected_error_substr = [
        shader, ': error: #version: versions before 150 do not allow '
        'a profile token\n']
