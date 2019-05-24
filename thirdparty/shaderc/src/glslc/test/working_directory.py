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

import os.path

import expect
from environment import File, Directory
from glslc_test_framework import inside_glslc_testsuite
from placeholder import FileShader

MINIMAL_SHADER = '#version 140\nvoid main() {}'

# @inside_glslc_testsuite('WorkDir')
class TestWorkDirNoArg(expect.ErrorMessage):
    """Tests -working-directory. Behavior cribbed from Clang."""

    glslc_args = ['-working-directory']
    expected_error = [
        "glslc: error: argument to '-working-directory' is missing "
        '(expected 1 value)\n',
        'glslc: error: no input files\n']


# @inside_glslc_testsuite('WorkDir')
class TestWorkDirEqNoArg(expect.ErrorMessage):
    """Tests -working-directory=<empty>. Behavior cribbed from Clang."""

    glslc_args = ['-working-directory=']
    expected_error = ['glslc: error: no input files\n']


EMPTY_SHADER_IN_SUBDIR = Directory(
    'subdir', [File('shader.vert', MINIMAL_SHADER)])

# @inside_glslc_testsuite('WorkDir')
class TestWorkDirEqNoArgCompileFile(expect.ValidNamedObjectFile):
    """Tests -working-directory=<empty> when compiling input file."""

    environment = Directory('.', [EMPTY_SHADER_IN_SUBDIR])
    glslc_args = ['-c', '-working-directory=', 'subdir/shader.vert']
    # Output file should be generated into subdir/.
    expected_object_filenames = ('subdir/shader.vert.spv',)


# @inside_glslc_testsuite('WorkDir')
class TestMultipleWorkDir(expect.ValidNamedObjectFile):
    """Tests that if there are multiple -working-directory=<dir> specified,
    only the last one takes effect."""

    environment = Directory('.', [EMPTY_SHADER_IN_SUBDIR])
    glslc_args = ['-c', '-working-directory=i-dont-exist',
                  '-working-directory', 'i-think/me-neither',
                  '-working-directory=subdir', 'shader.vert']
    # Output file should be generated into subdir/.
    expected_object_filenames = ('subdir/shader.vert.spv',)


# @inside_glslc_testsuite('WorkDir')
class TestWorkDirPosition(expect.ValidNamedObjectFile):
    """Tests that -working-directory=<dir> affects all files before and after
    it on the command line."""

    environment = Directory('subdir', [
        File('shader.vert', MINIMAL_SHADER),
        File('cool.frag', MINIMAL_SHADER),
        File('bla.vert', MINIMAL_SHADER)
    ])
    glslc_args = ['-c', 'shader.vert', 'bla.vert',
                  '-working-directory=subdir', 'cool.frag']
    # Output file should be generated into subdir/.
    expected_object_filenames = (
        'subdir/shader.vert.spv', 'subdir/cool.frag.spv', 'subdir/bla.vert.spv')


# @inside_glslc_testsuite('WorkDir')
class TestWorkDirDeepDir(expect.ValidNamedObjectFile):
    """Tests that -working-directory=<dir> works with directory hierarchies."""

    environment = Directory('subdir', [
        Directory('subsubdir', [
            File('one.vert', MINIMAL_SHADER),
            File('two.frag', MINIMAL_SHADER)
        ]),
        File('zero.vert', MINIMAL_SHADER)
    ])
    glslc_args = ['-c', 'zero.vert', 'subsubdir/one.vert',
                  'subsubdir/two.frag', '-working-directory=subdir']
    # Output file should be generated into subdir/.
    expected_object_filenames = (
        'subdir/zero.vert.spv', 'subdir/subsubdir/one.vert.spv',
        'subdir/subsubdir/two.frag.spv')


# @inside_glslc_testsuite('WorkDir')
class TestWorkDirCompileFile(expect.ValidNamedObjectFile):
    """Tests -working-directory=<dir> when compiling input file."""

    environment = Directory('.', [EMPTY_SHADER_IN_SUBDIR])
    glslc_args = ['-c', '-working-directory=subdir', 'shader.vert']
    # Output file should be generated into subdir/.
    expected_object_filenames = ('subdir/shader.vert.spv',)


# @inside_glslc_testsuite('WorkDir')
class TestWorkDirCompileFileOutput(expect.ValidNamedObjectFile):
    """Tests -working-directory=<dir> when compiling input file and specifying
    output filename."""

    environment = Directory('.', [
        Directory('subdir', [
            Directory('bin', []),
            File('shader.vert', MINIMAL_SHADER)
        ])
    ])
    glslc_args = ['-c', '-o', 'bin/spv', '-working-directory=subdir',
                  'shader.vert']
    # Output file should be generated into subdir/bin/.
    expected_object_filenames = ('subdir/bin/spv',)


# @inside_glslc_testsuite('WorkDir')
class TestWorkDirArgNoEq(expect.ValidNamedObjectFile):
    """Tests -working-directory <dir>."""

    environment = Directory('.', [EMPTY_SHADER_IN_SUBDIR])
    glslc_args = ['-working-directory', 'subdir', 'shader.vert']
    expected_object_filenames = ('a.spv',)


# @inside_glslc_testsuite('WorkDir')
class TestWorkDirEqInArg(expect.ValidNamedObjectFile):
    """Tests -working-directory=<dir-with-equal-sign-inside>."""

    environment = Directory('.', [
        Directory('=subdir', [File('shader.vert', MINIMAL_SHADER)]),
    ])
    glslc_args = ['-working-directory==subdir', 'shader.vert']
    expected_object_filenames = ('a.spv',)


# @inside_glslc_testsuite('WorkDir')
class TestWorkDirCompileFileAbsolutePath(expect.ValidObjectFile):
    """Tests -working-directory=<dir> when compiling input file with absolute
    path."""

    shader = FileShader(MINIMAL_SHADER, '.vert')
    glslc_args = ['-c', '-working-directory=subdir', shader]


# The -working-directory flag should not affect the placement of the link file.
# The following tests ensure that.

class WorkDirDoesntAffectLinkedFile(expect.ValidNamedObjectFile):
    """A base class for tests asserting that -working-directory has no impact
    on the location of the output link file.
    """
    environment = Directory('.', [
        Directory('subdir', [
            File('shader.vert', MINIMAL_SHADER),
            # Try to fake glslc into putting the linked file here, though it
            # shouldn't (because -working-directory doesn't impact -o).
            Directory('bin', [])]),
        File('shader.vert', "fake file, doesn't compile."),
        Directory('bin', [])])


# @inside_glslc_testsuite('WorkDir')
class TestWorkDirLinkFileDefaultLocation(WorkDirDoesntAffectLinkedFile):
    """Tests that -working-directory doesn't impact the default link-file
    location.
    """
    glslc_args = ['-working-directory=subdir', 'shader.vert']
    expected_object_filenames = ('a.spv',)


# @inside_glslc_testsuite('WorkDir')
class TestWorkDirLinkFileExplicit(WorkDirDoesntAffectLinkedFile):
    """Tests that -working-directory doesn't impact the named link-file
    location.
    """
    glslc_args = ['-o', 'b.spv', '-working-directory=subdir', 'shader.vert']
    expected_object_filenames = ('b.spv',)


# @inside_glslc_testsuite('WorkDir')
class TestWorkDirLinkFileInSubdir(WorkDirDoesntAffectLinkedFile):
    """Tests that -working-directory doesn't impact the link-file sent into an
    existing subdirectory.
    """
    glslc_args = ['-o', 'bin/spv', '-working-directory=subdir', 'shader.vert']
    expected_object_filenames = ('bin/spv',)


# @inside_glslc_testsuite('WorkDir')
class TestWorkDirLinkFileInvalidPath(expect.ErrorMessage):
    """Tests that -working-directory doesn't impact the error generated for an
    invalid -o path.
    """

    environment = Directory('.', [
        Directory('subdir', [
            File('shader.vert', MINIMAL_SHADER),
            Directory('missing', [])]),  # Present here, but missing in parent.
        File('shader.vert', "fake file, doesn't compile.")])

    glslc_args = [
        '-o', 'missing/spv', '-working-directory=subdir', 'shader.vert']

    expected_error = ['glslc: error: cannot open output file: ',
                      "'missing/spv': No such file or directory\n"]
