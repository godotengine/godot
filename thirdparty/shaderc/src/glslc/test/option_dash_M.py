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
from environment import File, Directory
from glslc_test_framework import inside_glslc_testsuite
from placeholder import FileShader
from glslc_test_framework import GlslCTest

MINIMAL_SHADER = '#version 140\nvoid main() {}'
EMPTY_SHADER_IN_CURDIR = Directory('.', [File('shader.vert', MINIMAL_SHADER)])
EMPTY_SHADER_IN_SUBDIR = Directory('subdir',
                                   [File('shader.vert', MINIMAL_SHADER)])


def process_test_specified_dependency_info_rules(test_specified_rules):
    """A helper function to process the expected dependency info rules
    specified in tests before checking the actual dependency rule output.

    This is required because the filename and path of temporary files created
    through FileShader is unknown at the time the expected dependency info rules
    are declared.

    Note this function process the given rule list in-place.
    """
    for rule in test_specified_rules:
        # If the 'target' value is not a hard-coded file name but a
        # FileShader, we need its full path, append extension to it and
        # strip the directory component from it to get the complete target
        # name.
        if isinstance(rule['target'], FileShader):
            rule['target'] = rule['target'].filename
        if 'target_extension' in rule:
            if rule['target_extension'] is not None:
                rule['target'] = rule['target'] + rule['target_extension']
            rule.pop('target_extension')
        rule['target'] = os.path.basename(rule['target'])

        # The dependency set may have FileShader too, we need to replace
        # them with their absolute paths.
        dependent_file_name_set = set()
        for dependent_file in rule['dependency']:
            if isinstance(dependent_file, FileShader):
                dependent_file_name_set.add(dependent_file.filename)
            else:
                dependent_file_name_set.add(dependent_file)
        rule['dependency'] = dependent_file_name_set


def parse_text_rules(text_lines):
    """ A helper function to read text lines and construct and returns a list of
    dependency rules which can be used for comparison.

    The list is built with the text order.  Each rule is described in the
    following way:
        {'target': <target name>, 'dependency': <set of dependent filenames>}
    """
    rules = []
    for line in text_lines:
        if line.strip() == "":
            continue
        rule = {'target': line.split(': ')[0].strip(),
                'dependency': set(line.split(': ')[-1].strip().split(' '))}
        rules.append(rule)
    return rules


class DependencyInfoStdoutMatch(GlslCTest):
    """Mixin class for tests that can expect dependency info in Stdout.

    To mix in this class, the subclass needs to provide
    dependency_rules_expected as a list of dictionaries, each dictionary
    describes one expected make rule for a target file. A expected rule should
    be specified in the following way:

        rule = {'target': <target name>,
                'target_extension': <.spv, .spvasm or None>,
                'dependency': <dependent file names>}

        The 'target_extension' field is optional, its value will be appended to
        'target' to get complete target name.

    And the list 'dependency_rules_expected' is a list of such rules and the
    order of the rules does matter.
    """

    def check_stdout_dependency_info(self, status):
        if not status.stdout:
            return False, 'Expect dependency rules on stdout'

        rules = parse_text_rules(status.stdout.decode('utf-8').split('\n'))

        process_test_specified_dependency_info_rules(
            self.dependency_rules_expected)

        if self.dependency_rules_expected != rules:
            return False, ('Incorrect dependency info:\n{ac_rules}\n'
                           'Expected:\n{ex_rules}\n'
                           'Stdout output:\n{ac_stdout}\n'.format(
                               ac_rules=rules,
                               ex_rules=self.dependency_rules_expected,
                               ac_stdout=status.stdout))
        return True, ''


@inside_glslc_testsuite('OptionsCapM')
class TestDashCapMSingleInputRelativePathNoInclude(DependencyInfoStdoutMatch):
    """Tests -M with single input file which doesn't contain #include and is
    represented in relative path.
    e.g. glslc -M shader.vert
      => shader.vert.spv: shader.vert
    """
    environment = EMPTY_SHADER_IN_CURDIR
    glslc_args = ['-M', 'shader.vert']
    dependency_rules_expected = [{'target': "shader.vert.spv",
                                  'dependency': {"shader.vert"}}]


@inside_glslc_testsuite('OptionsCapM')
class TestDashCapMSingleInputAbsolutePathNoInclude(DependencyInfoStdoutMatch):
    """Tests -M with single input file which doesn't contain #include and is
    represented in absolute path.
    e.g. glslc -M /usr/local/shader.vert
      => shader.vert.spv: /usr/local/shader.vert
    """
    shader = FileShader(MINIMAL_SHADER, '.vert')
    glslc_args = ['-M', shader]
    dependency_rules_expected = [{'target': shader,
                                  'target_extension': '.spv',
                                  'dependency': {shader}}]


@inside_glslc_testsuite('OptionsCapM')
class TestDashCapMSingleInputRelativePathWithInclude(
        DependencyInfoStdoutMatch):
    """Tests -M with single input file which does contain #include and is
    represented in relative path.
    e.g. glslc -M a.vert
      => a.vert.spv: a.vert b.vert
    """
    environment = Directory('.', [
        File('a.vert', '#version 140\n#include "b.vert"\nvoid main(){}\n'),
        File('b.vert', 'void foo(){}\n'),
    ])
    glslc_args = ['-M', 'a.vert']
    dependency_rules_expected = [{'target': 'a.vert.spv',
                                  'dependency': {'a.vert', 'b.vert'}}]


@inside_glslc_testsuite('OptionsCapM')
class TestDashCapMSingleInputRelativePathWithIncludeSubdir(
        DependencyInfoStdoutMatch):
    """Tests -M with single input file which does #include another file in a
    subdirectory of current directory and is represented in relative path.
    e.g. glslc -M a.vert
      => a.vert.spv: a.vert include/b.vert
    """
    environment = Directory('.', [
        File('a.vert', ('#version 140\n#include "include/b.vert"'
                        '\nvoid main(){}\n')),
        Directory('include', [File('b.vert', 'void foo(){}\n')]),
    ])
    glslc_args = ['-M', 'a.vert']
    dependency_rules_expected = [{'target': 'a.vert.spv',
                                  'dependency': {'a.vert', 'include/b.vert'}}]


@inside_glslc_testsuite('OptionsCapM')
class TestDashCapMSingleInputRelativePathWithDashI(DependencyInfoStdoutMatch):
    """Tests -M with single input file works with -I option. The #include
    directive does not specify 'include/' for the file to be include.
    e.g. glslc -M a.vert -I include
      => a.vert.spv: a.vert include/b.vert
    """
    environment = Directory('.', [
        File('a.vert', ('#version 140\n#include "b.vert"'
                        '\nvoid main(){}\n')),
        Directory('include', [File('b.vert', 'void foo(){}\n')]),
    ])
    glslc_args = ['-M', 'a.vert', '-I', 'include']
    dependency_rules_expected = [{'target': 'a.vert.spv',
                                  'dependency': {'a.vert', 'include/b.vert'}}]


@inside_glslc_testsuite('OptionsCapM')
class TestDashCapMSingleInputRelativePathWithNestedInclude(
        DependencyInfoStdoutMatch):
    """Tests -M with single input file under nested #include case. The input file
    is represented in relative path.
    e.g. glslc -M a.vert
      => a.vert.spv: a.vert b.vert c.vert
    """
    environment = Directory('.', [
        File('a.vert', '#version 140\n#include "b.vert"\nvoid main(){}\n'),
        File('b.vert', 'void foo(){}\n#include "c.vert"\n'),
        File('c.vert', 'void bar(){}\n'),
    ])
    glslc_args = ['-M', 'a.vert']

    dependency_rules_expected = [{'target': 'a.vert.spv',
                                  'dependency':
                                  {'a.vert', 'b.vert', 'c.vert'}}]


@inside_glslc_testsuite('OptionsCapM')
class TestDashCapMMultipleInputRelativePathNoInclude(
        DependencyInfoStdoutMatch):
    """Tests -M with multiple input file which don't contain #include and are
    represented in relative paths.
    e.g. glslc -M a.vert b.vert
      => a.vert.spv: a.vert
         b.vert.spv: b.vert
    """
    environment = Directory('.', [
        File('a.vert', MINIMAL_SHADER),
        File('b.vert', MINIMAL_SHADER),
    ])
    glslc_args = ['-M', 'a.vert', 'b.vert']

    dependency_rules_expected = [{'target': 'a.vert.spv',
                                  'dependency': {'a.vert'}},
                                 {'target': 'b.vert.spv',
                                  'dependency': {'b.vert'}}, ]


@inside_glslc_testsuite('OptionsCapM')
class TestDashCapMMultipleInputAbsolutePathNoInclude(
        DependencyInfoStdoutMatch):
    """Tests -M with single input file which doesn't contain #include and is
    represented in absolute path.
    e.g. glslc -M /usr/local/a.vert /usr/local/b.vert
      => a.vert.spv: /usr/local/a.vert
         b.vert.spv: /usr/local/b.vert
    """
    shader_a = FileShader(MINIMAL_SHADER, '.vert')
    shader_b = FileShader(MINIMAL_SHADER, '.vert')
    glslc_args = ['-M', shader_a, shader_b]

    dependency_rules_expected = [{'target': shader_a,
                                  'target_extension': '.spv',
                                  'dependency': {shader_a}},
                                 {'target': shader_b,
                                  'target_extension': '.spv',
                                  'dependency': {shader_b}}, ]


@inside_glslc_testsuite('OptionsCapM')
class TestDashCapMDashCapMT(DependencyInfoStdoutMatch):
    """Tests -MT works with -M. User can specify the target object name in the
    generated dependency info.
    e.g. glslc -M shader.vert -MT target
      => target: shader.vert
    """
    environment = EMPTY_SHADER_IN_CURDIR
    glslc_args = ['-M', 'shader.vert', '-MT', 'target']
    dependency_rules_expected = [{'target': 'target',
                                  'dependency': {'shader.vert'}}]


@inside_glslc_testsuite('OptionsCapM')
class TestDashCapMInputAbsolutePathWithInclude(DependencyInfoStdoutMatch):
    """Tests -M have included files represented in absolute paths when the input
    file is represented in absolute path.
    E.g. Assume a.vert has '#include "b.vert"'
         glslc -M /usr/local/a.vert
      => a.vert.spv: /usr/local/a.vert /usr/local/b.vert
    """
    environment = Directory('.', [File('b.vert', 'void foo(){}\n')])
    shader_main = FileShader(
        '#version 140\n#include "b.vert"\nvoid main(){}\n', '.vert')

    glslc_args = ['-M', shader_main]
    dependency_rules_expected = [{
        'target': shader_main,
        'target_extension': '.spv',
        'dependency': {shader_main}
        # The dependency here is not complete.  we can not get the absolute path
        # of b.vert here. It will be added in check_stdout_dependency_info()
    }]

    def check_stdout_dependency_info(self, status):
        # Add the absolute path of b.vert to the dependency set
        self.dependency_rules_expected[0]['dependency'].add(os.path.dirname(
            self.shader_main.filename) + '/b.vert')
        return DependencyInfoStdoutMatch.check_stdout_dependency_info(self,
                                                                      status)


@inside_glslc_testsuite('OptionsCapM')
class TestDashCapMSingleInputAbsolutePathWithIncludeSubdir(
        DependencyInfoStdoutMatch):
    """Tests -M with single input file which does #include another file in a
    subdirectory of current directory and is represented in absolute path.
    e.g. glslc -M /usr/local/a.vert
      => a.vert.spv: /usr/local/a.vert /usr/local/include/b.vert
    """
    environment = Directory('.', [
        Directory('include', [File('b.vert', 'void foo(){}\n')]),
    ])
    shader_main = FileShader('#version 140\n#include "include/b.vert"\n',
                             '.vert')
    glslc_args = ['-M', shader_main]
    dependency_rules_expected = [{
        'target': shader_main,
        'target_extension': '.spv',
        'dependency': {shader_main}
        # The dependency here is not complete.  we can not get the absolute
        # path of include/b.vert here. It will be added in
        # check_stdout_dependency_info()
    }]

    def check_stdout_dependency_info(self, status):
        # Add the absolute path of include/b.vert to the dependency set
        self.dependency_rules_expected[0]['dependency'].add(os.path.dirname(
            self.shader_main.filename) + '/include/b.vert')
        return DependencyInfoStdoutMatch.check_stdout_dependency_info(self,
                                                                      status)


@inside_glslc_testsuite('OptionsCapM')
class TestDashCapMOverridesOtherModes(DependencyInfoStdoutMatch):
    """Tests -M overrides other compiler mode options, includeing -E, -c and -S.
    """
    environment = Directory('.', [
        File('a.vert', MINIMAL_SHADER),
        File('b.vert', MINIMAL_SHADER),
    ])
    glslc_args = ['-M', '-E', '-c', '-S', 'a.vert', 'b.vert']
    dependency_rules_expected = [{'target': 'a.vert.spv',
                                  'dependency': {'a.vert'}},
                                 {'target': 'b.vert.spv',
                                  'dependency': {'b.vert'}}]


@inside_glslc_testsuite('OptionsCapM')
class TestDashCapMMEquivalentToCapM(DependencyInfoStdoutMatch):
    """Tests that -MM behaves as -M.
    e.g. glslc -MM shader.vert
      => shader.vert.spv: shader.vert
    """
    environment = EMPTY_SHADER_IN_CURDIR
    glslc_args = ['-MM', 'shader.vert']
    dependency_rules_expected = [{'target': 'shader.vert.spv',
                                  'dependency': {'shader.vert'}}]


@inside_glslc_testsuite('OptionsCapM')
class TestDashCapMImpliesDashCapE(DependencyInfoStdoutMatch,
                                  expect.NoOutputOnStderr):
    """Tests that -M implies -E, a .glsl file without an explict stage should
    not generate an error.
    e.g. glslc -M shader.glsl
      => shader.spv: shader.glsl
         <no error message should be generated>
    """
    environment = Directory('.', [File('shader.glsl', MINIMAL_SHADER)])
    glslc_args = ['-M', 'shader.glsl']
    dependency_rules_expected = [{'target': 'shader.spv',
                                  'dependency': {'shader.glsl'}}]


@inside_glslc_testsuite('OptionsCapM')
class TestDashCapMImpliesDashW(DependencyInfoStdoutMatch,
                               expect.NoOutputOnStderr):
    """Tests that -M implies -w, a deprecated attribute should not generate
    warning message.
    e.g. glslc -M shader.vert
      => shader.vert.spv: shader.vert
         <no warning message should be generated>
    """
    environment = Directory('.', [File(
            'shader.vert', """#version 400
               layout(location=0) attribute float x;
               void main() {}""")])
    glslc_args = ['-M', 'shader.vert']
    dependency_rules_expected = [{'target': 'shader.vert.spv',
                                  'dependency': {'shader.vert'}}]


@inside_glslc_testsuite('OptionsCapM')
class TestDashCapMMImpliesDashCapE(DependencyInfoStdoutMatch,
                                   expect.NoOutputOnStderr):
    """Tests that -M implies -E, a .glsl file without an explict stage should
    not generate an error.
    e.g. glslc -MM shader.glsl
      => shader.spv: shader.glsl
         <no error message should be generated>
    """
    environment = Directory('.', [File('shader.glsl', MINIMAL_SHADER)])
    glslc_args = ['-MM', 'shader.glsl']
    dependency_rules_expected = [{'target': 'shader.spv',
                                  'dependency': {'shader.glsl'}}]


@inside_glslc_testsuite('OptionsCapM')
class TestDashCapMMImpliesDashW(DependencyInfoStdoutMatch,
                                expect.NoOutputOnStderr):
    """Tests that -MM implies -w, a deprecated attribute should not generate
    warning message.
    e.g. glslc -MM shader.vert
      => shader.vert.spv: shader.vert
         <no warning message should be generated>
    """
    environment = Directory('.', [File(
        'shader.vert', """
           #version 400
           layout(location = 0) attribute float x;
           void main() {}""")])
    glslc_args = ['-MM', 'shader.vert']
    dependency_rules_expected = [{'target': 'shader.vert.spv',
                                  'dependency': {'shader.vert'}}]


@inside_glslc_testsuite('OptionsCapM')
class TestDashCapMD(expect.ValidFileContents, expect.ValidNamedObjectFile):
    """Tests that -MD generates dependency info file and compilation output.
    e.g. glslc -MD shader.vert
      => <a.spv: valid SPIR-V object file>
      => <shader.vert.spv.d: dependency info>
    """
    environment = EMPTY_SHADER_IN_CURDIR
    glslc_args = ['-MD', 'shader.vert']
    expected_object_filenames = ('a.spv', )
    target_filename = 'shader.vert.spv.d'
    expected_file_contents = ['shader.vert.spv: shader.vert\n']


class DependencyInfoFileMatch(GlslCTest):
    """Mixin class for tests that can expect dependency info files.

    To mix in this class, subclasses need to provide dependency_info_filenames
    and dependency_info_files_expected_contents which are two lists.
    list dependency_info_filenames contains the dependency info file names and
    list dependency_info_files_expected_contents contains the expected matching
    dependency rules.
    The item order of the two lists should match, which means:
        dependency_info_files_expected_contents[i] should describe the
        dependency rules saved in dependency_info_filenames[i]

    The content of each dependency info file is described in same 'list of dict'
    structure explained in class DependencyInfoStdoutMatch's doc string.
    """

    def check_dependency_info_files(self, status):
        dep_info_files = \
            [os.path.join(status.directory,
                          f) for f in self.dependency_info_filenames]
        for i, df in enumerate(dep_info_files):
            if not os.path.isfile(df):
                return False, 'Cannot find file: ' + df
            try:
                with open(df, 'r') as dff:
                    content = dff.read()
                    rules = parse_text_rules(content.split('\n'))

                    process_test_specified_dependency_info_rules(
                        self.dependency_info_files_expected_contents[i])

                    if self.dependency_info_files_expected_contents[
                            i] != rules:
                        return False, (
                            'Incorrect dependency info:\n{ac_rules}\n'
                            'Expected:\n{ex_rules}\n'
                            'Incorrect file output:\n{ac_out}\n'
                            'Incorrect dependency info file:\n{ac_file}\n'.format(
                                ac_rules=rules,
                                ex_rules=self.dependency_rules_expected,
                                ac_stdout=content,
                                ac_file=df))
            except IOError:
                return False, ('Could not open dependency info file ' + df +
                               ' for reading')
        return True, ''


@inside_glslc_testsuite('OptionsCapM')
class TestDashCapMWorksWithDashO(DependencyInfoFileMatch):
    """Tests -M works with -o option. When user specifies an output file name
    with -o, the dependency info should be dumped to the user specified output
    file.
    """
    environment = EMPTY_SHADER_IN_CURDIR
    glslc_args = ['-M', 'shader.vert', '-o', 'dep_info']
    dependency_info_filenames = ('dep_info', )
    dependency_info_files_expected_contents = []
    dependency_info_files_expected_contents.append(
        [{'target': 'shader.vert.spv',
          'dependency': {'shader.vert'}}])


@inside_glslc_testsuite('OptionsCapM')
class TestDashCapMDMultipleFile(expect.ValidNamedObjectFile,
                                DependencyInfoFileMatch):
    """Tests that -MD generates dependency info file for multiple files.
    e.g. glslc -MD a.vert b.vert -c
      => <a.vert.spv: valid SPIR-V object file>
      => <a.vert.spv.d: dependency info: "a.vert.spv: a.vert">
      => <b.vert.spv: valid SPIR-V object file>
      => <b.vert.spv.d: dependency info: "b.vert.spv: b.vert">
    """
    environment = Directory('.', [File('a.vert', MINIMAL_SHADER),
                                  File('b.vert', MINIMAL_SHADER)])
    glslc_args = ['-MD', 'a.vert', 'b.vert', '-c']
    expected_object_filenames = ('a.vert.spv', 'b.vert.spv', )

    dependency_info_filenames = ['a.vert.spv.d', 'b.vert.spv.d']
    dependency_info_files_expected_contents = []
    dependency_info_files_expected_contents.append([{'target': 'a.vert.spv',
                                                     'dependency': {'a.vert'}}
                                                    ])
    dependency_info_files_expected_contents.append([{'target': 'b.vert.spv',
                                                     'dependency': {'b.vert'}}
                                                    ])


@inside_glslc_testsuite('OptionsCapM')
class TestDashCapMDMultipleFilePreprocessingOnlyMode(expect.StdoutMatch,
                                                     DependencyInfoFileMatch):
    """Tests that -MD generates dependency info file for multiple files in
    preprocessing only mode.
    e.g. glslc -MD a.vert b.vert -E
      => stdout: preprocess result of a.vert and b.vert
      => <a.vert.spv.d: dependency info: "a.vert.spv: a.vert">
      => <b.vert.spv.d: dependency info: "b.vert.spv: b.vert">
    """
    environment = Directory('.', [File('a.vert', MINIMAL_SHADER),
                                  File('b.vert', MINIMAL_SHADER)])
    glslc_args = ['-MD', 'a.vert', 'b.vert', '-E']
    dependency_info_filenames = ['a.vert.spv.d', 'b.vert.spv.d']
    dependency_info_files_expected_contents = []
    dependency_info_files_expected_contents.append([{'target': 'a.vert.spv',
                                                     'dependency': {'a.vert'}}
                                                    ])
    dependency_info_files_expected_contents.append([{'target': 'b.vert.spv',
                                                     'dependency': {'b.vert'}}
                                                    ])
    expected_stdout = ("#version 140\nvoid main(){ }\n"
                       "#version 140\nvoid main(){ }\n")


@inside_glslc_testsuite('OptionsCapM')
class TestDashCapMDMultipleFileDisassemblyMode(expect.ValidNamedAssemblyFile,
                                               DependencyInfoFileMatch):
    """Tests that -MD generates dependency info file for multiple files in
    disassembly mode.
    e.g. glslc -MD a.vert b.vert -S
      => <a.vert.spvasm: valid SPIR-V assembly file>
      => <a.vert.spvasm.d: dependency info: "a.vert.spvasm: a.vert">
      => <b.vert.spvasm: valid SPIR-V assembly file>
      => <b.vert.spvasm.d: dependency info: "b.vert.spvasm: b.vert">
    """
    environment = Directory('.', [File('a.vert', MINIMAL_SHADER),
                                  File('b.vert', MINIMAL_SHADER)])
    glslc_args = ['-MD', 'a.vert', 'b.vert', '-S']
    expected_assembly_filenames = ('a.vert.spvasm', 'b.vert.spvasm', )
    dependency_info_filenames = ['a.vert.spvasm.d', 'b.vert.spvasm.d']

    dependency_info_files_expected_contents = []
    dependency_info_files_expected_contents.append([{'target': 'a.vert.spvasm',
                                                     'dependency': {'a.vert'}}
                                                    ])
    dependency_info_files_expected_contents.append([{'target': 'b.vert.spvasm',
                                                     'dependency': {'b.vert'}}
                                                    ])


@inside_glslc_testsuite('OptionsCapM')
class TestDashCapMT(expect.ValidFileContents, expect.ValidNamedObjectFile):
    """Tests that -MT generates dependency info file with specified target label.
    e.g. glslc -MD shader.vert -MT target_label
      => <a.spv: valid SPIR-V object file>
      => <shader.vert.spv.d: dependency info: "target_label: shader.vert">
    """
    environment = EMPTY_SHADER_IN_CURDIR
    glslc_args = ['-MD', 'shader.vert', '-MT', 'target_label']
    expected_object_filenames = ('a.spv', )
    target_filename = 'shader.vert.spv.d'
    expected_file_contents = ['target_label: shader.vert\n']


@inside_glslc_testsuite('OptionsCapM')
class TestDashCapMF(expect.ValidFileContents, expect.ValidNamedObjectFile):
    """Tests that -MF dumps dependency info into specified file.
    e.g. glslc -MD shader.vert -MF dep_file
      => <a.spv: valid SPIR-V object file>
      => <dep_file: dependency info: "shader.vert.spv: shader.vert">
    """
    environment = EMPTY_SHADER_IN_CURDIR
    glslc_args = ['-MD', 'shader.vert', '-MF', 'dep_file']
    expected_object_filenames = ('a.spv', )
    target_filename = 'dep_file'
    expected_file_contents = ['shader.vert.spv: shader.vert\n']


@inside_glslc_testsuite('OptionsCapM')
class TestDashCapMDSpecifyOutputFileName(expect.ValidFileContents,
                                         expect.ValidNamedObjectFile):
    """Tests that -MD has the default dependency info file name and target
    label correct when -o <output_file_name> appears in the command line.
    The default dependency info file name and target label should be deduced
    from the linking-disabled compilation output.
    e.g. glslc -MD subdir/shader.vert -c -o output
      => <./output: valid SPIR-V object file>
      => <./output.d: dependency info: "output: shader.vert">
    """
    environment = EMPTY_SHADER_IN_SUBDIR
    glslc_args = ['-MD', 'subdir/shader.vert', '-c', '-o', 'output']
    expected_object_filenames = ('output', )
    target_filename = 'output.d'
    expected_file_contents = ['output: subdir/shader.vert\n']


@inside_glslc_testsuite('OptionsCapM')
class TestDashCapMDWithDashMFDashMTDashO(expect.ValidFileContents,
                                         expect.ValidNamedObjectFile):
    """Tests that -MD, -MF, -MT and -o gernates dependency info file and
    compilation output file correctly
    e.g. glslc -MD subdir/shader.vert -c -o subdir/out -MF dep_info -MT label
      => <subdir/out: valid SPIR-V object file>
      => <dep_info: dependency info: "label: shader.vert">
    """
    environment = EMPTY_SHADER_IN_SUBDIR
    glslc_args = ['-MD', 'subdir/shader.vert', '-c', '-o', 'subdir/out', '-MF',
                  'dep_info', '-MT', 'label']
    expected_object_filenames = ('subdir/out', )
    target_filename = 'dep_info'
    expected_file_contents = ['label: subdir/shader.vert\n']


@inside_glslc_testsuite('OptionsCapM')
class TestDashCapMDWithDashMFDashMTDashODisassemblyMode(
        expect.ValidFileContents, expect.ValidNamedAssemblyFile):
    """Tests that -MD, -MF, -MT and -o gernates dependency info file and
    compilation output file correctly in disassembly mode
    e.g. glslc -MD subdir/shader.vert -s -o subdir/out -MF dep_info -MT label
      => <subdir/out: valid SPIR-V object file>
      => <dep_info: dependency info: "label: shader.vert">
    """
    environment = EMPTY_SHADER_IN_SUBDIR
    glslc_args = ['-MD', 'subdir/shader.vert', '-S', '-o', 'subdir/out', '-MF',
                  'dep_info', '-MT', 'label']
    expected_assembly_filenames = ('subdir/out', )
    target_filename = 'dep_info'
    expected_file_contents = ['label: subdir/shader.vert\n']


@inside_glslc_testsuite('OptionsCapM')
class TestErrorSetBothDashCapMAndDashCapMD(expect.StderrMatch):
    """Tests that when both -M (or -MM) and -MD are specified, glslc should exit
    with an error message complaining the case and neither dependency info
    output nor compilation output. This test has -MD before -M flag.
    """
    environment = EMPTY_SHADER_IN_CURDIR
    glslc_args = ['-MD', '-M', 'shader.vert']
    expected_stderr = ['glslc: error: both -M (or -MM) and -MD are specified. '
                       'Only one should be used at one time.\n']


@inside_glslc_testsuite('OptionsCapM')
class TestErrorSetBothDashCapMDAndDashCapM(expect.StderrMatch):
    """Tests that when both -M (or -MM) and -MD are specified, glslc should exit
    with an error message complaining the case and neither dependency info
    output nor compilation output. This test has -M before -MD flag.
    """
    environment = EMPTY_SHADER_IN_CURDIR
    glslc_args = ['-M', '-MD', 'shader.vert']
    expected_stderr = ['glslc: error: both -M (or -MM) and -MD are specified. '
                       'Only one should be used at one time.\n']


@inside_glslc_testsuite('OptionsCapM')
class TestErrorDashCapMFWithMultipleInputFiles(expect.StderrMatch):
    """Tests that when -MF option is specified, only one input file should be
    provided."""
    environment = Directory('.', [File('a.vert', MINIMAL_SHADER),
                                  File('b.vert', MINIMAL_SHADER)])
    glslc_args = ['-MD', 'a.vert', 'b.vert', '-c', '-MF', 'dep_info']
    expected_stderr = ['glslc: error: '
                       'to specify dependency info file name or dependency '
                       'info target, only one input file is allowed.\n']


@inside_glslc_testsuite('OptionsCapM')
class TestErrorDashCapMTWithMultipleInputFiles(expect.StderrMatch):
    """Tests that when -MT option is specified, only one input file should be
    provided."""
    environment = Directory('.', [File('a.vert', MINIMAL_SHADER),
                                  File('b.vert', MINIMAL_SHADER)])
    glslc_args = ['-M', 'a.vert', 'b.vert', '-c', '-MT', 'target']
    expected_stderr = ['glslc: error: '
                       'to specify dependency info file name or dependency '
                       'info target, only one input file is allowed.\n']


@inside_glslc_testsuite('OptionsCapM')
class TestErrorDashCapMFMissingDashMAndDashMD(expect.StderrMatch):
    """Tests that when only -MF is specified while -M and -MD are not specified,
    glslc should emit an error complaining that the user must specifiy either
    -M (-MM) or -MD to generate dependency info.
    """
    environment = EMPTY_SHADER_IN_CURDIR
    glslc_args = ['-MF', 'dep_info', 'shader.vert', '-c']
    expected_stderr = ['glslc: error: '
                       'to generate dependencies you must specify either -M '
                       '(-MM) or -MD\n']


@inside_glslc_testsuite('OptionsCapM')
class TestErrorDashCapMTMissingDashMAndMDWith(expect.StderrMatch):
    """Tests that when only -MF and -MT is specified while -M and -MD are not
    specified, glslc should emit an error complaining that the user must
    specifiy either -M (-MM) or -MD to generate dependency info.
    """
    environment = EMPTY_SHADER_IN_CURDIR
    glslc_args = ['-MF', 'dep_info', '-MT', 'target', 'shader.vert', '-c']
    expected_stderr = ['glslc: error: '
                       'to generate dependencies you must specify either -M '
                       '(-MM) or -MD\n']


@inside_glslc_testsuite('OptionsCapM')
class TestErrorMissingDependencyInfoFileName(expect.StderrMatch):
    """Tests that dependency file name is missing when -MF is specified."""
    environment = EMPTY_SHADER_IN_CURDIR
    glslc_args = ['target', 'shader.vert', '-c', '-MF']
    expected_stderr = ['glslc: error: '
                       'missing dependency info filename after \'-MF\'\n']


@inside_glslc_testsuite('OptionsCapM')
class TestErrorMissingDependencyTargetName(expect.StderrMatch):
    """Tests that dependency target name is missing when -MT is specified."""
    environment = EMPTY_SHADER_IN_CURDIR
    glslc_args = ['target', 'shader.vert', '-c', '-MT']
    expected_stderr = ['glslc: error: '
                       'missing dependency info target after \'-MT\'\n']
